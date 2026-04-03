# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""ClassificationExperiment: adapts TribeExperiment for emotion classification.

Key differences from :class:`TribeExperiment`:
* ``model_post_init`` skips subject-layer sizing (no ``subject_layers`` in EmotionEncoder).
* ``_setup_trainer`` determines ``n_outputs`` from ``data.num_classes`` instead of fMRI shape.
* ``_init_module`` creates a :class:`ClassificationModule` instead of :class:`BrainModule`.
"""

import logging
import typing as tp
from pathlib import Path

import numpy as np
import pydantic
import torch
import yaml
from exca import ConfDict, TaskInfra
from neuraltrain.losses import BaseLoss
from neuraltrain.metrics import BaseMetric
from neuraltrain.models import BaseModelConfig
from neuraltrain.optimizers.base import BaseOptimizer
from neuraltrain.utils import BaseExperiment, WandbLoggerConfig
from torch import nn
from torch.utils.data import DataLoader

from .data import ClassificationData
from .pl_module import ClassificationModule

LOGGER = logging.getLogger(__name__)


class ClassificationExperiment(BaseExperiment):
    """Experiment pipeline for emotion classification."""

    model_config = pydantic.ConfigDict(extra="forbid")

    data: ClassificationData
    seed: int | None = 42
    brain_model_config: BaseModelConfig
    loss: BaseLoss
    optim: BaseOptimizer
    metrics: list[BaseMetric]
    monitor: str = "val/accuracy"
    wandb_config: WandbLoggerConfig | None = None
    accelerator: str = "gpu"
    n_epochs: int | None = 30
    max_steps: int = -1
    patience: int | None = None
    limit_train_batches: int | None = None
    accumulate_grad_batches: int = 1
    enable_progress_bar: bool = True
    log_every_n_steps: int | None = None
    fast_dev_run: bool = False
    save_checkpoints: bool = True
    checkpoint_filename: str = "best"
    checkpoint_path: str | None = None
    load_checkpoint: bool = True
    test_only: bool = False

    _trainer: tp.Any = None
    _model: tp.Any = None
    _logger: tp.Any = None

    infra: TaskInfra = TaskInfra(version="1")

    def model_post_init(self, __context: tp.Any) -> None:
        super().model_post_init(__context)
        if self.infra.folder is None:
            raise ValueError("infra.folder needs to be specified to save the results.")
        self.infra.tasks_per_node = self.infra.gpus_per_node
        self.infra.slurm_use_srun = self.infra.gpus_per_node > 1
        if self.infra.gpus_per_node > 1:
            self.data.batch_size = self.data.batch_size // self.infra.gpus_per_node
        if self.accumulate_grad_batches > 1:
            self.data.batch_size = self.data.batch_size // self.accumulate_grad_batches

    def _get_checkpoint_path(self) -> Path | None:
        if self.checkpoint_path:
            p = Path(self.checkpoint_path)
            assert p.exists(), f"Checkpoint path {p} does not exist."
            return p
        p = Path(self.infra.folder) / "last.ckpt"
        return p if p.exists() else None

    def _init_module(self, model: nn.Module) -> ClassificationModule:
        checkpoint_path = self._get_checkpoint_path()
        if self.load_checkpoint and checkpoint_path is not None:
            LOGGER.info("Loading model from %s", checkpoint_path)
            init_fn = ClassificationModule.load_from_checkpoint
            init_kwargs: dict[str, tp.Any] = {
                "checkpoint_path": checkpoint_path,
                "strict": False,
            }
        else:
            init_fn = ClassificationModule
            init_kwargs = {}

        metrics = {
            split + "/" + metric.log_name: metric.build()
            for metric in self.metrics
            for split in ["val", "test"]
        }
        return init_fn(
            model=model,
            loss=self.loss.build(),
            optim_config=self.optim,
            metrics=nn.ModuleDict(metrics),
            config=ConfDict(self.model_dump()),
            **init_kwargs,
        )

    def _setup_trainer(
        self, train_loader: DataLoader, override_n_devices: int | None = None
    ) -> tp.Any:
        import lightning.pytorch as pl
        from lightning.pytorch.callbacks import (
            EarlyStopping,
            LearningRateMonitor,
            ModelCheckpoint,
        )

        batch = next(iter(train_loader))

        # Determine feature dimensions from the batch
        feature_dims: dict[str, tuple[int, int] | None] = {}
        for modality in self.data.features_to_use:
            if modality in batch.data and modality not in self.data.features_to_mask:
                if batch.data[modality].ndim == 4:
                    feature_dims[modality] = (
                        batch.data[modality].shape[1],
                        batch.data[modality].shape[2],
                    )
                elif batch.data[modality].ndim == 3:
                    feature_dims[modality] = (1, batch.data[modality].shape[1])
                else:
                    raise ValueError(
                        f"Unexpected ndim for {modality}: {batch.data[modality].ndim}"
                    )
            else:
                feature_dims[modality] = None

        # For classification: n_outputs = num_classes
        n_outputs = self.data.num_classes

        brain_model = self.brain_model_config.build(
            feature_dims=feature_dims,
            n_outputs=n_outputs,
            n_output_timesteps=self.data.duration_trs,
        )
        LOGGER.info("Extractor dims: %s", feature_dims)
        LOGGER.info("Num classes: %s", n_outputs)
        _ = brain_model(batch)
        total_params = sum(p.numel() for p in brain_model.parameters())
        LOGGER.info("Total parameters: %d", total_params)

        self._model = self._init_module(brain_model)

        mode = "max" if "accuracy" in self.monitor or "f1" in self.monitor else "min"
        callbacks = [LearningRateMonitor(logging_interval="epoch")]
        if self.patience is not None:
            callbacks.append(
                EarlyStopping(monitor=self.monitor, mode=mode, patience=self.patience)
            )
        if self.save_checkpoints:
            callbacks.append(
                ModelCheckpoint(
                    save_last=True,
                    save_top_k=1,
                    dirpath=self.infra.folder,
                    filename=self.checkpoint_filename,
                    monitor=self.monitor,
                    mode=mode,
                    save_on_train_epoch_end=True,
                )
            )

        trainer = pl.Trainer(
            strategy="auto" if self.infra.gpus_per_node == 1 else "fsdp",
            devices=override_n_devices or self.infra.gpus_per_node,
            accelerator=self.accelerator,
            max_epochs=self.n_epochs,
            max_steps=self.max_steps,
            limit_train_batches=self.limit_train_batches,
            enable_progress_bar=self.enable_progress_bar,
            log_every_n_steps=self.log_every_n_steps,
            fast_dev_run=self.fast_dev_run,
            callbacks=callbacks,
            logger=self._logger,
            enable_checkpointing=self.save_checkpoints,
            accumulate_grad_batches=self.accumulate_grad_batches,
        )
        self._trainer = trainer
        return trainer

    def fit(self, train_loader: DataLoader, valid_loader: DataLoader) -> None:
        self._trainer.fit(
            model=self._model,
            train_dataloaders=train_loader,
            val_dataloaders=valid_loader,
            ckpt_path=self._get_checkpoint_path(),
        )

    def test(self, test_loader: DataLoader) -> None:
        ckpt_path = None
        if self.checkpoint_path:
            ckpt_path = self.checkpoint_path
        elif self.save_checkpoints:
            best = Path(self.infra.folder) / "best.ckpt"
            if best.exists():
                ckpt_path = str(best)
        self._trainer.test(self._model, dataloaders=test_loader, ckpt_path=ckpt_path)

    def setup_run(self) -> None:
        import os

        os.makedirs(self.infra.folder, exist_ok=True)
        config_path = Path(self.infra.folder) / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(
                self.model_dump(), f, indent=4, default_flow_style=False, sort_keys=False
            )

    @infra.apply
    def run(self):
        import lightning.pytorch as pl

        self.setup_run()
        if self.wandb_config:
            self._logger = self.wandb_config.build(
                save_dir=self.infra.folder,
                xp_config=self.model_dump(),
                id=f"{self.wandb_config.group}-{self.infra.uid().split('-')[-1]}",
            )
        else:
            from lightning.pytorch.loggers import TensorBoardLogger
            self._logger = TensorBoardLogger(
                save_dir=self.infra.folder,
                name="tensorboard",
                default_hp_metric=False,
            )

        if self.seed is not None:
            pl.seed_everything(self.seed, workers=True)
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)

        loaders = self.data.get_loaders(
            split_to_build="val" if self.test_only else None
        )
        self._setup_trainer(next(iter(loaders.values())))

        if not self.test_only:
            self.fit(loaders["train"], loaders["val"])

        self.test(loaders["val"])
