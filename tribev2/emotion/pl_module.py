# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""ClassificationModule: Lightning module for emotion classification.

Replaces :class:`BrainModule` which hard-codes ``batch.data["fmri"]`` as the
prediction target.  Here the target comes from ``batch.data["emotion_label"]``
and the model output is ``(B, num_classes)`` logits.
"""

import typing as tp
from pathlib import Path

import lightning.pytorch as pl
from neuralset.dataloader import SegmentData
from neuraltrain.optimizers import BaseOptimizer
from torch import nn
from torchmetrics import Metric


class ClassificationModule(pl.LightningModule):
    """Lightning module for emotion classification training."""

    def __init__(
        self,
        model: nn.Module,
        loss: nn.Module,
        optim_config: BaseOptimizer,
        metrics: dict[str, Metric],
        checkpoint_path: Path | None = None,
        config: dict[str, tp.Any] | None = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.checkpoint_path = checkpoint_path
        self.config = config
        self.optim_config = optim_config
        self.loss = loss
        self.metrics = metrics

    def forward(self, batch: SegmentData):
        return self.model(batch)

    def on_save_checkpoint(self, checkpoint):
        checkpoint["model_build_args"] = {
            "feature_dims": self.model.feature_dims,
            "n_outputs": self.model.n_outputs,
            "n_output_timesteps": self.model.n_output_timesteps,
        }

    def _run_step(
        self, batch: SegmentData, batch_idx: int, step_name: str
    ):
        # Target: emotion label (integer class index) from LabelEncoder
        y_true = batch.data["emotion_label"]  # (B,) or (B, 1)
        if y_true.ndim > 1:
            y_true = y_true.squeeze(-1)
        y_true = y_true.long()

        # Prediction: (B, num_classes) logits
        y_pred = self.forward(batch)

        loss = self.loss(y_pred, y_true)
        log_kwargs = {
            "on_step": step_name == "train",
            "on_epoch": True,
            "logger": True,
            "prog_bar": True,
            "batch_size": y_pred.shape[0],
        }

        self.log(f"{step_name}/loss", loss, **log_kwargs)

        for metric_name, metric in self.metrics.items():
            if metric_name.startswith(step_name):
                metric.update(y_pred, y_true)
                self.log(metric_name, metric, **log_kwargs)

        return loss, y_pred.detach().cpu(), y_true.detach().cpu()

    def training_step(self, batch: SegmentData, batch_idx: int):
        loss, _, _ = self._run_step(batch, batch_idx, step_name="train")
        return loss

    def validation_step(self, batch: SegmentData, batch_idx: int, dataloader_idx: int = 0):
        _, y_pred, y_true = self._run_step(batch, batch_idx, step_name="val")
        return y_pred, y_true

    def test_step(self, batch: SegmentData, batch_idx: int, dataloader_idx: int = 0):
        _, y_pred, y_true = self._run_step(batch, batch_idx, step_name="test")
        return y_pred, y_true

    def configure_optimizers(self):
        optim_config = self.optim_config.copy()
        unfrozen_params = [p for p in self.parameters() if p.requires_grad]
        if self.config["max_steps"] > 0:
            total_steps = self.config["max_steps"]
        else:
            total_steps = self.trainer.estimated_stepping_batches
        return optim_config.build(unfrozen_params, total_steps=total_steps)
