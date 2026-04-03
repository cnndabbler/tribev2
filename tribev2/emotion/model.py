# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""EmotionEncoder: a classification head on top of tribev2's feature pipeline.

Reuses the multi-modal feature aggregation and transformer backbone from
:class:`FmriEncoderModel` but replaces the per-subject fMRI prediction head
with a simple linear classification layer.
"""

import logging
import typing as tp

import torch
from einops import rearrange
from neuralset.dataloader import SegmentData
from neuraltrain.models.base import BaseModelConfig
from neuraltrain.models.common import Mlp, SubjectLayersModel
from neuraltrain.models.transformer import TransformerEncoder
from torch import nn

from ..model import TemporalSmoothing

logger = logging.getLogger(__name__)


class EmotionEncoder(BaseModelConfig):
    """Config for the emotion classification encoder.

    This mirrors :class:`FmriEncoder` but drops fMRI-specific options
    (``subject_layers``, ``subject_embedding``, ``low_rank_head``) and adds
    ``num_classes`` for the classification head.
    """

    # architecture
    projector: BaseModelConfig = Mlp(norm_layer="layer", activation_layer="gelu")
    combiner: Mlp | None = Mlp(norm_layer="layer", activation_layer="gelu")
    encoder: TransformerEncoder | None = TransformerEncoder()
    # other hyperparameters
    time_pos_embedding: bool = True
    hidden: int = 256
    max_seq_len: int = 1024
    dropout: float = 0.0
    extractor_aggregation: tp.Literal["stack", "sum", "cat"] = "cat"
    layer_aggregation: tp.Literal["mean", "cat"] = "cat"
    linear_baseline: bool = False
    modality_dropout: float = 0.0
    temporal_dropout: float = 0.0
    temporal_smoothing: TemporalSmoothing | None = None
    # classification
    num_classes: int = 4

    def model_post_init(self, __context: tp.Any) -> None:
        if self.encoder is not None:
            for key in ["attn_dropout", "ff_dropout", "layer_dropout"]:
                setattr(self.encoder, key, self.dropout)
        if hasattr(self.projector, "dropout"):
            self.projector.dropout = self.dropout
        return super().model_post_init(__context)

    def build(
        self, feature_dims: dict[int], n_outputs: int, n_output_timesteps: int
    ) -> nn.Module:
        return EmotionEncoderModel(
            feature_dims,
            n_outputs,
            n_output_timesteps,
            config=self,
        )


class EmotionEncoderModel(nn.Module):
    """Classification model that reuses the multi-modal aggregation / transformer
    backbone from FmriEncoderModel but outputs ``(B, num_classes)`` logits.

    Key differences from :class:`FmriEncoderModel`:
    * No ``SubjectLayers`` predictor -- uses a plain ``nn.Linear`` head.
    * No ``pooler`` -- uses mean pooling over the time axis before the head.
    * ``forward()`` returns logits of shape ``(B, num_classes)``.
    """

    def __init__(
        self,
        feature_dims: dict[str, tuple[int, int]],
        n_outputs: int,
        n_output_timesteps: int,
        config: EmotionEncoder,
    ):
        super().__init__()
        self.config = config
        self.feature_dims = feature_dims
        self.n_outputs = n_outputs
        self.n_output_timesteps = n_output_timesteps
        hidden = config.hidden

        # --- Per-modality projectors (identical to FmriEncoderModel) ---
        self.projectors = nn.ModuleDict()
        for modality, tup in feature_dims.items():
            if tup is None:
                logger.warning(
                    "%s has no feature dimensions. Skipping projector.", modality
                )
                continue
            num_layers, feature_dim = tup
            input_dim = (
                feature_dim * num_layers
                if config.layer_aggregation == "cat"
                else feature_dim
            )
            output_dim = (
                hidden // len(feature_dims)
                if config.extractor_aggregation == "cat"
                else hidden
            )
            self.projectors[modality] = config.projector.build(input_dim, output_dim)

        # --- Combiner (identical to FmriEncoderModel) ---
        input_dim = (
            (hidden // len(feature_dims)) * len(feature_dims)
            if config.extractor_aggregation == "cat"
            else hidden
        )
        if config.combiner is not None:
            self.combiner = config.combiner.build(input_dim, hidden)
        else:
            assert (
                hidden % len(feature_dims) == 0
            ), "hidden must be divisible by the number of modalities if there is no combiner"
            self.combiner = nn.Identity()

        # --- Temporal smoothing ---
        if config.temporal_smoothing is not None:
            self.temporal_smoothing = config.temporal_smoothing.build(dim=hidden)

        # --- Transformer encoder ---
        if not config.linear_baseline:
            if config.time_pos_embedding:
                self.time_pos_embed = nn.Parameter(
                    torch.randn(1, config.max_seq_len, hidden)
                )
            self.encoder = config.encoder.build(dim=hidden)

        # --- Classification head (replaces SubjectLayers + pooler) ---
        self.classifier = nn.Linear(hidden, config.num_classes)

    # ------------------------------------------------------------------
    # Shared helpers -- same logic as FmriEncoderModel
    # ------------------------------------------------------------------

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def aggregate_features(self, batch: SegmentData) -> torch.Tensor:
        """Aggregate multi-modal features into a single tensor of shape ``(B, T, H)``."""
        tensors = []
        # Determine B, T from the first available modality
        for modality in batch.data.keys():
            if modality in self.feature_dims:
                break
        x = batch.data[modality]
        B, T = x.shape[0], x.shape[-1]
        for modality in self.feature_dims.keys():
            if modality not in self.projectors or modality not in batch.data:
                data = torch.zeros(
                    B, T, self.config.hidden // len(self.feature_dims)
                ).to(x.device)
            else:
                data = batch.data[modality]  # B, L, H, T
                data = data.to(torch.float32)
                if data.ndim == 3:
                    data = data.unsqueeze(1)
                # Aggregate over layers
                if self.config.layer_aggregation == "mean":
                    data = data.mean(dim=1)
                elif self.config.layer_aggregation == "cat":
                    data = rearrange(data, "b l d t -> b (l d) t")
                data = data.transpose(1, 2)
                assert data.ndim == 3  # B, T, D
                if isinstance(self.projectors[modality], SubjectLayersModel):
                    data = self.projectors[modality](
                        data.transpose(1, 2), batch.data["subject_id"]
                    ).transpose(1, 2)
                else:
                    data = self.projectors[modality](data)  # B, T, H
                if self.config.modality_dropout > 0 and self.training:
                    mask = torch.rand(data.shape[0]) < self.config.modality_dropout
                    data[mask, :] = torch.zeros_like(data[mask, :])
            tensors.append(data)
        if self.config.extractor_aggregation == "stack":
            out = torch.cat(tensors, dim=1)
        elif self.config.extractor_aggregation == "cat":
            out = torch.cat(tensors, dim=-1)
        elif self.config.extractor_aggregation == "sum":
            out = sum(tensors)
        if self.config.temporal_dropout > 0 and self.training:
            for batch_idx in range(out.shape[0]):
                mask = torch.rand(out.shape[1]) < self.config.temporal_dropout
                out[batch_idx, mask, :] = torch.zeros_like(out[batch_idx, mask, :])
        return out

    def transformer_forward(
        self, x: torch.Tensor, subject_id: tp.Any = None
    ) -> torch.Tensor:
        """Apply combiner, positional embedding, and transformer encoder."""
        x = self.combiner(x)
        if hasattr(self, "time_pos_embed"):
            x = x + self.time_pos_embed[:, : x.size(1)]
        x = self.encoder(x)
        return x

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, batch: SegmentData) -> torch.Tensor:
        """Return classification logits of shape ``(B, num_classes)``.

        Pipeline:
        1. Aggregate multi-modal features -> ``(B, T, H)``
        2. (optional) temporal smoothing
        3. Transformer encoder
        4. Mean-pool over time -> ``(B, H)``
        5. Linear head -> ``(B, num_classes)``
        """
        x = self.aggregate_features(batch)  # B, T, H
        if hasattr(self, "temporal_smoothing"):
            x = self.temporal_smoothing(x.transpose(1, 2)).transpose(1, 2)
        if not self.config.linear_baseline:
            x = self.transformer_forward(x)
        # Mean pool over time dimension
        x = x.mean(dim=1)  # B, H
        logits = self.classifier(x)  # B, num_classes
        return logits
