# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Export a trained EmotionEncoder checkpoint to ONNX format.

This script loads a ``ClassificationModule`` checkpoint, reconstructs the
:class:`EmotionEncoderModel`, wraps it in an export-friendly module, and
writes an ONNX graph suitable for inference in the SER sidecar service.

**Deployment note** -- The exported ONNX file contains only the
classification head (projectors + transformer + linear classifier).  It
does **not** include the Wav2VecBert feature extractor, which is far too
large for practical ONNX export.  In the SER sidecar the intended
architecture is:

    raw audio  -->  [PyTorch Wav2VecBert backbone]  -->  features
    features   -->  [ONNX classification head]      -->  logits

The sidecar runs the backbone in PyTorch (or via a separate optimised
runtime) and feeds the resulting feature tensor into the ONNX head.

Usage
-----
::

    uv run python -m tribev2.emotion.export_onnx \\
        --checkpoint /path/to/results/tribe_emotion/best.ckpt \\
        --output /path/to/care-whisper-ser/models/emotion_classifier_head.onnx \\
        --quantize-fp16
"""

import argparse
import logging
import sys
from pathlib import Path

import torch
from torch import nn

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Export wrapper
# ---------------------------------------------------------------------------

class ExportWrapper(nn.Module):
    """Thin wrapper that presents a plain-tensor interface for ONNX export.

    The underlying :class:`EmotionEncoderModel` expects a ``SegmentData``
    batch object whose ``.data`` dict carries the modality tensors.  This
    wrapper accepts a single ``audio_features`` tensor and constructs the
    minimal batch structure required by :meth:`EmotionEncoderModel.forward`.

    Parameters
    ----------
    model : nn.Module
        A fully-initialised :class:`EmotionEncoderModel`.
    modality : str
        Key under which the audio features are stored in
        ``model.feature_dims`` (typically ``"audio"``).
    """

    def __init__(self, model: nn.Module, modality: str = "audio") -> None:
        super().__init__()
        self.model = model
        self.modality = modality

    def forward(self, audio_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the classification head.

        Parameters
        ----------
        audio_features : Tensor
            Shape ``(B, num_layers, feature_dim, time_steps)`` -- the
            pre-extracted Wav2VecBert hidden states for an audio segment.

        Returns
        -------
        logits : Tensor
            Shape ``(B, num_classes)``.
        probabilities : Tensor
            Shape ``(B, num_classes)`` -- softmax over ``logits``.
        """
        # Build a minimal mock object that behaves like SegmentData for the
        # model's aggregate_features / forward path.
        batch = _MockBatch({self.modality: audio_features})
        logits = self.model(batch)
        probabilities = torch.softmax(logits, dim=-1)
        return logits, probabilities


class _MockBatch:
    """Mimics the ``SegmentData`` interface (only ``.data`` is needed)."""

    def __init__(self, data: dict[str, torch.Tensor]) -> None:
        self.data = data


# ---------------------------------------------------------------------------
# Checkpoint loading helpers
# ---------------------------------------------------------------------------

def _load_checkpoint(checkpoint_path: Path) -> dict:
    """Load a Lightning checkpoint from *checkpoint_path*.

    Accepts either a direct ``.ckpt`` file or a directory containing one.
    """
    path = Path(checkpoint_path)
    if path.is_dir():
        # Try common checkpoint file names inside the directory.
        for name in ("best.ckpt", "last.ckpt"):
            candidate = path / name
            if candidate.exists():
                path = candidate
                break
        else:
            ckpt_files = sorted(path.glob("*.ckpt"))
            if not ckpt_files:
                raise FileNotFoundError(
                    f"No .ckpt files found in directory: {path}"
                )
            path = ckpt_files[0]
            logger.warning("Using first checkpoint found: %s", path)

    logger.info("Loading checkpoint from %s", path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    return checkpoint


def _strip_prefix(state_dict: dict[str, torch.Tensor], prefix: str = "model.") -> dict[str, torch.Tensor]:
    """Remove a key prefix (e.g. ``model.``) from a state dict."""
    new_sd = {}
    for key, value in state_dict.items():
        if key.startswith(prefix):
            new_sd[key[len(prefix):]] = value
        else:
            new_sd[key] = value
    return new_sd


def reconstruct_model(checkpoint: dict, checkpoint_path: Path) -> nn.Module:
    """Rebuild the :class:`EmotionEncoderModel` from checkpoint metadata.

    The ``ClassificationModule.on_save_checkpoint`` stores
    ``model_build_args`` containing ``feature_dims``, ``n_outputs``, and
    ``n_output_timesteps``.  We also read the training config from the
    co-located ``config.yaml`` to ensure the architecture matches.
    """
    import yaml

    from .model import EmotionEncoder

    if "model_build_args" not in checkpoint:
        raise KeyError(
            "Checkpoint does not contain 'model_build_args'. "
            "Was it saved by ClassificationModule?"
        )

    build_args = checkpoint["model_build_args"]
    logger.info("Model build args: %s", build_args)

    # Try to load config.yaml from the same directory as the checkpoint
    ckpt_dir = checkpoint_path.parent if checkpoint_path.is_file() else checkpoint_path
    config_yaml = ckpt_dir / "config.yaml"
    if config_yaml.exists():
        with open(config_yaml) as f:
            saved_config = yaml.safe_load(f)
        model_config_dict = saved_config.get("brain_model_config", {})
        logger.info("Loaded model config from %s: %s", config_yaml, model_config_dict)
        config = EmotionEncoder.model_validate(model_config_dict)
    else:
        logger.warning("No config.yaml found at %s, using defaults", ckpt_dir)
        config = EmotionEncoder()

    model = config.build(**build_args)

    # Load weights -- checkpoint keys are prefixed with "model." by Lightning.
    raw_sd = checkpoint.get("state_dict", {})
    sd = _strip_prefix(raw_sd, prefix="model.")
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        logger.warning("Missing keys when loading state dict: %s", missing)
    if unexpected:
        logger.warning("Unexpected keys when loading state dict: %s", unexpected)

    model.eval()
    return model


# ---------------------------------------------------------------------------
# ONNX export
# ---------------------------------------------------------------------------

def export_onnx(
    model: nn.Module,
    output_path: Path,
    feature_dims: dict[str, tuple[int, int]],
    opset_version: int = 17,
) -> Path:
    """Export the classification head to ONNX.

    Parameters
    ----------
    model : nn.Module
        The reconstructed :class:`EmotionEncoderModel`.
    output_path : Path
        Destination ``.onnx`` file.
    feature_dims : dict
        Mapping ``{modality: (num_layers, feature_dim)}``.  Used to build a
        dummy input tensor of the correct shape.
    opset_version : int
        ONNX opset version (default 17).

    Returns
    -------
    Path
        The path to the written ONNX file.
    """
    # Determine the audio modality key and its dimensions.
    modality = None
    for key in feature_dims:
        if feature_dims[key] is not None:
            modality = key
            break
    if modality is None:
        raise ValueError("No non-None modality found in feature_dims")

    num_layers, feature_dim = feature_dims[modality]
    logger.info(
        "Exporting modality=%s  num_layers=%d  feature_dim=%d",
        modality, num_layers, feature_dim,
    )

    wrapper = ExportWrapper(model, modality=modality)
    wrapper.eval()

    # Dummy input: (batch=1, num_layers, feature_dim, time_steps=16)
    dummy_time_steps = 16
    dummy_input = torch.randn(1, num_layers, feature_dim, dummy_time_steps)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        wrapper,
        (dummy_input,),
        str(output_path),
        input_names=["audio_features"],
        output_names=["logits", "probabilities"],
        dynamic_axes={
            "audio_features": {0: "batch_size", 3: "time_steps"},
            "logits": {0: "batch_size"},
            "probabilities": {0: "batch_size"},
        },
        opset_version=opset_version,
    )

    logger.info("ONNX model saved to %s", output_path)
    return output_path


# ---------------------------------------------------------------------------
# Optional FP16 quantisation
# ---------------------------------------------------------------------------

def quantize_fp16(input_path: Path, output_path: Path | None = None) -> Path:
    """Convert an ONNX model to float16 weights.

    Requires the ``onnx`` and ``onnxconverter-common`` packages.

    Parameters
    ----------
    input_path : Path
        Path to the full-precision ONNX file.
    output_path : Path or None
        Where to write the quantised model.  Defaults to replacing the
        ``.onnx`` suffix with ``_fp16.onnx``.

    Returns
    -------
    Path
        The path to the quantised ONNX file.
    """
    import onnx
    from onnxconverter_common import float16

    if output_path is None:
        output_path = input_path.with_name(
            input_path.stem + "_fp16" + input_path.suffix
        )

    logger.info("Quantising %s -> %s", input_path, output_path)
    model = onnx.load(str(input_path))
    model_fp16 = float16.convert_float_to_float16(model)
    onnx.save(model_fp16, str(output_path))
    logger.info("FP16 model saved to %s", output_path)
    return output_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a trained EmotionEncoder to ONNX format.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to a .ckpt file or a directory containing one.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output path for the ONNX file.  Defaults to "
            "'<checkpoint_dir>/emotion_classifier_head.onnx'."
        ),
    )
    parser.add_argument(
        "--quantize-fp16",
        action="store_true",
        default=False,
        help="Additionally export a float16-quantised ONNX model.",
    )
    parser.add_argument(
        "--opset-version",
        type=int,
        default=17,
        help="ONNX opset version (default: 17).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Enable DEBUG-level logging.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # 1. Load checkpoint
    checkpoint = _load_checkpoint(args.checkpoint)

    # 2. Reconstruct model
    model = reconstruct_model(checkpoint, args.checkpoint)
    feature_dims = checkpoint["model_build_args"]["feature_dims"]

    # 3. Determine output path
    output_path = args.output
    if output_path is None:
        ckpt_dir = (
            args.checkpoint
            if args.checkpoint.is_dir()
            else args.checkpoint.parent
        )
        output_path = ckpt_dir / "emotion_classifier_head.onnx"

    # 4. Export to ONNX
    onnx_path = export_onnx(
        model,
        output_path=output_path,
        feature_dims=feature_dims,
        opset_version=args.opset_version,
    )

    # 5. Optionally quantise to FP16
    if args.quantize_fp16:
        quantize_fp16(onnx_path)

    logger.info("Done.")


if __name__ == "__main__":
    main()
