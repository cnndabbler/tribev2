# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Pre-extract audio or text features for the combined emotion dataset.

Run both modalities in parallel from two terminal tabs to saturate the GPU
(Wav2VecBert=2.4GB + Qwen3-0.6B=1.2GB = 3.6GB, fits in 24GB VRAM):

    # Terminal 1:
    DATAPATH=... SAVEPATH=... uv run python -m tribev2.emotion.pre_extract --modality audio

    # Terminal 2 (simultaneously):
    DATAPATH=... SAVEPATH=... uv run python -m tribev2.emotion.pre_extract --modality text

After both complete, run training (extraction is skipped, only cached features used):
    DATAPATH=... SAVEPATH=... uv run python -m tribev2.grids.emotion_combined
"""

import argparse
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

PROJECT_NAME = "tribe_emotion_combined"
NUM_CLASSES = 6


def get_events(cachedir: str):
    """Load all events from all three datasets via ClassificationData."""
    import copy
    from tribev2.emotion.data import ClassificationData
    from tribev2.grids.emotion_combined import combined_config
    from tribev2.studies.emotion_audio import CremadEmotion, IEMOCAPEmotion, RavdessEmotion  # noqa: F401 — register studies

    data_config = copy.deepcopy(combined_config["data"])
    data_config.pop("num_classes", None)
    # Override cache folder to the shared project cache
    for key in ("audio_feature", "text_feature"):
        if key in data_config:
            data_config[key]["infra"]["folder"] = cachedir
            data_config[key]["infra"]["keep_in_ram"] = False

    d = ClassificationData(**data_config)
    logger.info("Loading events from all studies...")
    events = d.get_events()
    logger.info("Total events: %d (types: %s)", len(events), events.type.value_counts().to_dict())
    return events


def extract_audio(cachedir: str) -> None:
    """Pre-extract Wav2VecBert features for all audio events."""
    import torch
    from tribev2.grids.emotion_combined import audio_feature
    import copy

    cfg = copy.deepcopy(audio_feature)
    cfg["infra"]["folder"] = cachedir

    from neuralset.extractors import Wav2VecBert
    extractor = Wav2VecBert(**{k: v for k, v in cfg.items() if k != "name"})

    logger.info("Loading Wav2VecBert onto GPU...")
    events = get_events(cachedir)
    logger.info("Extracting audio features...")
    extractor.prepare(events)
    logger.info("Audio extraction complete.")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def extract_text(cachedir: str) -> None:
    """Pre-extract Qwen3-0.6B text features for all word events."""
    import torch
    from tribev2.grids.emotion_combined import text_feature
    import copy

    cfg = copy.deepcopy(text_feature)
    cfg["infra"]["folder"] = cachedir
    cfg["batch_size"] = 64   # max batching for Qwen3-0.6B on A5000

    from neuralset.extractors import HuggingFaceText
    extractor = HuggingFaceText(**{k: v for k, v in cfg.items() if k != "name"})

    logger.info("Loading Qwen3-0.6B onto GPU...")
    events = get_events(cachedir)
    logger.info("Extracting text features...")
    extractor.prepare(events)
    logger.info("Text extraction complete.")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="Pre-extract emotion features")
    parser.add_argument(
        "--modality",
        choices=["audio", "text"],
        required=True,
        help="Which modality to extract",
    )
    args = parser.parse_args()

    BASEDIR = os.environ["SAVEPATH"]
    cachedir = os.path.join(BASEDIR, "cache", PROJECT_NAME)
    Path(cachedir).mkdir(parents=True, exist_ok=True)

    logger.info("Cache dir: %s", cachedir)
    logger.info("Modality: %s", args.modality)

    if args.modality == "audio":
        extract_audio(cachedir)
    else:
        extract_text(cachedir)


if __name__ == "__main__":
    main()
