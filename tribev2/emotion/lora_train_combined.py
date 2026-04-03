# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""LoRA fine-tuning on combined RAVDESS + CREMA-D + IEMOCAP dataset.

Extends lora_train.py by adding IEMOCAPEmotion to the training data.
IEMOCAP contributes naturalistic dyadic speech (~9,900 clips) to supplement
the acted speech in RAVDESS/CREMA-D.

IEMOCAP has no disgust class — those clips simply don't appear in IEMOCAP.
RAVDESS + CREMA-D still fully cover all 6 classes.

Usage:
    DATAPATH=... SAVEPATH=... LORA_R=128 uv run python -m tribev2.emotion.lora_train_combined
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Import everything from the base LoRA script, then override what changes
# ---------------------------------------------------------------------------
from tribev2.emotion.lora_train import (  # noqa: F401
    BACKBONE_NAME,
    BATCH_SIZE,
    EMOTION_LABELS,
    EPOCHS,
    LR_HEAD,
    LR_LORA,
    LORA_ALPHA,
    LORA_DROPOUT,
    LORA_LAYERS,
    LORA_R,
    LORA_TARGET_MODULES,
    MAX_AUDIO_SEC,
    NUM_CLASSES,
    PATIENCE,
    SAMPLE_RATE,
    WARMUP_STEPS,
    EmotionAudioDataset,
    LoRAEmotionModel,
    build_model,
    evaluate,
    train_one_epoch,
)
import tribev2.emotion.lora_train as _base

# Override save dir to reflect combined dataset
SAVEDIR = os.path.join(
    os.getenv("SAVEPATH", "/home/didierlacroix1/data/emotion_save"),
    "results", f"tribe_emotion_lora_combined_r{LORA_R}",
)
Path(SAVEDIR).mkdir(parents=True, exist_ok=True)

# Patch the base module's SAVEDIR so main() uses ours
_base.SAVEDIR = SAVEDIR

# Increase batch size — larger dataset benefits from bigger batches on A5000
_base.BATCH_SIZE = 16


def load_clips() -> tuple[list[dict], list[dict]]:
    """Load clips from RAVDESS + CREMA-D + IEMOCAP."""
    from tribev2.studies.emotion_audio import CremadEmotion, IEMOCAPEmotion, RavdessEmotion

    datapath = Path(os.getenv("DATAPATH", "/home/didierlacroix1/data/emotion"))
    train_clips, val_clips = [], []

    # RAVDESS + CREMA-D (same as original, 6 classes including disgust)
    for StudyClass, subdir in [
        (RavdessEmotion, "RavdessEmotion"),
        (CremadEmotion, "CremadEmotion"),
    ]:
        study = StudyClass(path=datapath / subdir)
        for tl in study.iter_timelines():
            events = study._load_timeline_events(tl)
            split = events.iloc[0].get("split", "train")
            filepath = events.iloc[0]["filepath"]
            clip = {"filepath": filepath, "emotion": tl["emotion"], "split": split}
            (val_clips if split == "val" else train_clips).append(clip)

    # IEMOCAP (5 classes: neutral, happy, sad, angry, fear — no disgust)
    iemocap = IEMOCAPEmotion(path=datapath / "IEMOCAPEmotion")
    for tl in iemocap.iter_timelines():
        filepath = str(datapath / "IEMOCAPEmotion" / f"{tl['filename']}.wav")
        clip = {"filepath": filepath, "emotion": tl["emotion"], "split": tl["split"]}
        (val_clips if tl["split"] == "val" else train_clips).append(clip)

    import logging
    logger = logging.getLogger(__name__)
    logger.info(
        "Combined dataset: %d train clips, %d val clips",
        len(train_clips), len(val_clips),
    )

    from collections import Counter
    train_dist = Counter(c["emotion"] for c in train_clips)
    val_dist = Counter(c["emotion"] for c in val_clips)
    logger.info("Train emotion distribution: %s", dict(sorted(train_dist.items())))
    logger.info("Val emotion distribution:   %s", dict(sorted(val_dist.items())))

    return train_clips, val_clips


# Patch load_clips in the base module so main() picks it up
_base.load_clips = load_clips


if __name__ == "__main__":
    _base.main()
