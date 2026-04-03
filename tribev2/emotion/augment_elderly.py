# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Generate elderly-voice-augmented copies of RAVDESS + CREMA-D training clips.

Simulates three aging voice characteristics:
1. Lower pitch / reduced pitch range (vocal fold stiffening)
2. Slower speaking rate (reduced motor control)
3. Vocal tremor / jitter (neuromuscular degeneration)

Augmented clips are saved alongside originals in a new directory.
Each original clip gets one augmented variant with randomized parameters.

Usage:
    DATAPATH=... uv run python -m tribev2.emotion.augment_elderly
"""

import logging
import os
import random
from pathlib import Path

import numpy as np
import soundfile

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000

# Augmentation parameter ranges
PITCH_SHIFT_SEMITONES = (-3.0, -1.0)   # lower pitch
TIME_STRETCH_FACTOR = (1.1, 1.3)       # slower speech
TREMOR_FREQ_HZ = (3.0, 7.0)           # vocal tremor frequency
TREMOR_DEPTH = (0.02, 0.08)           # tremor amplitude modulation depth


def pitch_shift(audio: np.ndarray, sr: int, semitones: float) -> np.ndarray:
    """Shift pitch without changing duration using librosa."""
    import librosa
    return librosa.effects.pitch_shift(audio, sr=sr, n_steps=semitones)


def time_stretch(audio: np.ndarray, factor: float) -> np.ndarray:
    """Slow down audio without changing pitch using librosa."""
    import librosa
    return librosa.effects.time_stretch(audio, rate=1.0 / factor)


def add_tremor(audio: np.ndarray, sr: int, freq_hz: float, depth: float) -> np.ndarray:
    """Add vocal tremor via low-frequency amplitude modulation."""
    t = np.arange(len(audio)) / sr
    modulation = 1.0 + depth * np.sin(2 * np.pi * freq_hz * t)
    return (audio * modulation).astype(np.float32)


def augment_clip(audio: np.ndarray, sr: int) -> np.ndarray:
    """Apply random elderly voice augmentation to a clip."""
    # Random parameters
    semitones = random.uniform(*PITCH_SHIFT_SEMITONES)
    stretch = random.uniform(*TIME_STRETCH_FACTOR)
    tremor_freq = random.uniform(*TREMOR_FREQ_HZ)
    tremor_depth = random.uniform(*TREMOR_DEPTH)

    # Apply augmentations
    aug = pitch_shift(audio, sr, semitones)
    aug = time_stretch(aug, stretch)
    aug = add_tremor(aug, sr, tremor_freq, tremor_depth)

    # Normalize to original RMS to avoid volume changes
    orig_rms = np.sqrt(np.mean(audio ** 2)) + 1e-8
    aug_rms = np.sqrt(np.mean(aug ** 2)) + 1e-8
    aug = aug * (orig_rms / aug_rms)

    return aug.astype(np.float32)


def load_training_clips() -> list[dict]:
    """Load training clip metadata from RAVDESS + CREMA-D."""
    from tribev2.studies.emotion_audio import CremadEmotion, RavdessEmotion

    datapath = Path(os.getenv("DATAPATH", "/home/didierlacroix1/data/emotion"))
    clips = []

    for StudyClass, subdir in [
        (RavdessEmotion, "RavdessEmotion"),
        (CremadEmotion, "CremadEmotion"),
    ]:
        study = StudyClass(path=datapath / subdir)
        for tl in study.iter_timelines():
            events = study._load_timeline_events(tl)
            split = events.iloc[0].get("split", "train")
            if split != "train":
                continue
            filepath = events.iloc[0]["filepath"]
            clips.append({
                "filepath": filepath,
                "emotion": tl["emotion"],
                "dataset": subdir,
            })

    return clips


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
    )

    datapath = Path(os.getenv("DATAPATH", "/home/didierlacroix1/data/emotion"))
    outdir = datapath / "AugmentedElderly"
    outdir.mkdir(parents=True, exist_ok=True)

    clips = load_training_clips()
    logger.info("Training clips to augment: %d", len(clips))

    random.seed(42)
    np.random.seed(42)

    from collections import Counter
    emotion_counts = Counter()
    success = 0
    failed = 0

    for i, clip in enumerate(clips):
        src = clip["filepath"]
        emotion = clip["emotion"]
        dataset = clip["dataset"]

        # Output path: AugmentedElderly/{dataset}_{basename}
        basename = Path(src).stem
        out_path = outdir / f"{dataset}_{basename}_elderly.wav"

        if out_path.exists():
            emotion_counts[emotion] += 1
            success += 1
            continue

        try:
            audio, sr = soundfile.read(src)
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            audio = audio.astype(np.float32)

            if sr != SAMPLE_RATE:
                from scipy.signal import resample
                n = int(len(audio) * SAMPLE_RATE / sr)
                audio = resample(audio, n)
                sr = SAMPLE_RATE

            aug = augment_clip(audio, sr)
            soundfile.write(str(out_path), aug, sr)

            emotion_counts[emotion] += 1
            success += 1
        except Exception as e:
            logger.warning("Failed %s: %s", src, e)
            failed += 1

        if (i + 1) % 500 == 0:
            logger.info("Progress: %d/%d (success=%d, failed=%d)", i + 1, len(clips), success, failed)

    logger.info("Done. Augmented %d clips, %d failed.", success, failed)
    logger.info("Output dir: %s", outdir)
    logger.info("Emotion distribution: %s", dict(sorted(emotion_counts.items())))

    # Write metadata CSV for easy loading
    import csv
    meta_path = outdir / "metadata.csv"
    with open(meta_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["filepath", "emotion", "dataset", "augmentation"])
        writer.writeheader()
        for clip in clips:
            basename = Path(clip["filepath"]).stem
            out_path = outdir / f"{clip['dataset']}_{basename}_elderly.wav"
            if out_path.exists():
                writer.writerow({
                    "filepath": str(out_path),
                    "emotion": clip["emotion"],
                    "dataset": clip["dataset"],
                    "augmentation": "elderly_voice",
                })
    logger.info("Metadata saved to %s", meta_path)


if __name__ == "__main__":
    main()
