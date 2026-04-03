# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Export IEMOCAP from HuggingFace to disk as WAV files + metadata CSV.

Maps IEMOCAP's 10 emotions to 6 classes (matching RAVDESS/CREMA-D system):
  neutral ← neutral
  happy   ← happy + excited
  sad     ← sad
  angry   ← angry + frustrated
  fear    ← fear (107 naturalistic clips)
  (disgust=2, surprise=110, other=26 dropped — too few or no equivalent)

Creates train/val splits by speaker (80/20 by speaker ID derived from filename).

Usage:
    HF_DATASETS_CACHE=~/.cache/huggingface/datasets \\
    DATAPATH=~/data/emotion \\
    uv run python -m tribev2.emotion.prepare_iemocap
"""

import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from tqdm import tqdm

logger = logging.getLogger(__name__)

DATADIR = Path(os.getenv("DATAPATH", os.path.expanduser("~/data/emotion"))) / "IEMOCAPEmotion"

# 6-class mapping: map IEMOCAP → our label (matches RAVDESS/CREMA-D system)
EMOTION_MAP = {
    "neutral": "neutral",
    "happy": "happy",
    "excited": "happy",
    "sad": "sad",
    "angry": "angry",
    "frustrated": "angry",
    "fear": "fear",
    # disgust (2), surprise (110), other (26) → dropped
}


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

    import os as _os
    _os.environ.setdefault("HF_DATASETS_CACHE", str(Path.home() / ".cache/huggingface/datasets"))

    from datasets import load_dataset

    logger.info("Loading IEMOCAP from HuggingFace cache...")
    ds = load_dataset("AbstractTTS/IEMOCAP")["train"]

    DATADIR.mkdir(parents=True, exist_ok=True)

    rows = []
    skipped = 0

    for i, sample in enumerate(tqdm(ds, desc="Exporting WAV")):
        emotion_raw = sample["major_emotion"]
        emotion = EMOTION_MAP.get(emotion_raw)
        if emotion is None:
            skipped += 1
            continue

        # Extract speaker ID from filename: e.g. Ses01F_impro01_F000 → speaker Ses01F
        filename = Path(sample["file"]).stem  # e.g. Ses01F_impro01_F000
        parts = filename.split("_")
        speaker_id = parts[0] if parts else f"spk{i:04d}"  # e.g. Ses01F, Ses02M, ...

        # Save WAV
        wav_path = DATADIR / f"{filename}.wav"
        if not wav_path.exists():
            audio = sample["audio"]
            arr = np.array(audio["array"], dtype=np.float32)
            sf.write(str(wav_path), arr, audio["sampling_rate"])

        # Save transcription TSV alongside WAV (for ExtractWordsFromAudio cache)
        tsv_path = DATADIR / f"{filename}.tsv"
        if not tsv_path.exists():
            text = sample["transcription"].strip()
            if text:
                duration = len(sample["audio"]["array"]) / sample["audio"]["sampling_rate"]
                words = text.split()
                word_dur = duration / max(len(words), 1)
                word_rows = []
                for j, w in enumerate(words):
                    word_rows.append({
                        "text": w,
                        "start": j * word_dur,
                        "duration": word_dur,
                        "sequence_id": 0,
                        "sentence": text,
                    })
                pd.DataFrame(word_rows).to_csv(tsv_path, sep="\t", index=False)
            else:
                pd.DataFrame().to_csv(tsv_path, sep="\t", index=False)

        rows.append({
            "filename": filename,
            "speaker_id": speaker_id,
            "emotion": emotion,
            "emotion_raw": emotion_raw,
            "transcription": sample["transcription"].strip(),
            "duration": len(sample["audio"]["array"]) / sample["audio"]["sampling_rate"],
            "gender": sample["gender"],
        })

    df = pd.DataFrame(rows)

    # Train/val split by speaker (last 20% of unique speakers → val)
    speakers = sorted(df["speaker_id"].unique())
    n_val = max(1, int(len(speakers) * 0.2))
    val_speakers = set(speakers[-n_val:])
    df["split"] = df["speaker_id"].apply(lambda s: "val" if s in val_speakers else "train")

    df.to_csv(DATADIR / "metadata.csv", index=False)

    logger.info(
        "Done: %d clips saved, %d skipped (unmapped emotion)",
        len(df), skipped,
    )
    logger.info("Emotion distribution:\n%s", df["emotion"].value_counts().to_string())
    logger.info("Split distribution:\n%s", df["split"].value_counts().to_string())
    logger.info(
        "Speakers: %d total, %d val: %s",
        len(speakers), len(val_speakers), sorted(val_speakers),
    )
    logger.info("Output: %s", DATADIR)


if __name__ == "__main__":
    main()
