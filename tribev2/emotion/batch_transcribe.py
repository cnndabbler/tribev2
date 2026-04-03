# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Batch transcribe all emotion audio clips using WhisperX.

Loads the model ONCE and processes all clips in a single pass,
saving .tsv transcripts alongside each .wav file. The tribev2
ExtractWordsFromAudio transform checks for these .tsv files
and skips transcription if they exist.

Usage:
    DATAPATH=~/data/emotion uv run python -m tribev2.emotion.batch_transcribe
"""

import logging
import os
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

logger = logging.getLogger(__name__)

DATADIR = os.getenv("DATAPATH", os.path.expanduser("~/data/emotion"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
COMPUTE_TYPE = "float16" if DEVICE == "cuda" else "int8"
BATCH_SIZE = 16
LANGUAGE = "en"


def get_all_wav_files() -> list[Path]:
    """Find all WAV files in RAVDESS and CREMA-D that don't have transcripts yet."""
    wavs = []
    for root, _, files in os.walk(DATADIR):
        for f in sorted(files):
            if not f.endswith(".wav"):
                continue
            wav_path = Path(root) / f
            tsv_path = wav_path.with_suffix(".tsv")
            if not tsv_path.exists():
                wavs.append(wav_path)
    return wavs


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
    )

    wavs = get_all_wav_files()
    logger.info("Found %d WAV files without transcripts in %s", len(wavs), DATADIR)

    if not wavs:
        logger.info("All files already transcribed. Nothing to do.")
        return

    # Load WhisperX model ONCE
    import whisperx

    logger.info("Loading WhisperX model (device=%s, compute_type=%s)...", DEVICE, COMPUTE_TYPE)
    model = whisperx.load_model(
        "large-v3",
        device=DEVICE,
        compute_type=COMPUTE_TYPE,
        language=LANGUAGE,
    )

    # Load alignment model ONCE
    logger.info("Loading alignment model...")
    align_model, align_metadata = whisperx.load_align_model(
        language_code=LANGUAGE,
        device=DEVICE,
    )

    # Process all files
    success = 0
    errors = 0

    for wav_path in tqdm(wavs, desc="Transcribing"):
        try:
            audio = whisperx.load_audio(str(wav_path))
            result = model.transcribe(audio, batch_size=BATCH_SIZE, language=LANGUAGE)

            # Align
            result = whisperx.align(
                result["segments"],
                align_model,
                align_metadata,
                audio,
                DEVICE,
                return_char_alignments=False,
            )

            # Extract words
            words = []
            for i, segment in enumerate(result["segments"]):
                sentence = segment.get("text", "").replace('"', "")
                for word in segment.get("words", []):
                    if "start" not in word:
                        continue
                    words.append({
                        "text": word["word"].replace('"', ""),
                        "start": word["start"],
                        "duration": word["end"] - word["start"],
                        "sequence_id": i,
                        "sentence": sentence,
                    })

            # Save as TSV (same format ExtractWordsFromAudio expects)
            df = pd.DataFrame(words)
            tsv_path = wav_path.with_suffix(".tsv")
            df.to_csv(tsv_path, sep="\t", index=False)
            success += 1

        except Exception as e:
            logger.warning("Failed to transcribe %s: %s", wav_path.name, e)
            # Write empty TSV so we don't retry
            pd.DataFrame().to_csv(wav_path.with_suffix(".tsv"), sep="\t", index=False)
            errors += 1

    logger.info("Done: %d success, %d errors, %d total", success, errors, success + errors)


if __name__ == "__main__":
    main()
