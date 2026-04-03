# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""One-shot extraction of backbone features for emotion clips.

Extracts Wav2VecBert (audio) and Qwen3-0.6B (text) features for all clips
across RAVDESS, CREMA-D, and IEMOCAP. Saves per-dataset .pt files with
full temporal sequences (not mean-pooled) for temporal head training.

Audio: (N, T_audio, 1024) — T_audio=199 for 4s clips (Wav2VecBert 50Hz output)
Text:  (N, T_text, 1024)  — T_text=max_tokens, with lengths stored separately

Stored as float16 to save disk/RAM (~7.5GB audio + ~2.3GB text).

Both models fit on A5000 simultaneously (2.4GB + 1.2GB = 3.6GB).

Usage:
    DATAPATH=... SAVEPATH=... uv run python -m tribev2.emotion.cache_features
    DATAPATH=... SAVEPATH=... uv run python -m tribev2.emotion.cache_features --modality audio
    DATAPATH=... SAVEPATH=... uv run python -m tribev2.emotion.cache_features --modality text
"""

import argparse
import csv
import logging
import os
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

logger = logging.getLogger(__name__)

BACKBONE_NAME = "facebook/w2v-bert-2.0"
TEXT_MODEL_NAME = "Qwen/Qwen3-0.6B"
EMOTION_LABELS = ["neutral", "happy", "sad", "angry", "fear", "disgust"]
SAMPLE_RATE = 16000
MAX_AUDIO_SEC = 4.0
BATCH_SIZE_AUDIO = 32
BATCH_SIZE_TEXT = 64
MAX_TEXT_TOKENS = 64  # max tokens for text sequences


def load_all_clips() -> dict[str, list[dict]]:
    """Load clip metadata from all three Study classes.

    Returns dict keyed by dataset name, each value a list of clip dicts
    with keys: filepath, emotion, split, tsv_path.
    """
    from tribev2.studies.emotion_audio import CremadEmotion, IEMOCAPEmotion, RavdessEmotion

    datapath = Path(os.getenv("DATAPATH", "/home/didierlacroix1/data/emotion"))
    all_clips: dict[str, list[dict]] = {}

    # RAVDESS + CREMA-D
    for StudyClass, name, subdir in [
        (RavdessEmotion, "ravdess", "RavdessEmotion"),
        (CremadEmotion, "cremad", "CremadEmotion"),
    ]:
        clips = []
        study = StudyClass(path=datapath / subdir)
        for tl in study.iter_timelines():
            events = study._load_timeline_events(tl)
            filepath = events.iloc[0]["filepath"]
            split = events.iloc[0].get("split", "train")
            tsv_path = str(Path(filepath).with_suffix(".tsv"))
            clips.append({
                "filepath": filepath,
                "emotion": tl["emotion"],
                "split": split,
                "tsv_path": tsv_path,
            })
        all_clips[name] = clips

    # IEMOCAP
    iemocap_clips = []
    iemocap = IEMOCAPEmotion(path=datapath / "IEMOCAPEmotion")
    for tl in iemocap.iter_timelines():
        filepath = str(datapath / "IEMOCAPEmotion" / f"{tl['filename']}.wav")
        tsv_path = str(datapath / "IEMOCAPEmotion" / f"{tl['filename']}.tsv")
        iemocap_clips.append({
            "filepath": filepath,
            "emotion": tl["emotion"],
            "split": tl["split"],
            "tsv_path": tsv_path,
        })
    all_clips["iemocap"] = iemocap_clips

    for name, clips in all_clips.items():
        logger.info("%s: %d clips", name, len(clips))

    return all_clips


def get_sentence(tsv_path: str) -> str:
    """Read the sentence from a TSV transcription file."""
    try:
        with open(tsv_path, "r") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                sentence = row.get("sentence", "").strip()
                if sentence:
                    return sentence
    except (FileNotFoundError, StopIteration, KeyError):
        pass
    return ""


@torch.no_grad()
def extract_audio_features(
    clips: list[dict], device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract full-sequence Wav2VecBert features for all clips.

    Returns:
        feats: (N, T, 1024) float16 — full temporal sequence
        lengths: (N,) int — valid frame count per clip
    """
    from transformers import AutoFeatureExtractor, Wav2Vec2BertModel
    import soundfile

    logger.info("Loading Wav2VecBert...")
    model = Wav2Vec2BertModel.from_pretrained(BACKBONE_NAME).to(device).eval()
    feature_extractor = AutoFeatureExtractor.from_pretrained(BACKBONE_NAME)
    max_samples = int(MAX_AUDIO_SEC * SAMPLE_RATE)

    all_feats = []
    all_lengths = []
    seq_len = None  # determined from first batch

    for i in tqdm(range(0, len(clips), BATCH_SIZE_AUDIO), desc="Audio"):
        batch_clips = clips[i : i + BATCH_SIZE_AUDIO]
        audios = []
        for clip in batch_clips:
            audio, sr = soundfile.read(clip["filepath"])
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            if sr != SAMPLE_RATE:
                from scipy.signal import resample
                n = int(len(audio) * SAMPLE_RATE / sr)
                audio = resample(audio, n)
            audio = audio.astype(np.float32)
            if len(audio) > max_samples:
                audio = audio[:max_samples]
            elif len(audio) < max_samples:
                audio = np.pad(audio, (0, max_samples - len(audio)))
            audios.append(audio)

        inputs = feature_extractor(
            audios, sampling_rate=SAMPLE_RATE, return_tensors="pt", padding=True,
        )
        input_features = inputs["input_features"].to(device)
        outputs = model(input_features, output_hidden_states=False)
        hidden = outputs.last_hidden_state  # (B, T, 1024)

        if seq_len is None:
            seq_len = hidden.shape[1]
            logger.info("Audio sequence length: %d frames", seq_len)

        all_feats.append(hidden.cpu().half())
        all_lengths.extend([seq_len] * len(batch_clips))

    del model
    torch.cuda.empty_cache()
    return torch.cat(all_feats, dim=0), torch.tensor(all_lengths, dtype=torch.long)


@torch.no_grad()
def extract_text_features(
    clips: list[dict], device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract full-sequence Qwen3-0.6B features for all clip sentences.

    Returns:
        feats: (N, MAX_TEXT_TOKENS, 1024) float16 — padded/truncated sequences
        lengths: (N,) int — valid token count per clip
    """
    from transformers import AutoModel, AutoTokenizer

    logger.info("Loading Qwen3-0.6B...")
    tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_NAME, trust_remote_code=True)
    model = AutoModel.from_pretrained(TEXT_MODEL_NAME, trust_remote_code=True).to(device).eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    sentences = []
    empty_count = 0
    for clip in clips:
        s = get_sentence(clip["tsv_path"])
        if not s:
            empty_count += 1
        sentences.append(s if s else "unknown")
    if empty_count > 0:
        logger.warning("%d clips with empty/missing transcriptions (using 'unknown')", empty_count)

    all_feats = []
    all_lengths = []
    for i in tqdm(range(0, len(sentences), BATCH_SIZE_TEXT), desc="Text"):
        batch_sentences = sentences[i : i + BATCH_SIZE_TEXT]
        inputs = tokenizer(
            batch_sentences, return_tensors="pt", padding="max_length",
            truncation=True, max_length=MAX_TEXT_TOKENS,
        ).to(device)
        outputs = model(**inputs, output_hidden_states=False)
        hidden = outputs.last_hidden_state  # (B, T, 1024)

        # Store lengths (non-padding token count)
        lengths = inputs["attention_mask"].sum(dim=1)  # (B,)
        all_feats.append(hidden.cpu().half())
        all_lengths.append(lengths.cpu())

    del model
    torch.cuda.empty_cache()
    return torch.cat(all_feats, dim=0), torch.cat(all_lengths, dim=0)


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="Cache backbone features for emotion clips")
    parser.add_argument(
        "--modality", choices=["audio", "text", "both"], default="both",
        help="Which modality to extract (default: both)",
    )
    args = parser.parse_args()

    savepath = os.getenv("SAVEPATH", "/home/didierlacroix1/data/emotion_save")
    cachedir = os.path.join(savepath, "cache", "cached_features")
    Path(cachedir).mkdir(parents=True, exist_ok=True)
    logger.info("Cache dir: %s", cachedir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    label_map = {e: i for i, e in enumerate(EMOTION_LABELS)}
    all_clips = load_all_clips()

    for dataset_name, clips in all_clips.items():
        out_path = os.path.join(cachedir, f"{dataset_name}.pt")
        logger.info("Processing %s (%d clips)...", dataset_name, len(clips))

        # Load existing cache if partially done
        existing = {}
        if os.path.exists(out_path):
            existing = torch.load(out_path, map_location="cpu")
            logger.info("  Loaded existing cache: %s", list(existing.keys()))

        labels = torch.tensor([label_map[c["emotion"]] for c in clips], dtype=torch.long)
        splits = [c["split"] for c in clips]
        filepaths = [c["filepath"] for c in clips]

        audio_feats = audio_lengths = text_feats = text_lengths = None

        if args.modality in ("audio", "both"):
            if "audio_feats" in existing and existing["audio_feats"].shape[0] == len(clips) and existing["audio_feats"].ndim == 3:
                logger.info("  Audio features already cached (seq), skipping")
                audio_feats = existing["audio_feats"]
                audio_lengths = existing["audio_lengths"]
            else:
                audio_feats, audio_lengths = extract_audio_features(clips, device)
                logger.info("  Audio features: %s", audio_feats.shape)
        else:
            audio_feats = existing.get("audio_feats")
            audio_lengths = existing.get("audio_lengths")

        if args.modality in ("text", "both"):
            if "text_feats" in existing and existing["text_feats"].shape[0] == len(clips) and existing["text_feats"].ndim == 3:
                logger.info("  Text features already cached (seq), skipping")
                text_feats = existing["text_feats"]
                text_lengths = existing["text_lengths"]
            else:
                text_feats, text_lengths = extract_text_features(clips, device)
                logger.info("  Text features: %s", text_feats.shape)
        else:
            text_feats = existing.get("text_feats")
            text_lengths = existing.get("text_lengths")

        save_dict = {
            "labels": labels,
            "splits": splits,
            "filepaths": filepaths,
            "label_map": label_map,
            "emotion_labels": EMOTION_LABELS,
        }
        if audio_feats is not None:
            save_dict["audio_feats"] = audio_feats
            save_dict["audio_lengths"] = audio_lengths
        if text_feats is not None:
            save_dict["text_feats"] = text_feats
            save_dict["text_lengths"] = text_lengths

        torch.save(save_dict, out_path)
        logger.info("  Saved to %s", out_path)

    logger.info("Done.")


if __name__ == "__main__":
    main()
