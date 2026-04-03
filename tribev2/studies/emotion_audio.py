# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Emotion audio studies: RAVDESS and CREMA-D mapped to 6 emotion classes.

Both datasets provide short speech clips labeled with categorical emotions.
We use the 6 emotions shared across both datasets: neutral, happy, sad,
angry, fear, disgust.  RAVDESS "calm" folds into neutral; "surprised"
folds into angry (both are high-arousal).

RAVDESS
-------
Livingstone SR, Russo FA (2018). The Ryerson Audio-Visual Database of
Emotional Speech and Song (RAVDESS). PLoS ONE 13(5): e0196391.

CREMA-D
-------
Cao H, Cooper DG, Keutmann MK, Gur RC, Nenkova A, Verma R (2014).
CREMA-D: Crowd-Sourced Emotional Multimodal Actors Dataset.
IEEE Transactions on Affective Computing 5(4): 377-390.
"""

import typing as tp
from pathlib import Path

import pandas as pd
import soundfile
from neuralset.events import study

# Canonical label order — used by LabelEncoder and the sidecar.
EMOTION_LABELS: list[str] = ["neutral", "happy", "sad", "angry", "fear", "disgust"]

# IEMOCAP 6-class label order (no disgust — used when training with IEMOCAP)
IEMOCAP_EMOTION_LABELS: list[str] = ["neutral", "happy", "sad", "angry", "fear"]


class RavdessEmotion(study.Study):
    """RAVDESS speech emotion clips mapped to 6 classes."""

    device: tp.ClassVar[str] = "Audio"
    licence: tp.ClassVar[str] = "CC-BY-NC-SA 4.0"
    url: tp.ClassVar[str] = "https://zenodo.org/record/1188976"
    description: tp.ClassVar[str] = (
        "Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS) "
        "mapped to 6 emotion classes for speech emotion recognition."
    )

    # RAVDESS filename emotion code -> our 6 target labels.
    # 02 (calm) -> neutral, 08 (surprised) -> angry (both high-arousal)
    _EMOTION_MAP: tp.ClassVar[dict[str, str]] = {
        "01": "neutral",
        "02": "neutral",    # calm -> neutral
        "03": "happy",
        "04": "sad",
        "05": "angry",
        "06": "fear",
        "07": "disgust",
        "08": "angry",      # surprised -> angry (high arousal)
    }

    _NUM_ACTORS: tp.ClassVar[int] = 24

    def _download(self) -> None:
        raise NotImplementedError("Download method not implemented yet")

    def iter_timelines(self) -> tp.Iterator[dict[str, tp.Any]]:
        audio_dir = self.path / "download"
        for actor_idx in range(1, self._NUM_ACTORS + 1):
            actor_dir = audio_dir / f"Actor_{actor_idx:02d}"
            if not actor_dir.is_dir():
                continue
            subject = f"Actor_{actor_idx:02d}"
            for wav in sorted(actor_dir.glob("*.wav")):
                parts = wav.stem.split("-")
                if len(parts) != 7:
                    continue
                emotion_code = parts[2]
                if emotion_code not in self._EMOTION_MAP:
                    continue
                yield dict(
                    subject=subject,
                    filename=wav.name,
                    emotion=self._EMOTION_MAP[emotion_code],
                )

    def _load_timeline_events(self, timeline: dict[str, tp.Any]) -> pd.DataFrame:
        subject = timeline["subject"]
        filename = timeline["filename"]
        filepath = self.path / "download" / subject / filename
        info = soundfile.info(str(filepath))
        duration = info.duration

        actor_idx = int(subject.split("_")[1])
        split = "val" if actor_idx > round(self._NUM_ACTORS * 0.8) else "train"

        event = dict(
            type="Audio",
            start=0.0,
            duration=duration,
            filepath=str(filepath),
            split=split,
        )
        return pd.DataFrame([event])


class CremadEmotion(study.Study):
    """CREMA-D speech emotion clips mapped to 6 classes."""

    device: tp.ClassVar[str] = "Audio"
    licence: tp.ClassVar[str] = "Open Database License (ODbL)"
    url: tp.ClassVar[str] = "https://github.com/CheyneyComputerScience/CREMA-D"
    description: tp.ClassVar[str] = (
        "Crowd-Sourced Emotional Multimodal Actors Dataset (CREMA-D) "
        "mapped to 6 emotion classes for speech emotion recognition."
    )

    _EMOTION_MAP: tp.ClassVar[dict[str, str]] = {
        "NEU": "neutral",
        "HAP": "happy",
        "SAD": "sad",
        "ANG": "angry",
        "FEA": "fear",
        "DIS": "disgust",
    }

    def _download(self) -> None:
        raise NotImplementedError("Download method not implemented yet")

    def iter_timelines(self) -> tp.Iterator[dict[str, tp.Any]]:
        audio_dir = self.path / "download" / "AudioWAV"
        if not audio_dir.is_dir():
            return
        for wav in sorted(audio_dir.glob("*.wav")):
            parts = wav.stem.split("_")
            if len(parts) < 3:
                continue
            actor_id = parts[0]
            emotion_tag = parts[2]
            if emotion_tag not in self._EMOTION_MAP:
                continue
            yield dict(
                subject=actor_id,
                filename=wav.name,
                emotion=self._EMOTION_MAP[emotion_tag],
            )

    # Cache val actors so we don't re-glob 7k+ files per timeline
    _val_actors_cache: tp.ClassVar[set[str] | None] = None

    def _get_val_actors(self) -> set[str]:
        if CremadEmotion._val_actors_cache is None:
            audio_dir = self.path / "download" / "AudioWAV"
            all_actors = sorted(
                {f.stem.split("_")[0] for f in audio_dir.glob("*.wav") if "_" in f.stem}
            )
            cutoff = round(len(all_actors) * 0.8)
            CremadEmotion._val_actors_cache = set(all_actors[cutoff:])
        return CremadEmotion._val_actors_cache

    def _load_timeline_events(self, timeline: dict[str, tp.Any]) -> pd.DataFrame:
        filename = timeline["filename"]
        filepath = self.path / "download" / "AudioWAV" / filename
        info = soundfile.info(str(filepath))
        duration = info.duration

        split = "val" if timeline["subject"] in self._get_val_actors() else "train"

        event = dict(
            type="Audio",
            start=0.0,
            duration=duration,
            filepath=str(filepath),
            split=split,
        )
        return pd.DataFrame([event])


class IEMOCAPEmotion(study.Study):
    """IEMOCAP naturalistic dyadic speech mapped to 5 emotion classes.

    Uses the pre-exported WAV + metadata.csv from tribev2.emotion.prepare_iemocap.
    Emotion mapping (6-class, matching RAVDESS/CREMA-D except no disgust):
        neutral ← neutral
        happy   ← happy + excited
        sad     ← sad
        angry   ← angry + frustrated
        fear    ← fear
    Train/val split is by speaker (Ses05F, Ses05M held out as val).
    """

    device: tp.ClassVar[str] = "Audio"
    licence: tp.ClassVar[str] = "USC non-commercial research"
    url: tp.ClassVar[str] = "https://sail.usc.edu/iemocap/"
    description: tp.ClassVar[str] = (
        "Interactive Emotional Dyadic Motion Capture (IEMOCAP) "
        "naturalistic dyadic conversations mapped to 5 emotion classes."
    )

    def _download(self) -> None:
        raise NotImplementedError(
            "Run: DATAPATH=... uv run python -m tribev2.emotion.prepare_iemocap"
        )

    def iter_timelines(self) -> tp.Iterator[dict[str, tp.Any]]:
        metadata_path = self.path / "metadata.csv"
        if not metadata_path.exists():
            return
        df = pd.read_csv(metadata_path)
        for _, row in df.iterrows():
            yield dict(
                subject=row["speaker_id"],
                filename=row["filename"],
                emotion=row["emotion"],
                split=row["split"],
            )

    def _load_timeline_events(self, timeline: dict[str, tp.Any]) -> pd.DataFrame:
        filename = timeline["filename"]
        filepath = self.path / f"{filename}.wav"
        info = soundfile.info(str(filepath))
        duration = info.duration

        event = dict(
            type="Audio",
            start=0.0,
            duration=duration,
            filepath=str(filepath),
            split=timeline["split"],
        )
        return pd.DataFrame([event])
