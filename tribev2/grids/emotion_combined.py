# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Combined multi-modal emotion classification: RAVDESS + CREMA-D + IEMOCAP.

Audio + text co-processing on all three datasets:
- RAVDESS + CREMA-D: 6 classes, acted speech (strong audio signal)
- IEMOCAP: 5 classes (no disgust), naturalistic dyadic speech (strong text signal)

The combined dataset (~18,700 clips) covers:
  neutral, happy, sad, angry, fear  — all three datasets
  disgust                            — RAVDESS + CREMA-D only

NUM_CLASSES = 6 to match the production sidecar.
IEMOCAP samples without disgust simply don't contribute disgust signal.

Usage:
    DATAPATH=... SAVEPATH=... uv run python -m tribev2.grids.emotion_combined
"""

import os
from pathlib import Path

PROJECT_NAME = "tribe_emotion_combined"
NUM_CLASSES = 6

DATADIR = os.getenv("DATAPATH")
BASEDIR = os.getenv("SAVEPATH")
CACHEDIR = os.path.join(BASEDIR, "cache", PROJECT_NAME)
SAVEDIR = os.path.join(BASEDIR, "results", PROJECT_NAME)

for path in [CACHEDIR, SAVEDIR, DATADIR]:
    Path(path).mkdir(parents=True, exist_ok=True)

audio_feature = {
    "name": "Wav2VecBert",
    "frequency": 2,
    "layers": [0.5, 0.75, 1.0],
    "event_types": "Audio",
    "aggregation": "sum",
    "allow_missing": True,
    "infra": {
        "cluster": None,
        "folder": CACHEDIR,
        "keep_in_ram": True,
        "mode": "cached",
        "gpus_per_node": 1,
    },
}

text_feature = {
    "name": "HuggingFaceText",
    "model_name": "Qwen/Qwen3-0.6B",
    "event_types": "Word",
    "contextualized": False,
    "frequency": 2,
    "layers": [0.5, 0.75, 1.0],
    "aggregation": "sum",
    "batch_size": 64,           # max batching on RTX A5000 24GB
    "allow_missing": True,
    "infra": {
        "cluster": None,
        "folder": CACHEDIR,
        "keep_in_ram": True,
        "mode": "cached",
        "gpus_per_node": 1,
    },
}

combined_config = {
    "infra": {
        "cluster": None,
        "folder": SAVEDIR,
        "gpus_per_node": 1,
        "cpus_per_task": 8,
        "mem_gb": 48,
        "mode": "retry",
    },
    "data": {
        "frequency": 2,
        "duration_trs": 6,
        "overlap_trs_train": 0,
        "overlap_trs_val": 0,
        "shuffle_val": True,
        "num_workers": 8,
        "layers_to_use": [0.5, 0.75, 1.0],
        "layer_aggregation": "group_mean",
        "study": {
            "names": ["RavdessEmotion", "CremadEmotion", "IEMOCAPEmotion"],
            "path": DATADIR,
            "query": None,
            "transforms": {
                # Transcription pipeline: pre-computed TSV files act as cache
                # for RAVDESS/CREMA-D (WhisperX) and IEMOCAP (ground-truth words)
                "extractwords": {"name": "ExtractWordsFromAudio"},
                "addtext": {"name": "AddText"},
                "addsentence": {
                    "name": "AddSentenceToWords",
                    "max_unmatched_ratio": 0.5,
                },
                "addcontext": {
                    "name": "AddContextToWords",
                    "sentence_only": False,
                    "max_context_len": 1024,
                    "split_field": "",
                },
                "removemissing": {"name": "RemoveMissing"},
            },
        },
        "features_to_use": ["text", "audio"],
        "text_feature": text_feature,
        "audio_feature": audio_feature,
        "batch_size": 32,
        "num_classes": NUM_CLASSES,
    },
    "wandb_config": None,
    "brain_model_config": {
        "name": "EmotionEncoder",
        "hidden": 512,
        "extractor_aggregation": "cat",
        "layer_aggregation": "cat",
        "combiner": {"name": "Mlp", "norm_layer": "layer", "activation_layer": "gelu"},
        "encoder": {
            "depth": 4,
            "heads": 8,
        },
        "dropout": 0.1,
        "modality_dropout": 0.3,
        "num_classes": NUM_CLASSES,
    },
    "metrics": [
        {
            "log_name": "accuracy",
            "name": "MulticlassAccuracy",
            "kwargs": {"num_classes": NUM_CLASSES},
        },
        {
            "log_name": "f1",
            "name": "MulticlassF1Score",
            "kwargs": {"num_classes": NUM_CLASSES, "average": "weighted"},
        },
    ],
    "loss": {"name": "CrossEntropyLoss", "kwargs": {}},
    "optim": {
        "name": "LightningOptimizer",
        "optimizer": {
            "name": "Adam",
            "lr": 1e-3,
            "kwargs": {"weight_decay": 0.0},
        },
        "scheduler": {
            "name": "OneCycleLR",
            "kwargs": {"max_lr": 1e-3, "pct_start": 0.1},
        },
    },
    "n_epochs": 30,
    "monitor": "val/accuracy",
    "patience": 10,
    "enable_progress_bar": True,
    "log_every_n_steps": 5,
    "seed": 42,
}


if __name__ == "__main__":
    from ..emotion import ClassificationExperiment
    from ..studies.emotion_audio import *  # noqa: F401,F403

    exp = ClassificationExperiment(**combined_config)
    exp.infra.clear_job()
    out = exp.run()
    print(out)
