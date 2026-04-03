# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Default configuration dictionary for speech emotion classification experiments."""
import os
from pathlib import Path

PROJECT_NAME = "tribe_emotion"

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
}
audio_feature["infra"] = {
    "cluster": None,  # local for now, change to "slurm" for distributed
    "folder": CACHEDIR,
    "keep_in_ram": True,
    "mode": "cached",
    "gpus_per_node": 1,
}

emotion_config = {
    "infra": {
        "cluster": None,  # local execution
        "folder": SAVEDIR,
        "gpus_per_node": 1,
        "cpus_per_task": 8,
        "mem_gb": 32,
        "mode": "retry",
    },
    "data": {
        "frequency": 2,
        "duration_trs": 6,  # 3 seconds at 2Hz
        "overlap_trs_train": 0,
        "overlap_trs_val": 0,
        "shuffle_val": True,
        "num_workers": 8,
        "layers_to_use": [0.5, 0.75, 1.0],
        "layer_aggregation": "group_mean",
        "study": {
            "names": ["RavdessEmotion", "CremadEmotion"],
            "path": DATADIR,
            "query": None,
            "transforms": {},  # splits already assigned by Study classes (by subject)
        },
        "features_to_use": ["audio"],
        "audio_feature": audio_feature,
        "batch_size": 32,
        "num_classes": 6,
    },
    "wandb_config": None,  # disable W&B for initial dev
    "brain_model_config": {
        "name": "EmotionEncoder",
        "hidden": 256,
        "extractor_aggregation": "cat",
        "layer_aggregation": "cat",
        "combiner": None,
        "encoder": {
            "depth": 4,
            "heads": 4,
        },
        "modality_dropout": 0.0,
        "num_classes": 6,
    },
    "metrics": [
        {
            "log_name": "accuracy",
            "name": "MulticlassAccuracy",
            "kwargs": {"num_classes": 6},
        },
        {
            "log_name": "f1",
            "name": "MulticlassF1Score",
            "kwargs": {"num_classes": 6, "average": "weighted"},
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
    from ..emotion import ClassificationExperiment  # noqa: F401 — also registers EmotionEncoder, ClassificationData
    from ..studies.emotion_audio import *  # noqa: F401,F403 — register emotion studies

    exp = ClassificationExperiment(**emotion_config)
    exp.infra.clear_job()
    out = exp.run()
    print(out)
