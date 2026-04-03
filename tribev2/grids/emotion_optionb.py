# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Option B: Train with audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim.

This backbone was fine-tuned on MSP-Podcast for emotion recognition,
so its representations already encode arousal/valence/dominance dimensions.
Expected to outperform the general-purpose w2v-bert-2.0 (Option A).
"""

import os
from pathlib import Path

PROJECT_NAME = "tribe_emotion_optb"
NUM_CLASSES = 6

DATADIR = os.getenv("DATAPATH")
BASEDIR = os.getenv("SAVEPATH")
CACHEDIR = os.path.join(BASEDIR, "cache", PROJECT_NAME)
SAVEDIR = os.path.join(BASEDIR, "results", PROJECT_NAME)

for path in [CACHEDIR, SAVEDIR, DATADIR]:
    Path(path).mkdir(parents=True, exist_ok=True)

# Option B uses our patched Wav2VecEmotion extractor (handles vocab_size=None)
audio_feature = {
    "name": "Wav2VecEmotion",
    "model_name": "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim",
    "frequency": 2,
    "layers": [0.5, 0.75, 1.0],
    "event_types": "Audio",
    "aggregation": "sum",
    "infra": {
        "cluster": None,
        "folder": CACHEDIR,
        "keep_in_ram": True,
        "mode": "cached",
        "gpus_per_node": 1,
    },
}

optionb_config = {
    "infra": {
        "cluster": None,
        "folder": SAVEDIR,
        "gpus_per_node": 1,
        "cpus_per_task": 8,
        "mem_gb": 32,
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
            "names": ["RavdessEmotion", "CremadEmotion"],
            "path": DATADIR,
            "query": None,
            "transforms": {},
        },
        "features_to_use": ["audio"],
        "audio_feature": audio_feature,
        "batch_size": 32,
        "num_classes": NUM_CLASSES,
    },
    "wandb_config": None,
    "brain_model_config": {
        "name": "EmotionEncoder",
        "hidden": 256,
        "extractor_aggregation": "cat",
        "layer_aggregation": "cat",
        "combiner": None,
        "encoder": {"depth": 4, "heads": 4},
        "modality_dropout": 0.0,
        "dropout": 0.1,
        "num_classes": NUM_CLASSES,
    },
    "metrics": [
        {"log_name": "accuracy", "name": "MulticlassAccuracy",
         "kwargs": {"num_classes": NUM_CLASSES}},
        {"log_name": "f1", "name": "MulticlassF1Score",
         "kwargs": {"num_classes": NUM_CLASSES, "average": "weighted"}},
    ],
    "loss": {"name": "CrossEntropyLoss", "kwargs": {}},
    "optim": {
        "name": "LightningOptimizer",
        "optimizer": {"name": "Adam", "lr": 5e-4, "kwargs": {"weight_decay": 0.0}},
        "scheduler": {"name": "OneCycleLR", "kwargs": {"max_lr": 5e-4, "pct_start": 0.1}},
    },
    "n_epochs": 50,
    "monitor": "val/accuracy",
    "patience": 15,
    "enable_progress_bar": True,
    "log_every_n_steps": 5,
    "seed": 42,
}


if __name__ == "__main__":
    from ..emotion import ClassificationExperiment
    from ..studies.emotion_audio import *  # noqa: F401,F403

    exp = ClassificationExperiment(**optionb_config)
    exp.infra.clear_job()
    out = exp.run()
    print(out)
