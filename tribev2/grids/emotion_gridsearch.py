# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Grid search for speech emotion classification.

Strategies to improve over the 69% baseline:

1. **Architecture search**: encoder depth, hidden dim, dropout
2. **Layer selection**: which Wav2VecBert layers carry emotion signal
3. **Learning rate + schedule**: the most impactful single hyperparameter
4. **Class-weighted loss**: handle class imbalance across datasets
5. **Data augmentation**: via neuralset's audio transforms (future)

Run all grid points:
    DATAPATH=... SAVEPATH=... uv run python -m tribev2.grids.emotion_gridsearch

Run a single config by index:
    DATAPATH=... SAVEPATH=... uv run python -m tribev2.grids.emotion_gridsearch --index 0
"""

import argparse
import copy
import itertools
import os
import sys
from pathlib import Path

from exca import ConfDict

PROJECT_NAME = "tribe_emotion_grid"
NUM_CLASSES = 6

DATADIR = os.getenv("DATAPATH")
BASEDIR = os.getenv("SAVEPATH")
CACHEDIR = os.path.join(BASEDIR, "cache", PROJECT_NAME)
SAVEDIR = os.path.join(BASEDIR, "results", PROJECT_NAME)

for path in [CACHEDIR, SAVEDIR, DATADIR]:
    Path(path).mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Base config (shared across all grid points)
# ---------------------------------------------------------------------------

audio_feature = {
    "name": "Wav2VecBert",
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

base_config = {
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
        "dropout": 0.0,
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
        "optimizer": {"name": "Adam", "lr": 1e-3, "kwargs": {"weight_decay": 0.0}},
        "scheduler": {"name": "OneCycleLR", "kwargs": {"max_lr": 1e-3, "pct_start": 0.1}},
    },
    "n_epochs": 50,
    "monitor": "val/accuracy",
    "patience": 15,
    "enable_progress_bar": True,
    "log_every_n_steps": 5,
    "seed": 42,
}

# ---------------------------------------------------------------------------
# Grid search dimensions
# ---------------------------------------------------------------------------

GRID = {
    # Learning rate (most impactful)
    "optim.optimizer.lr": [5e-4, 1e-3, 2e-3],
    "optim.scheduler.kwargs.max_lr": [5e-4, 1e-3, 2e-3],

    # Encoder depth (capacity vs overfitting)
    "brain_model_config.encoder.depth": [2, 4, 8],

    # Hidden dimension
    "brain_model_config.hidden": [128, 256],

    # Dropout (regularization)
    "brain_model_config.dropout": [0.0, 0.1, 0.2],
}

# Paired parameters (lr and max_lr must match)
PAIRED_KEYS = [("optim.optimizer.lr", "optim.scheduler.kwargs.max_lr")]


def generate_grid_configs() -> list[tuple[str, dict]]:
    """Generate all grid point configs.

    Paired parameters are zipped (not crossed) so lr always matches max_lr.
    """
    # Separate paired from independent dimensions
    paired_values = {}
    independent_keys = []
    paired_done = set()

    for k1, k2 in PAIRED_KEYS:
        vals = list(zip(GRID[k1], GRID[k2]))
        paired_values[(k1, k2)] = vals
        paired_done.add(k1)
        paired_done.add(k2)

    for k in GRID:
        if k not in paired_done:
            independent_keys.append(k)

    # Build all combinations
    all_dims = []
    dim_keys = []

    for (k1, k2), vals in paired_values.items():
        all_dims.append(vals)
        dim_keys.append((k1, k2))

    for k in independent_keys:
        all_dims.append(GRID[k])
        dim_keys.append(k)

    configs = []
    for combo in itertools.product(*all_dims):
        updates = {}
        for key_spec, val in zip(dim_keys, combo):
            if isinstance(key_spec, tuple):
                for k, v in zip(key_spec, val):
                    updates[k] = v
            else:
                updates[key_spec] = val

        cfg = ConfDict(copy.deepcopy(base_config))
        cfg.update(updates)

        # Create a descriptive name
        parts = []
        for key_spec, val in zip(dim_keys, combo):
            if isinstance(key_spec, tuple):
                k = key_spec[0].split(".")[-1]
                parts.append(f"{k}={val[0]}")
            else:
                k = key_spec.split(".")[-1]
                parts.append(f"{k}={val}")

        name = "_".join(parts)
        cfg["infra"]["folder"] = os.path.join(SAVEDIR, name)
        configs.append((name, dict(cfg)))

    return configs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=int, default=None,
                        help="Run a single grid point by index")
    parser.add_argument("--list", action="store_true",
                        help="List all grid point names and exit")
    args = parser.parse_args()

    configs = generate_grid_configs()

    if args.list:
        for i, (name, _) in enumerate(configs):
            print(f"  [{i:3d}] {name}")
        print(f"\nTotal: {len(configs)} configurations")
        sys.exit(0)

    from ..emotion import ClassificationExperiment  # noqa: F401
    from ..studies.emotion_audio import *  # noqa: F401,F403

    if args.index is not None:
        configs = [configs[args.index]]
        print(f"Running single config: {configs[0][0]}")

    results = []
    for i, (name, cfg) in enumerate(configs):
        print(f"\n{'='*60}")
        print(f"[{i+1}/{len(configs)}] {name}")
        print(f"{'='*60}")
        try:
            exp = ClassificationExperiment(**cfg)
            exp.infra.clear_job()
            exp.run()
            results.append((name, "OK"))
        except Exception as e:
            print(f"FAILED: {e}")
            results.append((name, f"FAILED: {e}"))

    print(f"\n{'='*60}")
    print("GRID SEARCH RESULTS")
    print(f"{'='*60}")
    for name, status in results:
        print(f"  {name}: {status}")
