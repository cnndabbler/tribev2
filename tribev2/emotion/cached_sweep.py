# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Hyperparameter sweep on cached features.

Runs a grid of hidden_dim, lr, dropout, label_smoothing, modality_dropout
combinations using cached_train's infrastructure. Sub-second epochs means
the full sweep completes in minutes.

Usage:
    SAVEPATH=... uv run python -m tribev2.emotion.cached_sweep
    SAVEPATH=... uv run python -m tribev2.emotion.cached_sweep --modality audio
"""

import argparse
import itertools
import logging
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from tribev2.emotion.cached_train import (
    AUDIO_DIM,
    EMOTION_LABELS,
    NUM_CLASSES,
    CachedFeatureDataset,
    build_head,
    evaluate,
    train_one_epoch,
)

logger = logging.getLogger(__name__)

# Sweep grid
GRID = {
    "hidden_dim": [256, 512, 1024],
    "lr": [5e-4, 1e-3, 3e-3],
    "dropout": [0.1, 0.2, 0.3],
    "label_smoothing": [0.0, 0.1],
    "modality_dropout": [0.0, 0.15, 0.3],
}

BATCH_SIZE = 512
EPOCHS = 200
PATIENCE = 30
WEIGHT_DECAY = 0.01


def run_one(
    config: dict,
    train_ds: CachedFeatureDataset,
    val_ds: CachedFeatureDataset,
    device: torch.device,
    input_dim: int,
    modality: str,
) -> dict:
    """Train one config, return results."""
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=0, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=0, pin_memory=True,
    )

    model = build_head(
        input_dim, config["hidden_dim"], NUM_CLASSES, config["dropout"],
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["lr"], weight_decay=WEIGHT_DECAY,
    )
    total_steps = EPOCHS * len(train_loader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=config["lr"], total_steps=total_steps, pct_start=0.1,
    )

    audio_dim = AUDIO_DIM if modality == "both" else 0
    best_acc = 0.0
    best_f1 = 0.0
    best_epoch = 0
    patience_counter = 0

    for epoch in range(EPOCHS):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, device,
            modality_dropout=config["modality_dropout"] if modality == "both" else 0.0,
            audio_dim=audio_dim,
            label_smoothing=config["label_smoothing"],
        )
        val_loss, val_acc, val_f1 = evaluate(model, val_loader, device)

        if val_acc > best_acc:
            best_acc = val_acc
            best_f1 = val_f1
            best_epoch = epoch + 1
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                break

    return {
        **config,
        "best_acc": best_acc,
        "best_f1": best_f1,
        "best_epoch": best_epoch,
        "final_epoch": epoch + 1,
    }


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="Hyperparameter sweep on cached features")
    parser.add_argument(
        "--modality", choices=["audio", "text", "both"], default="both",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    savepath = os.getenv("SAVEPATH", "/home/didierlacroix1/data/emotion_save")
    cachedir = os.path.join(savepath, "cache", "cached_features")

    cache_files = []
    for name in ["ravdess", "cremad", "iemocap"]:
        p = os.path.join(cachedir, f"{name}.pt")
        if os.path.exists(p):
            cache_files.append(p)
    assert cache_files, "No cache files found."

    train_ds = CachedFeatureDataset(cache_files, "train", args.modality)
    val_ds = CachedFeatureDataset(cache_files, "val", args.modality)
    input_dim = train_ds.input_dim

    # For non-both modality, skip modality_dropout sweep
    grid = dict(GRID)
    if args.modality != "both":
        grid["modality_dropout"] = [0.0]

    keys = list(grid.keys())
    combos = list(itertools.product(*[grid[k] for k in keys]))
    logger.info("Sweep: %d configurations, modality=%s, input_dim=%d", len(combos), args.modality, input_dim)

    results = []
    for i, values in enumerate(combos):
        config = dict(zip(keys, values))
        tag = " ".join(f"{k}={v}" for k, v in config.items())
        logger.info("[%d/%d] %s", i + 1, len(combos), tag)

        result = run_one(config, train_ds, val_ds, device, input_dim, args.modality)
        results.append(result)
        logger.info(
            "  -> acc=%.3f f1=%.3f (epoch %d/%d)",
            result["best_acc"], result["best_f1"], result["best_epoch"], result["final_epoch"],
        )

    # Sort by accuracy
    results.sort(key=lambda r: r["best_acc"], reverse=True)

    logger.info("")
    logger.info("=" * 80)
    logger.info("TOP 10 RESULTS (modality=%s)", args.modality)
    logger.info("=" * 80)
    for i, r in enumerate(results[:10]):
        logger.info(
            "#%d  acc=%.3f  f1=%.3f  hidden=%d lr=%.4f drop=%.2f ls=%.2f md=%.2f  (epoch %d)",
            i + 1, r["best_acc"], r["best_f1"],
            r["hidden_dim"], r["lr"], r["dropout"],
            r["label_smoothing"], r["modality_dropout"], r["best_epoch"],
        )

    # Save results
    savedir = os.path.join(savepath, "results", f"tribe_emotion_cached_sweep_{args.modality}")
    Path(savedir).mkdir(parents=True, exist_ok=True)
    torch.save(results, os.path.join(savedir, "sweep_results.pt"))
    logger.info("Results saved to %s", savedir)


if __name__ == "__main__":
    main()
