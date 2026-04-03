# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Fast temporal head training on cached backbone features (audio + text).

Pre-requisite: run cache_features.py to extract full-sequence features.

Replaces mean pooling + MLP with temporal-aware architectures:
  --head attn_pool   : Self-attentive pooling + MLP (quick win, ~+3% F1)
  --head bilstm      : BiLSTM on raw sequences + MLP
  --head hybrid      : Attentive pooling + BiLSTM + MLP (default)

Usage:
    SAVEPATH=... uv run python -m tribev2.emotion.cached_train
    SAVEPATH=... uv run python -m tribev2.emotion.cached_train --head attn_pool
    SAVEPATH=... uv run python -m tribev2.emotion.cached_train --modality audio --head bilstm
"""

import argparse
import logging
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config (env-overridable)
# ---------------------------------------------------------------------------

EMOTION_LABELS = ["neutral", "happy", "sad", "angry", "fear", "disgust"]
NUM_CLASSES = 6
FEAT_DIM = 1024  # both Wav2VecBert and Qwen3-0.6B output 1024

BATCH_SIZE = int(os.getenv("BATCH_SIZE", "64"))
LR = float(os.getenv("LR", "1e-3"))
EPOCHS = int(os.getenv("EPOCHS", "200"))
PATIENCE = int(os.getenv("PATIENCE", "30"))
HIDDEN_DIM = int(os.getenv("HIDDEN_DIM", "256"))
DROPOUT = float(os.getenv("DROPOUT", "0.2"))
LABEL_SMOOTHING = float(os.getenv("LABEL_SMOOTHING", "0.1"))
WEIGHT_DECAY = float(os.getenv("WEIGHT_DECAY", "0.01"))
MODALITY_DROPOUT = float(os.getenv("MODALITY_DROPOUT", "0.3"))

SAVEPATH = os.getenv("SAVEPATH", "/home/didierlacroix1/data/emotion_save")


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class CachedSequenceDataset(Dataset):
    """Load pre-computed sequence features from .pt cache files."""

    def __init__(
        self,
        cache_files: list[str],
        split: str,
        modality: str = "both",
    ):
        self.modality = modality
        all_audio, all_audio_len = [], []
        all_text, all_text_len = [], []
        all_labels = []

        for path in cache_files:
            data = torch.load(path, map_location="cpu")
            mask = torch.tensor([s == split for s in data["splits"]])

            if modality in ("audio", "both"):
                assert "audio_feats" in data, f"No audio_feats in {path}"
                all_audio.append(data["audio_feats"][mask].float())
                all_audio_len.append(data["audio_lengths"][mask])
            if modality in ("text", "both"):
                assert "text_feats" in data, f"No text_feats in {path}"
                all_text.append(data["text_feats"][mask].float())
                all_text_len.append(data["text_lengths"][mask])
            all_labels.append(data["labels"][mask])

        self.audio = torch.cat(all_audio, dim=0) if all_audio else None
        self.audio_len = torch.cat(all_audio_len, dim=0) if all_audio_len else None
        self.text = torch.cat(all_text, dim=0) if all_text else None
        self.text_len = torch.cat(all_text_len, dim=0) if all_text_len else None
        self.labels = torch.cat(all_labels, dim=0)

        logger.info(
            "%s set: %d samples | audio=%s text=%s",
            split, len(self.labels),
            self.audio.shape if self.audio is not None else None,
            self.text.shape if self.text is not None else None,
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {"label": self.labels[idx]}
        if self.audio is not None:
            item["audio"] = self.audio[idx]
            item["audio_len"] = self.audio_len[idx]
        if self.text is not None:
            item["text"] = self.text[idx]
            item["text_len"] = self.text_len[idx]
        return item


# ---------------------------------------------------------------------------
# Model components
# ---------------------------------------------------------------------------

class SelfAttentivePool(nn.Module):
    """Self-attentive pooling: learns to weight salient frames.

    h_i -> score_i = w^T tanh(W h_i + b)
    alpha = softmax(scores)
    output = sum(alpha_i * h_i)
    """

    def __init__(self, dim: int, hidden: int = 128):
        super().__init__()
        self.proj = nn.Linear(dim, hidden)
        self.score = nn.Linear(hidden, 1)

    def forward(self, hidden: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden: (B, T, D)
            lengths: (B,) valid frame counts
        Returns:
            pooled: (B, D)
        """
        B, T, D = hidden.shape
        scores = self.score(torch.tanh(self.proj(hidden))).squeeze(-1)  # (B, T)

        # Mask padding positions
        mask = torch.arange(T, device=hidden.device).unsqueeze(0) >= lengths.unsqueeze(1)
        scores = scores.masked_fill(mask, float("-inf"))

        weights = F.softmax(scores, dim=1).unsqueeze(-1)  # (B, T, 1)
        return (hidden * weights).sum(dim=1)  # (B, D)


class AttnPoolHead(nn.Module):
    """Attentive pooling + MLP classifier."""

    def __init__(self, feat_dim: int, hidden_dim: int, num_classes: int, dropout: float):
        super().__init__()
        self.pool = SelfAttentivePool(feat_dim)
        self.classifier = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Dropout(dropout),
            nn.Linear(feat_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, hidden: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        pooled = self.pool(hidden, lengths)
        return self.classifier(pooled)


class BiLSTMHead(nn.Module):
    """BiLSTM temporal encoder + MLP classifier."""

    def __init__(self, feat_dim: int, hidden_dim: int, num_classes: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(
            feat_dim, hidden_dim, num_layers=2,
            bidirectional=True, batch_first=True, dropout=dropout,
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, hidden: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        packed = pack_padded_sequence(
            hidden, lengths.cpu().clamp(min=1), batch_first=True, enforce_sorted=False,
        )
        out, _ = self.lstm(packed)
        out, _ = pad_packed_sequence(out, batch_first=True)
        # Take the output at the last valid timestep for each sequence
        idx = (lengths - 1).clamp(min=0).long()
        last = out[torch.arange(out.size(0), device=out.device), idx]
        return self.classifier(last)


class HybridHead(nn.Module):
    """Attentive pooling + BiLSTM + MLP classifier.

    Combines attention-weighted representation with temporal modeling.
    """

    def __init__(self, feat_dim: int, hidden_dim: int, num_classes: int, dropout: float):
        super().__init__()
        self.attn_pool = SelfAttentivePool(feat_dim)
        self.lstm = nn.LSTM(
            feat_dim, hidden_dim, num_layers=1,
            bidirectional=True, batch_first=True, dropout=0.0,
        )
        # Concat attn_pool (feat_dim) + lstm last (hidden_dim*2)
        combined_dim = feat_dim + hidden_dim * 2
        self.classifier = nn.Sequential(
            nn.LayerNorm(combined_dim),
            nn.Dropout(dropout),
            nn.Linear(combined_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, hidden: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        # Attentive pooling branch
        attn_out = self.attn_pool(hidden, lengths)

        # BiLSTM branch
        packed = pack_padded_sequence(
            hidden, lengths.cpu().clamp(min=1), batch_first=True, enforce_sorted=False,
        )
        lstm_out, _ = self.lstm(packed)
        lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True)
        idx = (lengths - 1).clamp(min=0).long()
        lstm_last = lstm_out[torch.arange(lstm_out.size(0), device=lstm_out.device), idx]

        combined = torch.cat([attn_out, lstm_last], dim=-1)
        return self.classifier(combined)


class MultiModalTemporalHead(nn.Module):
    """Wraps a temporal head for multi-modal input.

    Each modality gets its own temporal encoder, then features are
    concatenated before the classifier.
    """

    def __init__(
        self,
        head_type: str,
        feat_dim: int,
        hidden_dim: int,
        num_classes: int,
        dropout: float,
        modalities: list[str],
    ):
        super().__init__()
        self.modalities = modalities
        head_cls = {"attn_pool": AttnPoolHead, "bilstm": BiLSTMHead, "hybrid": HybridHead}[head_type]

        if len(modalities) == 1:
            self.head = head_cls(feat_dim, hidden_dim, num_classes, dropout)
        else:
            # Per-modality encoders (without classifier)
            self.encoders = nn.ModuleDict()
            for mod in modalities:
                enc = head_cls(feat_dim, hidden_dim, 0, dropout)  # placeholder
                # Remove classifier, we'll use a shared one
                self.encoders[mod] = enc

            # Determine per-encoder output dim
            if head_type == "attn_pool":
                enc_out_dim = feat_dim
            elif head_type == "bilstm":
                enc_out_dim = hidden_dim * 2
            else:  # hybrid
                enc_out_dim = feat_dim + hidden_dim * 2

            combined_dim = enc_out_dim * len(modalities)
            self.classifier = nn.Sequential(
                nn.LayerNorm(combined_dim),
                nn.Dropout(dropout),
                nn.Linear(combined_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes),
            )

    def _encode_single(self, encoder, hidden, lengths, head_type):
        """Run encoder but return pre-classifier features."""
        if head_type == "attn_pool":
            return encoder.pool(hidden, lengths)
        elif head_type == "bilstm":
            packed = pack_padded_sequence(
                hidden, lengths.cpu().clamp(min=1), batch_first=True, enforce_sorted=False,
            )
            out, _ = encoder.lstm(packed)
            out, _ = pad_packed_sequence(out, batch_first=True)
            idx = (lengths - 1).clamp(min=0).long()
            return out[torch.arange(out.size(0), device=out.device), idx]
        else:  # hybrid
            attn_out = encoder.attn_pool(hidden, lengths)
            packed = pack_padded_sequence(
                hidden, lengths.cpu().clamp(min=1), batch_first=True, enforce_sorted=False,
            )
            lstm_out, _ = encoder.lstm(packed)
            lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True)
            idx = (lengths - 1).clamp(min=0).long()
            lstm_last = lstm_out[torch.arange(lstm_out.size(0), device=lstm_out.device), idx]
            return torch.cat([attn_out, lstm_last], dim=-1)

    def forward(self, batch: dict, head_type: str, modality_dropout: float = 0.0) -> torch.Tensor:
        if len(self.modalities) == 1:
            mod = self.modalities[0]
            return self.head(batch[f"{mod}_hidden"], batch[f"{mod}_len"])

        features = []
        for mod in self.modalities:
            feat = self._encode_single(
                self.encoders[mod], batch[f"{mod}_hidden"], batch[f"{mod}_len"], head_type,
            )
            # Modality dropout during training
            if self.training and modality_dropout > 0 and torch.rand(1).item() < modality_dropout:
                feat = torch.zeros_like(feat)
            features.append(feat)

        combined = torch.cat(features, dim=-1)
        return self.classifier(combined)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: MultiModalTemporalHead,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
    head_type: str,
    modality_dropout: float = 0.0,
    label_smoothing: float = 0.1,
) -> float:
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        batch_gpu = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch_gpu[k] = v.to(device)
            else:
                batch_gpu[k] = v
        labels = batch_gpu.pop("label")

        # Rename keys for the model
        model_batch = {}
        for k, v in batch_gpu.items():
            if k == "audio":
                model_batch["audio_hidden"] = v
            elif k == "audio_len":
                model_batch["audio_len"] = v
            elif k == "text":
                model_batch["text_hidden"] = v
            elif k == "text_len":
                model_batch["text_len"] = v

        logits = model(model_batch, head_type, modality_dropout=modality_dropout)
        loss = F.cross_entropy(logits, labels, label_smoothing=label_smoothing)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(
    model: MultiModalTemporalHead,
    loader: DataLoader,
    device: torch.device,
    head_type: str,
) -> tuple[float, float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds, all_labels = [], []

    for batch in loader:
        batch_gpu = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch_gpu[k] = v.to(device)
            else:
                batch_gpu[k] = v
        labels = batch_gpu.pop("label")

        model_batch = {}
        for k, v in batch_gpu.items():
            if k == "audio":
                model_batch["audio_hidden"] = v
            elif k == "audio_len":
                model_batch["audio_len"] = v
            elif k == "text":
                model_batch["text_hidden"] = v
            elif k == "text_len":
                model_batch["text_len"] = v

        logits = model(model_batch, head_type)
        loss = F.cross_entropy(logits, labels)
        preds = logits.argmax(dim=-1)

        total_loss += loss.item() * labels.size(0)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    acc = correct / max(total, 1)
    avg_loss = total_loss / max(total, 1)

    from sklearn.metrics import f1_score
    f1 = f1_score(all_labels, all_preds, average="weighted")

    return avg_loss, acc, f1


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="Train temporal head on cached features")
    parser.add_argument(
        "--modality", choices=["audio", "text", "both"], default="both",
    )
    parser.add_argument(
        "--head", choices=["attn_pool", "bilstm", "hybrid"], default="hybrid",
        help="Head architecture (default: hybrid = attn_pool + BiLSTM)",
    )
    parser.add_argument(
        "--datasets", default="ravdess,cremad,iemocap",
        help="Comma-separated dataset names to use (default: ravdess,cremad,iemocap)",
    )
    args = parser.parse_args()

    dataset_names = [d.strip() for d in args.datasets.split(",")]
    dataset_tag = "+".join(dataset_names)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cachedir = os.path.join(SAVEPATH, "cache", "cached_features")
    savedir = os.path.join(SAVEPATH, "results", f"tribe_emotion_cached_{args.head}_{args.modality}_{dataset_tag}")
    Path(savedir).mkdir(parents=True, exist_ok=True)

    logger.info("Device: %s", device)
    logger.info("Head: %s, Modality: %s", args.head, args.modality)
    logger.info("Cache dir: %s", cachedir)
    logger.info("Save dir: %s", savedir)

    # Find cache files
    cache_files = []
    for name in dataset_names:
        p = os.path.join(cachedir, f"{name}.pt")
        if os.path.exists(p):
            cache_files.append(p)
            logger.info("Found cache: %s", p)
    assert cache_files, "No cache files found. Run cache_features.py first."

    # Build datasets
    train_ds = CachedSequenceDataset(cache_files, "train", args.modality)
    val_ds = CachedSequenceDataset(cache_files, "val", args.modality)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=0, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=0, pin_memory=True,
    )

    # Build model
    modalities = []
    if args.modality in ("audio", "both"):
        modalities.append("audio")
    if args.modality in ("text", "both"):
        modalities.append("text")

    model = MultiModalTemporalHead(
        head_type=args.head,
        feat_dim=FEAT_DIM,
        hidden_dim=HIDDEN_DIM,
        num_classes=NUM_CLASSES,
        dropout=DROPOUT,
        modalities=modalities,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    logger.info("Model params: %d", total_params)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    total_steps = EPOCHS * len(train_loader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=LR, total_steps=total_steps, pct_start=0.1,
    )

    # Resume
    start_epoch = 0
    best_acc = 0.0
    patience_counter = 0
    ckpt_path = os.path.join(savedir, "checkpoint.pt")
    if os.path.exists(ckpt_path):
        logger.info("Resuming from checkpoint: %s", ckpt_path)
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        if ckpt.get("optimizer_state_dict") is not None:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if ckpt.get("scheduler_state_dict") is not None:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_acc = ckpt["best_acc"]
        patience_counter = ckpt["patience_counter"]
        logger.info(
            "Resumed at epoch %d (best_acc=%.3f, patience=%d/%d)",
            start_epoch + 1, best_acc, patience_counter, PATIENCE,
        )

    # TensorBoard
    from torch.utils.tensorboard import SummaryWriter
    tb_dir = os.path.join(savedir, "tensorboard")
    writer = SummaryWriter(log_dir=tb_dir)

    if start_epoch == 0:
        writer.add_text("config", "\n".join([
            f"head: {args.head}",
            f"modality: {args.modality}",
            f"hidden_dim: {HIDDEN_DIM}",
            f"dropout: {DROPOUT}",
            f"lr: {LR}",
            f"batch_size: {BATCH_SIZE}",
            f"label_smoothing: {LABEL_SMOOTHING}",
            f"modality_dropout: {MODALITY_DROPOUT}",
            f"model_params: {total_params}",
            f"train_samples: {len(train_ds)}",
            f"val_samples: {len(val_ds)}",
        ]))

    for epoch in range(start_epoch, EPOCHS):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, device,
            head_type=args.head,
            modality_dropout=MODALITY_DROPOUT if args.modality == "both" else 0.0,
            label_smoothing=LABEL_SMOOTHING,
        )
        val_loss, val_acc, val_f1 = evaluate(model, val_loader, device, args.head)

        logger.info(
            "Epoch %d/%d: train=%.4f val=%.4f acc=%.3f f1=%.3f",
            epoch + 1, EPOCHS, train_loss, val_loss, val_acc, val_f1,
        )

        writer.add_scalars("loss", {"train": train_loss, "val": val_loss}, epoch + 1)
        writer.add_scalar("accuracy/val", val_acc, epoch + 1)
        writer.add_scalar("f1/val", val_f1, epoch + 1)

        if val_acc > best_acc:
            best_acc = val_acc
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_acc": val_acc,
                "val_f1": val_f1,
                "val_loss": val_loss,
                "head_type": args.head,
                "modality": args.modality,
                "hidden_dim": HIDDEN_DIM,
                "num_classes": NUM_CLASSES,
            }, os.path.join(savedir, "best_head.pt"))
            logger.info("  Best model saved (acc=%.3f, f1=%.3f)", val_acc, val_f1)
        else:
            patience_counter += 1

        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_acc": best_acc,
            "patience_counter": patience_counter,
        }, ckpt_path)

        if patience_counter >= PATIENCE:
            logger.info("Early stopping at epoch %d", epoch + 1)
            break

    writer.close()

    # Final eval with best model
    best_ckpt = torch.load(os.path.join(savedir, "best_head.pt"), map_location=device)
    model.load_state_dict(best_ckpt["model_state_dict"])
    val_loss, val_acc, val_f1 = evaluate(model, val_loader, device, args.head)
    logger.info("FINAL: val_acc=%.4f val_f1=%.4f val_loss=%.4f", val_acc, val_f1, val_loss)

    # Per-class accuracy
    model.eval()
    class_correct = [0] * NUM_CLASSES
    class_total = [0] * NUM_CLASSES
    with torch.no_grad():
        for batch in val_loader:
            batch_gpu = {}
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch_gpu[k] = v.to(device)
                else:
                    batch_gpu[k] = v
            labels = batch_gpu.pop("label")

            model_batch = {}
            for k, v in batch_gpu.items():
                if k == "audio":
                    model_batch["audio_hidden"] = v
                elif k == "audio_len":
                    model_batch["audio_len"] = v
                elif k == "text":
                    model_batch["text_hidden"] = v
                elif k == "text_len":
                    model_batch["text_len"] = v

            preds = model(model_batch, args.head).argmax(dim=-1)
            for p, l in zip(preds, labels):
                class_total[l.item()] += 1
                if p.item() == l.item():
                    class_correct[l.item()] += 1

    logger.info("Per-class accuracy:")
    for i, name in enumerate(EMOTION_LABELS):
        if class_total[i] > 0:
            acc = class_correct[i] / class_total[i]
            logger.info("  %s: %.1f%% (%d/%d)", name, acc * 100, class_correct[i], class_total[i])
        else:
            logger.info("  %s: N/A (0 samples)", name)


if __name__ == "__main__":
    main()
