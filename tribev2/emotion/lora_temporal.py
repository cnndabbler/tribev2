# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""LoRA fine-tuning with temporal emotion head (attentive pooling + BiLSTM).

Combines:
- LoRA backbone adaptation (+5% from lora_train.py)
- Temporal head modeling (+5% from cached_train.py hybrid head)

Instead of mean-pooling the backbone output into a single vector, this
preserves the full temporal sequence and uses a hybrid head (self-attentive
pooling + BiLSTM) to capture emotionally salient frames.

Usage:
    DATAPATH=... SAVEPATH=... LORA_R=128 uv run python -m tribev2.emotion.lora_temporal
"""

import logging
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

BACKBONE_NAME = "facebook/w2v-bert-2.0"
NUM_CLASSES = 6
EMOTION_LABELS = ["neutral", "happy", "sad", "angry", "fear", "disgust"]

LORA_R = int(os.getenv("LORA_R", "128"))
LORA_ALPHA = LORA_R * 2
LORA_DROPOUT = 0.15
LORA_TARGET_MODULES = ["linear_q", "linear_k", "linear_v", "linear_out"]
LORA_LAYERS = list(range(18, 24))

BATCH_SIZE = 8
LR_HEAD = 5e-4
LR_LORA = 5e-5
EPOCHS = 40
PATIENCE = 12
MAX_AUDIO_SEC = 4.0
SAMPLE_RATE = 16000
HEAD_HIDDEN = 128  # smaller to prevent overfitting (3.3M params)

DATADIR = os.getenv("DATAPATH", "/home/didierlacroix1/data/emotion")
SAVEDIR = os.path.join(
    os.getenv("SAVEPATH", "/home/didierlacroix1/data/emotion_save"),
    "results", f"tribe_emotion_lora_temporal_r{LORA_R}",
)
Path(SAVEDIR).mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class EmotionAudioDataset(Dataset):
    def __init__(self, clips: list[dict], max_samples: int):
        self.clips = clips
        self.max_samples = max_samples
        self.label_map = {e: i for i, e in enumerate(EMOTION_LABELS)}

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx):
        clip = self.clips[idx]
        import soundfile
        audio, sr = soundfile.read(clip["filepath"])
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != SAMPLE_RATE:
            from scipy.signal import resample
            n = int(len(audio) * SAMPLE_RATE / sr)
            audio = resample(audio, n)
        audio = audio.astype(np.float32)
        if len(audio) > self.max_samples:
            audio = audio[:self.max_samples]
        elif len(audio) < self.max_samples:
            audio = np.pad(audio, (0, self.max_samples - len(audio)))
        label = self.label_map[clip["emotion"]]
        return torch.tensor(audio), torch.tensor(label, dtype=torch.long)


def load_clips() -> tuple[list[dict], list[dict]]:
    from tribev2.studies.emotion_audio import CremadEmotion, RavdessEmotion

    datapath = Path(DATADIR)
    train_clips, val_clips = [], []
    for StudyClass, subdir in [
        (RavdessEmotion, "RavdessEmotion"),
        (CremadEmotion, "CremadEmotion"),
    ]:
        study = StudyClass(path=datapath / subdir)
        for tl in study.iter_timelines():
            events = study._load_timeline_events(tl)
            split = events.iloc[0].get("split", "train")
            filepath = events.iloc[0]["filepath"]
            clip = {"filepath": filepath, "emotion": tl["emotion"], "split": split}
            (val_clips if split == "val" else train_clips).append(clip)
    return train_clips, val_clips


# ---------------------------------------------------------------------------
# Temporal head components
# ---------------------------------------------------------------------------

class SelfAttentivePool(nn.Module):
    def __init__(self, dim: int, hidden: int = 128):
        super().__init__()
        self.proj = nn.Linear(dim, hidden)
        self.score = nn.Linear(hidden, 1)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        scores = self.score(torch.tanh(self.proj(hidden))).squeeze(-1)  # (B, T)
        weights = F.softmax(scores, dim=1).unsqueeze(-1)  # (B, T, 1)
        return (hidden * weights).sum(dim=1)  # (B, D)


class HybridTemporalHead(nn.Module):
    """Attentive pooling + BiLSTM + classifier."""

    def __init__(self, feat_dim: int, hidden_dim: int, num_classes: int, dropout: float = 0.2):
        super().__init__()
        self.attn_pool = SelfAttentivePool(feat_dim)
        self.lstm = nn.LSTM(
            feat_dim, hidden_dim, num_layers=1,
            bidirectional=True, batch_first=True,
        )
        combined_dim = feat_dim + hidden_dim * 2
        self.classifier = nn.Sequential(
            nn.LayerNorm(combined_dim),
            nn.Dropout(dropout),
            nn.Linear(combined_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        # Attentive pooling branch
        attn_out = self.attn_pool(hidden)  # (B, D)

        # BiLSTM branch — take last timestep
        lstm_out, _ = self.lstm(hidden)  # (B, T, H*2)
        lstm_last = lstm_out[:, -1, :]  # (B, H*2)

        combined = torch.cat([attn_out, lstm_last], dim=-1)
        return self.classifier(combined)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class LoRATemporalModel(nn.Module):
    """Wav2VecBert + LoRA + hybrid temporal head."""

    def __init__(self, backbone, num_classes: int, hidden_size: int, head_hidden: int):
        super().__init__()
        self.backbone = backbone
        self.head = HybridTemporalHead(hidden_size, head_hidden, num_classes)

    def forward(self, input_features):
        outputs = self.backbone(input_features, output_hidden_states=False)
        hidden = outputs.last_hidden_state  # (B, T, 1024)
        return self.head(hidden)  # (B, C)


def build_model(device: torch.device) -> tuple[LoRATemporalModel, int]:
    from peft import LoraConfig, get_peft_model
    from transformers import Wav2Vec2BertModel

    logger.info("Loading backbone: %s", BACKBONE_NAME)
    backbone = Wav2Vec2BertModel.from_pretrained(BACKBONE_NAME)
    hidden_size = backbone.config.hidden_size

    for param in backbone.parameters():
        param.requires_grad = False

    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=LORA_TARGET_MODULES,
        layers_to_transform=LORA_LAYERS,
        lora_dropout=LORA_DROPOUT,
        bias="none",
    )
    backbone = get_peft_model(backbone, lora_config)
    backbone.print_trainable_parameters()

    model = LoRATemporalModel(backbone, NUM_CLASSES, hidden_size, HEAD_HIDDEN)
    model.to(device)

    lora_params = sum(p.numel() for n, p in model.named_parameters() if "lora_" in n and p.requires_grad)
    head_params = sum(p.numel() for n, p in model.named_parameters() if "head." in n and p.requires_grad)
    logger.info("LoRA params: %d, Head params: %d, Total trainable: %d",
                lora_params, head_params, lora_params + head_params)

    return model, hidden_size


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, scheduler, feature_extractor, device):
    model.train()
    total_loss = 0.0
    n_batches = 0

    for audio, labels in tqdm(loader, desc="Train", leave=False):
        audio = audio.numpy()
        inputs = feature_extractor(
            audio, sampling_rate=SAMPLE_RATE, return_tensors="pt", padding=True,
        )
        input_features = inputs["input_features"].to(device)
        labels = labels.to(device)

        logits = model(input_features)
        loss = F.cross_entropy(logits, labels)

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
def evaluate(model, loader, feature_extractor, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds, all_labels = [], []

    for audio, labels in tqdm(loader, desc="Eval", leave=False):
        audio = audio.numpy()
        inputs = feature_extractor(
            audio, sampling_rate=SAMPLE_RATE, return_tensors="pt", padding=True,
        )
        input_features = inputs["input_features"].to(device)
        labels = labels.to(device)

        logits = model(input_features)
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)
    logger.info("Save dir: %s", SAVEDIR)
    logger.info("Head hidden: %d", HEAD_HIDDEN)

    # Load data
    train_clips, val_clips = load_clips()
    logger.info("Train: %d clips, Val: %d clips", len(train_clips), len(val_clips))

    max_samples = int(MAX_AUDIO_SEC * SAMPLE_RATE)
    train_ds = EmotionAudioDataset(train_clips, max_samples)
    val_ds = EmotionAudioDataset(val_clips, max_samples)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=0, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=0, pin_memory=True,
    )

    # Build model
    model, hidden_size = build_model(device)

    from transformers import AutoFeatureExtractor
    feature_extractor = AutoFeatureExtractor.from_pretrained(BACKBONE_NAME)

    # Optimizer with differential learning rates
    lora_params = [p for n, p in model.named_parameters() if "lora_" in n and p.requires_grad]
    head_params = [p for n, p in model.named_parameters() if "head." in n and p.requires_grad]

    optimizer = torch.optim.AdamW([
        {"params": lora_params, "lr": LR_LORA},
        {"params": head_params, "lr": LR_HEAD},
    ], weight_decay=0.01)

    total_steps = EPOCHS * len(train_loader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[LR_LORA, LR_HEAD],
        total_steps=total_steps,
        pct_start=0.1,
    )

    # Resume
    start_epoch = 0
    best_acc = 0.0
    patience_counter = 0
    ckpt_path = os.path.join(SAVEDIR, "checkpoint.pt")
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
    tb_dir = os.path.join(SAVEDIR, "tensorboard")
    writer = SummaryWriter(log_dir=tb_dir)

    if start_epoch == 0:
        writer.add_text("config", "\n".join([
            f"backbone: {BACKBONE_NAME}",
            f"head: hybrid_temporal (attn_pool + BiLSTM)",
            f"head_hidden: {HEAD_HIDDEN}",
            f"lora_r: {LORA_R}",
            f"lora_alpha: {LORA_ALPHA}",
            f"lora_layers: {LORA_LAYERS}",
            f"lr_head: {LR_HEAD}",
            f"lr_lora: {LR_LORA}",
            f"batch_size: {BATCH_SIZE}",
            f"lora_params: {sum(p.numel() for p in lora_params)}",
            f"head_params: {sum(p.numel() for p in head_params)}",
            f"train_clips: {len(train_clips)}",
            f"val_clips: {len(val_clips)}",
        ]))

    for epoch in range(start_epoch, EPOCHS):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler,
            feature_extractor, device,
        )
        val_loss, val_acc, val_f1 = evaluate(
            model, val_loader, feature_extractor, device,
        )

        logger.info(
            "Epoch %d/%d: train_loss=%.4f val_loss=%.4f val_acc=%.3f val_f1=%.3f",
            epoch + 1, EPOCHS, train_loss, val_loss, val_acc, val_f1,
        )

        writer.add_scalars("loss", {"train": train_loss, "val": val_loss}, epoch + 1)
        writer.add_scalar("accuracy/val", val_acc, epoch + 1)
        writer.add_scalar("f1/val", val_f1, epoch + 1)
        writer.add_scalar("accuracy/best", max(best_acc, val_acc), epoch + 1)
        for i, pg in enumerate(optimizer.param_groups):
            name = "lr/lora" if i == 0 else "lr/head"
            writer.add_scalar(name, pg["lr"], epoch + 1)

        if val_acc > best_acc:
            best_acc = val_acc
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_acc": val_acc,
                "val_f1": val_f1,
                "val_loss": val_loss,
            }, os.path.join(SAVEDIR, "best_lora.pt"))
            logger.info("  Best model saved (acc=%.3f, f1=%.3f)", val_acc, val_f1)
        else:
            patience_counter += 1

        # Resumable checkpoint
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_acc": best_acc,
            "patience_counter": patience_counter,
            "val_acc": val_acc,
            "val_f1": val_f1,
        }, ckpt_path)

        if patience_counter >= PATIENCE:
            logger.info("Early stopping at epoch %d", epoch + 1)
            break

    writer.close()

    # Final eval
    logger.info("Loading best model for final eval...")
    best_ckpt = torch.load(os.path.join(SAVEDIR, "best_lora.pt"), map_location=device)
    model.load_state_dict(best_ckpt["model_state_dict"])
    val_loss, val_acc, val_f1 = evaluate(model, val_loader, feature_extractor, device)
    logger.info("FINAL: val_acc=%.4f val_f1=%.4f val_loss=%.4f", val_acc, val_f1, val_loss)

    # Per-class accuracy
    model.eval()
    class_correct = [0] * NUM_CLASSES
    class_total = [0] * NUM_CLASSES
    with torch.no_grad():
        for audio, labels in val_loader:
            audio = audio.numpy()
            inputs = feature_extractor(
                audio, sampling_rate=SAMPLE_RATE, return_tensors="pt", padding=True,
            )
            input_features = inputs["input_features"].to(device)
            preds = model(input_features).argmax(dim=-1).cpu()
            for p, l in zip(preds, labels):
                class_total[l.item()] += 1
                if p.item() == l.item():
                    class_correct[l.item()] += 1

    logger.info("Per-class accuracy:")
    for i, name in enumerate(EMOTION_LABELS):
        if class_total[i] > 0:
            acc = class_correct[i] / class_total[i]
            logger.info("  %s: %.1f%% (%d/%d)", name, acc * 100, class_correct[i], class_total[i])

    # Merge LoRA and save
    logger.info("Merging LoRA weights...")
    model.backbone = model.backbone.merge_and_unload()
    torch.save({
        "model_state_dict": model.state_dict(),
        "val_acc": val_acc,
        "val_f1": val_f1,
        "num_classes": NUM_CLASSES,
        "emotion_labels": EMOTION_LABELS,
        "backbone_name": BACKBONE_NAME,
        "hidden_size": hidden_size,
        "head_hidden": HEAD_HIDDEN,
        "head_type": "hybrid_temporal",
    }, os.path.join(SAVEDIR, "merged_model.pt"))
    logger.info("Merged model saved.")


if __name__ == "__main__":
    main()
