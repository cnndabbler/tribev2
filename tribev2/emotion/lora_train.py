# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""LoRA fine-tuning of Wav2VecBert backbone for emotion classification.

Unlike the cached-feature pipeline, this runs the backbone end-to-end
during training with LoRA adapters on the top transformer layers.
After training, adapters are merged back into the backbone for
zero-overhead inference.

Usage:
    DATAPATH=... SAVEPATH=... uv run python -m tribev2.emotion.lora_train
"""

import logging
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

BACKBONE_NAME = "facebook/w2v-bert-2.0"
NUM_CLASSES = 6
EMOTION_LABELS = ["neutral", "happy", "sad", "angry", "fear", "disgust"]

# LoRA config — overridable via env vars
LORA_R = int(os.getenv("LORA_R", "64"))
LORA_ALPHA = LORA_R * 2  # standard scaling: alpha = 2 * rank
LORA_DROPOUT = 0.15
LORA_TARGET_MODULES = ["linear_q", "linear_k", "linear_v", "linear_out"]  # all attention projections
LORA_LAYERS = list(range(18, 24))  # top 6 of 24 layers

# Training config
BATCH_SIZE = 8
LR_HEAD = 5e-4
LR_LORA = 5e-5
EPOCHS = 40
PATIENCE = 12
WARMUP_STEPS = 100
MAX_AUDIO_SEC = 4.0  # pad/truncate to fixed length
SAMPLE_RATE = 16000

DATADIR = os.getenv("DATAPATH", "/home/didierlacroix1/data/emotion")
SAVEDIR = os.path.join(
    os.getenv("SAVEPATH", "/home/didierlacroix1/data/emotion_save"),
    "results", f"tribe_emotion_lora_r{LORA_R}",
)
Path(SAVEDIR).mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Dataset: loads raw audio + emotion label
# ---------------------------------------------------------------------------

class EmotionAudioDataset(Dataset):
    """Simple dataset that loads raw audio and returns (waveform, label)."""

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

        # Resample if needed
        if sr != SAMPLE_RATE:
            from scipy.signal import resample
            n = int(len(audio) * SAMPLE_RATE / sr)
            audio = resample(audio, n)

        audio = audio.astype(np.float32)

        # Pad or truncate to fixed length
        if len(audio) > self.max_samples:
            audio = audio[:self.max_samples]
        elif len(audio) < self.max_samples:
            audio = np.pad(audio, (0, self.max_samples - len(audio)))

        label = self.label_map[clip["emotion"]]
        return torch.tensor(audio), torch.tensor(label, dtype=torch.long)


def load_clips() -> tuple[list[dict], list[dict]]:
    """Load clip metadata from the Study classes."""
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
            if split == "val":
                val_clips.append(clip)
            else:
                train_clips.append(clip)

    return train_clips, val_clips


# ---------------------------------------------------------------------------
# Model: Backbone + LoRA + classification head
# ---------------------------------------------------------------------------

class LoRAEmotionModel(nn.Module):
    """Wav2VecBert with LoRA adapters + mean pooling + linear head."""

    def __init__(self, backbone, num_classes: int, hidden_size: int):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )

    def forward(self, input_features):
        outputs = self.backbone(input_features, output_hidden_states=False)
        # Mean pool over time
        hidden = outputs.last_hidden_state  # (B, T, H)
        pooled = hidden.mean(dim=1)  # (B, H)
        return self.head(pooled)  # (B, C)


def build_model(device: torch.device) -> tuple[LoRAEmotionModel, int]:
    """Build the backbone + LoRA + head model."""
    from peft import LoraConfig, get_peft_model
    from transformers import Wav2Vec2BertModel

    logger.info("Loading backbone: %s", BACKBONE_NAME)
    backbone = Wav2Vec2BertModel.from_pretrained(BACKBONE_NAME)
    hidden_size = backbone.config.hidden_size  # 1024

    # Freeze everything first
    for param in backbone.parameters():
        param.requires_grad = False

    # Apply LoRA to top layers
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

    model = LoRAEmotionModel(backbone, NUM_CLASSES, hidden_size)
    model.to(device)
    return model, hidden_size


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    feature_extractor,
    device: torch.device,
) -> float:
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
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    feature_extractor,
    device: torch.device,
) -> tuple[float, float, float]:
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

    # Weighted F1
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

    # Load data
    logger.info("Loading clips...")
    train_clips, val_clips = load_clips()
    logger.info("Train: %d clips, Val: %d clips", len(train_clips), len(val_clips))

    max_samples = int(MAX_AUDIO_SEC * SAMPLE_RATE)
    train_ds = EmotionAudioDataset(train_clips, max_samples)
    val_ds = EmotionAudioDataset(val_clips, max_samples)

    # num_workers=0 to avoid fork+CUDA deadlock
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

    # Feature extractor (preprocessing only, no model)
    from transformers import AutoFeatureExtractor
    feature_extractor = AutoFeatureExtractor.from_pretrained(BACKBONE_NAME)

    # Optimizer with differential learning rates
    lora_params = [p for n, p in model.named_parameters() if "lora_" in n and p.requires_grad]
    head_params = [p for n, p in model.named_parameters() if "head." in n and p.requires_grad]
    logger.info("LoRA params: %d, Head params: %d",
                sum(p.numel() for p in lora_params),
                sum(p.numel() for p in head_params))

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

    # Resume from checkpoint if available
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
    logger.info("TensorBoard logs: %s", tb_dir)

    if start_epoch == 0:
        writer.add_text("config", "\n".join([
            f"backbone: {BACKBONE_NAME}",
            f"lora_r: {LORA_R}",
            f"lora_alpha: {LORA_ALPHA}",
            f"lora_dropout: {LORA_DROPOUT}",
            f"lora_targets: {LORA_TARGET_MODULES}",
            f"lora_layers: {LORA_LAYERS}",
            f"lr_head: {LR_HEAD}",
            f"lr_lora: {LR_LORA}",
            f"batch_size: {BATCH_SIZE}",
            f"max_audio_sec: {MAX_AUDIO_SEC}",
            f"lora_params: {sum(p.numel() for p in lora_params)}",
            f"head_params: {sum(p.numel() for p in head_params)}",
            f"train_clips: {len(train_clips)}",
            f"val_clips: {len(val_clips)}",
        ]))

    # Training loop
    global_step = 0

    for epoch in range(start_epoch, EPOCHS):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler,
            feature_extractor, device,
        )
        val_loss, val_acc, val_f1 = evaluate(
            model, val_loader, feature_extractor, device,
        )
        global_step = (epoch + 1) * len(train_loader)

        logger.info(
            "Epoch %d/%d: train_loss=%.4f val_loss=%.4f val_acc=%.3f val_f1=%.3f",
            epoch + 1, EPOCHS, train_loss, val_loss, val_acc, val_f1,
        )

        # TensorBoard logging
        writer.add_scalars("loss", {"train": train_loss, "val": val_loss}, epoch + 1)
        writer.add_scalar("accuracy/val", val_acc, epoch + 1)
        writer.add_scalar("f1/val", val_f1, epoch + 1)
        writer.add_scalar("accuracy/best", max(best_acc, val_acc), epoch + 1)

        # Log learning rates
        for i, pg in enumerate(optimizer.param_groups):
            name = "lr/lora" if i == 0 else "lr/head"
            writer.add_scalar(name, pg["lr"], epoch + 1)

        if val_acc > best_acc:
            best_acc = val_acc
            patience_counter = 0
            best_path = os.path.join(SAVEDIR, "best_lora.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_acc": val_acc,
                "val_f1": val_f1,
                "val_loss": val_loss,
            }, best_path)
            logger.info("  Best model saved (acc=%.3f, f1=%.3f)", val_acc, val_f1)
        else:
            patience_counter += 1

        # Save resumable checkpoint every epoch
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_acc": best_acc,
            "patience_counter": patience_counter,
            "val_acc": val_acc,
            "val_f1": val_f1,
            "val_loss": val_loss,
        }, ckpt_path)

        if patience_counter >= PATIENCE:
            logger.info("Early stopping at epoch %d", epoch + 1)
            break

    writer.add_hparams(
        {
            "lora_r": LORA_R,
            "lora_alpha": LORA_ALPHA,
            "lr_head": LR_HEAD,
            "lr_lora": LR_LORA,
            "dropout": LORA_DROPOUT,
            "n_layers": len(LORA_LAYERS),
            "n_targets": len(LORA_TARGET_MODULES),
        },
        {
            "hparam/best_acc": best_acc,
            "hparam/best_f1": val_f1,
        },
    )
    writer.close()

    # Final evaluation with best model
    logger.info("Loading best model for final eval...")
    best_ckpt = torch.load(os.path.join(SAVEDIR, "best_lora.pt"), map_location=device)
    model.load_state_dict(best_ckpt["model_state_dict"])
    val_loss, val_acc, val_f1 = evaluate(model, val_loader, feature_extractor, device)
    logger.info("FINAL: val_acc=%.4f val_f1=%.4f val_loss=%.4f", val_acc, val_f1, val_loss)

    # Merge LoRA back into backbone and save
    logger.info("Merging LoRA weights into backbone...")
    model.backbone = model.backbone.merge_and_unload()
    merged_path = os.path.join(SAVEDIR, "merged_model.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "val_acc": val_acc,
        "val_f1": val_f1,
        "num_classes": NUM_CLASSES,
        "emotion_labels": EMOTION_LABELS,
        "backbone_name": BACKBONE_NAME,
        "hidden_size": hidden_size,
    }, merged_path)
    logger.info("Merged model saved to %s", merged_path)


if __name__ == "__main__":
    main()
