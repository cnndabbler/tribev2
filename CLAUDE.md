# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

TRIBE v2 is a deep multimodal brain encoding model (Meta Research) that predicts fMRI responses to naturalistic stimuli (video, audio, text). It combines VJEPA2, DINOv2, Wav2Vec-BERT, and Llama-3.2 features via a Transformer encoder, mapping onto fsaverage5 cortical surface (~20k vertices).

## Commands

### Install
```bash
uv pip install -e ".[training,plotting,test]"
```

### Test (quick local run)
```bash
# Requires DATAPATH and SAVEPATH env vars
uv run python -m tribev2.grids.test_run
```

### Lint/Format
```bash
uv run black --check .
uv run isort --check .
```
Black line length: 88. isort profile: "black".

### Run experiments
```bash
# Local debug with default config
uv run python -m tribev2.grids.defaults

# Grid search on Slurm
uv run python -m tribev2.grids.run_cortical
uv run python -m tribev2.grids.run_subcortical
```

## Required Environment Variables

- `DATAPATH`: Root path to study datasets
- `SAVEPATH`: Root path for cache and results (creates `cache/` and `results/` subdirectories)
- `SLURM_PARTITION`, `SLURM_CONSTRAINT`: For Slurm cluster execution
- `WANDB_ENTITY`: For Weights & Biases logging

## Architecture

### Pipeline Flow
`TribeExperiment` (main.py) orchestrates the full pipeline:
1. **Data loading**: `MultiStudyLoader` (utils.py) loads multiple fMRI datasets, applies event transforms (speech-to-text, text annotation, chunking)
2. **Feature extraction**: Per-modality extractors (text, video, audio, image) run via neuralset/exca with Slurm caching
3. **Model**: `FmriEncoder` (model.py) — per-modality projectors → combiner → TransformerEncoder → per-subject prediction layers
4. **Training**: `BrainModule` (pl_module.py) wraps the model in PyTorch Lightning with MSE loss and Pearson correlation metrics

### Config System
- `grids/defaults.py`: Full default config dict (infrastructure, data, model, optimizer)
- `grids/configs.py`: Named overrides (`mini_config` for lighter models, `base_config` for full)
- Configs are plain dicts merged via `exca.ConfDict.update()`
- `grids/test_run.py`: Quick 3-epoch local test on Algonauts2025Bold

### Key Libraries (internal to Meta ecosystem)
- **neuralset**: Neural event/timeline data pipeline (extractors, segments, dataloaders)
- **neuraltrain**: Training framework (losses, metrics, models, optimizers, `BaseExperiment`)
- **exca**: Job infrastructure (`TaskInfra` for Slurm), `ConfDict` for config management

### Datasets (tribev2/studies/)
Each study class registers with neuralset. Supported: `Algonauts2025Bold`, `Wen2017`, `Lahner2024Bold`, `Lebel2023Bold`, `RavdessEmotion`, `CremadEmotion`, `IEMOCAPEmotion`.

### Feature Extractors
Configured in defaults.py. Each uses neuralset's extractor system with multi-layer extraction and Slurm-based caching:
- **Text**: HuggingFaceText (Llama-3.2-3B default, Qwen3-0.6B in mini)
- **Video**: HuggingFaceVideo wrapping VJEPA2
- **Image**: HuggingFaceImage wrapping DINOv2-large
- **Audio**: Wav2VecBert
- **fMRI**: FmriExtractor with TribeSurfaceProjector (fsaverage5 mesh, 5s hemodynamic offset)

### Visualization (tribev2/plotting/)
Cortical surface plots via nilearn and PyVista. Subcortical visualization in subcortical.py.

## Emotion Classification (tribev2/emotion/)

Speech emotion recognition pipeline for Care Whisper elderly care smart glasses. 6 classes: neutral, happy, sad, angry, fear, disgust.

### Production model
LoRA r128 + elderly voice augmentation on RAVDESS+CREMA-D. 69.4% val, 70.7% TESS (out-of-domain), 96.0% fear. Deployed as care-whisper-ser sidecar on port 3031.

### Training commands
```bash
# LoRA fine-tuning (production pipeline)
DATAPATH=... SAVEPATH=... LORA_R=128 uv run python -m tribev2.emotion.lora_train

# Generate elderly voice augmented clips
DATAPATH=... uv run python -m tribev2.emotion.augment_elderly

# Fine-tune with augmented data (from existing checkpoint)
DATAPATH=... SAVEPATH=... LORA_R=128 uv run python -m tribev2.emotion.lora_train_augmented

# Cached feature extraction + fast head training
DATAPATH=... SAVEPATH=... uv run python -m tribev2.emotion.cache_features
SAVEPATH=... uv run python -m tribev2.emotion.cached_train --head hybrid --modality both
```

### Key files
- `emotion/lora_train.py`: LoRA fine-tuning with mean-pool MLP head (production architecture)
- `emotion/augment_elderly.py`: Elderly voice augmentation (pitch shift, time stretch, tremor)
- `emotion/lora_train_augmented.py`: Fine-tuning with augmented data
- `emotion/cache_features.py`: Extract and cache Wav2VecBert + Qwen3-0.6B features
- `emotion/cached_train.py`: Fast temporal head training on cached features
- `studies/emotion_audio.py`: RavdessEmotion, CremadEmotion, IEMOCAPEmotion study classes
- `doc/emotion_experiment_report.md`: Full experiment report with results and dataset evaluations

### TESS evaluation
Out-of-domain test on 2,400 unseen TESS clips (2 speakers: 64yo + 26yo). Run via care-whisper-ser sidecar:
```bash
cd ~/containers/care-whisper-ser && uv run python tests/test_tess_full.py
```

## Python Version
Requires Python >= 3.11. Uses `|` union syntax and modern type hints.

## License
CC-BY-NC 4.0 (non-commercial use only).
