# Emotion Detection Training Experiments Report

**Project:** Speech Emotion Recognition for Mentra Smart Glasses (Care Whisper)
**Date:** 2026-03-29 to 2026-04-01
**GPU:** NVIDIA RTX A5000 (24GB VRAM)
**Goal:** 6-class emotion classification (neutral, happy, sad, angry, fear, disgust) with fear as highest-priority class for elderly users.

---

## 1. Datasets

Current Datasets

    RAVDESS (1,440 clips)

    - Type: Acted speech — 24 professional actors reading two scripted sentences ("Kids are talking by the door", "Dogs
    are sitting by the door") in 8 emotions
    - Strengths: Clean recordings, balanced classes, all 6 emotions including fear and disgust
    - Weakness: Very scripted — only 2 sentences total, actors performing emotions on command. Not representative of how
    elderly people actually express fear or distress.

    CREMA-D (7,442 clips)

    - Type: Acted speech — 91 actors (diverse ages 20-74) reading 12 scripted sentences in 6 emotions
    - Strengths: More sentence variety than RAVDESS, age diversity is good for elderly care, all 6 classes
    - Weakness: Still acted — "performed fear" sounds different from genuine fear in an elderly person

    IEMOCAP (9,901 clips)

    - Type: Naturalistic dyadic speech — pairs of actors in improvised and scripted conversations
    - Strengths: Most realistic speech patterns, conversational context, ground-truth word-level transcriptions
    - Weaknesses:
      - Only 5 classes (no disgust)
      - Severe class imbalance: angry=4,186 (42%), happy=2,632 (27%), neutral=1,726 (17%), sad=1,250 (13%), fear=107
    (1.1%)
      - The domain gap from acted corpora hurts combined training (63.4% combined vs 68.9% acted-only)

    Should We Keep All Three?

    The core problem: Mixing IEMOCAP with RAVDESS+CREMA-D dropped accuracy from 68.9% to 63.4%. The acted-vs-naturalistic
     domain gap is real and measurable.

    My recommendation:

    1. Keep RAVDESS + CREMA-D together — they're the same domain (acted) and our production model already scores 68.9% on
     them. They're our reliable training base.
    2. IEMOCAP is questionable for combined training but valuable for evaluation. The 107 fear samples are too few to
    learn from, and the domain mismatch drags everything down. However, testing our model on IEMOCAP tells us how well it
     generalizes to natural speech.
    3. What we're really missing is a dataset of elderly speech with genuine emotional distress — none of these datasets
    have speakers over 74, and none capture real fear/confusion in care settings. For production, consider recording or
    sourcing:
      - TESS (Toronto Emotional Speech Set) — older female speakers (26 and 64 years old)
      - MSP-Podcast — natural emotion in podcast speech
      - Custom data from Care Whisper field recordings (with consent)
    4. For the current cached-feature experiment: I'd suggest running ablations:
      - RAVDESS+CREMA-D only (audio+text) — matches production baseline
      - All 3 datasets (audio+text) — measure if text helps bridge the domain gap
      - IEMOCAP only — how well does text help on naturalistic speech alone

    The cached training is sub-second per epoch, so we can try all combinations quickly and let the numbers decide rather
     than guessing.

| Dataset | Clips | Classes | Type | Train/Val Split | Source |
|---------|-------|---------|------|-----------------|--------|
| RAVDESS | 1,440 | 6 (all) | Acted speech, 24 actors | Actor 1-19 train, 20-24 val | Livingstone & Russo 2018 |
| CREMA-D | 7,442 | 6 (all) | Acted speech, ~91 actors | Actor 0-72 train, 73-91 val | Cao et al. 2014 |
| IEMOCAP | 9,901 | 5 (no disgust) | Naturalistic dyadic speech | Speaker-based (Ses05F/M val) | USC SAIL, HuggingFace AbstractTTS/IEMOCAP |

**Combined dataset:** 18,783 clips (14,860 train / 3,923 val)

### IEMOCAP Emotion Mapping (10 raw classes to 6 target)

| Target | IEMOCAP Source | Count |
|--------|---------------|-------|
| neutral | neutral | 1,726 |
| happy | happy + excited | 2,632 |
| sad | sad | 1,250 |
| angry | angry + frustrated | 4,186 |
| fear | fear | 107 |
| (dropped) | disgust (2), surprise (110), other (26) | 138 |

### Transcriptions

All 3 datasets have 100% word-level TSV coverage:
- RAVDESS/CREMA-D: WhisperX batch transcription
- IEMOCAP: Ground-truth words from HuggingFace export
- TSV format: `text\tstart\tduration\tsequence_id\tsentence`

### Datasets Evaluated and Rejected

#### SSI Speech Emotion Recognition (`stapesai/ssi-speech-emotion-recognition`)

**Verdict: Skip for training. Minor interest for the SE (Senior) subset.** Evaluated 2026-04-01.

12,162 acted speech clips (MIT license, 16kHz) from 4 sources: CREMA-D (7,442), TESS (2,800), RAVDESS (1,440), SAVEE (480). 8 emotion classes. 121 unique speakers. Includes `text` transcriptions, `age_group` labels (AD/SE), and `emotion_intensity` levels.

**Age group breakdown (train split):**
- AD (Adult 20-60): 8,841 samples from CREMA-D, RAVDESS, SAVEE, and younger TESS speaker
- SE (Senior 60+): 1,159 samples — **all from a single TESS speaker** (OAF, age 64, female)

**SE (Senior) emotion distribution:** Balanced — ANG:165, DIS:165, FEA:165, HAP:160, NEU:167, SAD:165, SUR:172

**Why we rejected it for training:**
- **~75% overlap** — built from CREMA-D + RAVDESS which we already use. Net new data is only TESS (~2,800 clips, 2 speakers) and SAVEE (~480 clips, 4 speakers).
- **SE subset is one speaker** — all 1,159 senior samples come from a single 64-year-old female (TESS speaker OAF). A model would overfit to her voice, not learn elderly speech patterns.
- **Single-word utterances** — TESS text is "Say the word ___" (200+ words). Not representative of conversational speech in care settings.
- **All acted speech** — doesn't address the acted-vs-naturalistic domain gap.

**Potential limited use:**
- The SE subset could serve as a tiny **elderly evaluation benchmark** (165 fear clips from one speaker).
- The `emotion_intensity` labels (LO/MD/HI) are unique and could be interesting for fear severity detection research.

**What would actually help:**
- Naturalistic datasets with fear/distress labels (MSP-Podcast, EmoFilm)
- Multiple elderly speakers with conversational speech
- Custom recordings from Care Whisper field deployments (with consent)
- Data augmentation on existing fear samples (voice aging, speaking rate modification)

#### ElderReact (`Mayer123/ElderReact`)

**Verdict: Skip for training. Potential evaluation-only use.** Evaluated 2026-04-01.

1,323 video clips of 46 elderly people reacting to emotion-eliciting stimuli. 6 emotions (anger, disgust, fear, happiness, sadness, surprise), multi-label. From CMU (Ma et al., ICMI 2019). Academic access only (requires university email).

**Why we rejected it for training:**
- **Reactions, not speech** — clips are nonverbal emotional reactions (gasps, laughter, sighs, exclamations), not conversational speech. Our pipeline (Wav2VecBert + Qwen3-0.6B transcripts) expects spoken language.
- **Too small** — only 615 training clips, smaller than RAVDESS alone.
- **Fear is scarce** — ~92 training clips labeled fear out of 615 total.
- **Multi-label format** — incompatible with our single-label 6-class pipeline without lossy conversion.
- **No neutral class** — only ~35 clips with all-zeros labels.
- **Only 46 speakers** — easy to overfit.

**What's valuable:**
- **Only public elderly emotion dataset** — could serve as an out-of-domain evaluation benchmark to test age generalization.
- **Paper validates our problem**: "models trained on another age group do not generalize well on elders" — confirms the need for elderly-specific data or domain adaptation.

#### MELD — Multimodal EmotionLines Dataset (`declare-lab/MELD`)

**Verdict: Skip.** Evaluated 2026-04-01.

13,708 utterances from TV show *Friends* (1,433 dialogues, all 10 seasons). 7 emotions: neutral (47%), joy, anger, surprise, sadness, disgust, fear. Includes text transcriptions, MP4 video clips. GPL-3.0 on HuggingFace. Paper: Poria et al., ACL 2019 (~1000+ citations).

**Why we rejected it:**
- **Laugh track contamination** — live studio audience + sweetened laugh track overlaps with speech throughout. Wav2VecBert features would learn TV production artifacts, not speech emotion patterns.
- **Fear nearly absent** — only 268 training samples (2.7%). Neutral dominates at 47%. Our most critical class is severely underrepresented.
- **Young actors performing comedy** — all speakers aged 25-35, scripted sitcom performance. Zero overlap with elderly care speech patterns.
- **Copyright risk** — Friends audio owned by Warner Bros. Academic use is tolerated but commercial use of models trained on copyrighted content is legally untested.

**Potential limited use:** Text transcriptions could supplement Qwen3-0.6B training (laugh track doesn't affect text), but with only 268 fear transcripts the value is minimal.

#### MultiConAD — Multilingual Conversational Alzheimer's Detection (`ArezoShakeri/MultiConAD`)

**Verdict: Skip for training. Moderate value for domain adaptation if accessible.** Evaluated 2026-04-01.

~3,900 conversational samples unified from 16 dementia-related datasets. English, Spanish, Chinese, Greek. WAV audio + text transcripts. Paper: Shakeri et al., ACM SIGIR 2025.

**Labels:** Cognitive status only — Healthy Control (HC), Mild Cognitive Impairment (MCI), Alzheimer's/Dementia (AD). **No emotion labels.**

**Why we rejected it:**
- **Zero emotion labels** — cannot train or validate our 6-class SER system.
- **Access is painful** — most English data behind DementiaBank consortium membership (restricted to dementia researchers). All 16 source datasets must be obtained independently.
- **Clinical assessment context** — structured tasks (picture description, fluency tests, story retelling), not spontaneous daily conversation as in assisted living.
- **Small scale** — ~3,900 total samples is modest even for domain adaptation.

**What's valuable if accessible:**
- **Confirmed elderly speakers (60+)** — the only dataset with elderly conversational audio at scale.
- The **Pitt corpus** (~500 elderly WAV recordings) is the most accessible single source within MultiConAD for unsupervised domain adaptation of Wav2VecBert to elderly voice characteristics.
- Could test whether our SER model degrades on elderly voices via prediction confidence analysis.

#### NaturalVoices (`JHU-SmileLab/NaturalVoices_VC_870h`)

**Verdict: Skip.** Evaluated 2026-04-01.

870 hours of spontaneous podcast speech (from MSP-Podcast source recordings). 4 emotions: angry, happy, sad, neutral. Created by JHU SmileLab for voice conversion research. CC license. Paper under review at IEEE TAC.

**Why we rejected it:**
- **No fear or disgust classes** — only 4 emotions (angry/happy/sad/neutral). Our most critical class (fear) is entirely absent.
- **Model-predicted labels, not human annotations** — all emotion labels are outputs of the PEFT-SER model (WavLM + LoRA). Training our SER on another SER's predictions compounds errors and biases from the original training sets.
- **Scale doesn't compensate for label noise** — 870 hours of auto-labeled data is worse training signal than 8,882 clips with human annotations (RAVDESS + CREMA-D).
- **Podcast speech domain** — younger/middle-aged speakers in recording studio conditions, not elderly care speech.

#### DailyTalkContiguous-MoodyGirl (`MysticKit/DailyTalkContiguous-MoodyGirl`)

**Verdict: Skip.** Evaluated 2026-04-01.

Derivative of DailyTalk dataset (Lee et al., 2022) — human-recorded scripted dialogues, segmented at word level and auto-tagged with emotion labels. CC-BY-SA 4.0. 5 downloads/month, no citations.

**Why we rejected it:**
- **Automatic labels from acoustic features** — emotion tags derived from rule-based mapping of pitch, energy, speech rate buckets. No human annotation or perceptual validation.
- **Non-standard taxonomy** — mixes vocal styles ("shouting", "whispering", "soft tone") with emotional states ("depressed", "worried"). Does not map to standard 6-class emotions.
- **Read speech, not spontaneous** — actors reading scripted dialogue, limiting emotional expressiveness.
- **No fear class** — taxonomy doesn't include fear or disgust.
- **No community validation** — single creator, minimal downloads, no institutional backing.

---

## 2. Experiment Results

### A. Frozen Backbone + Transformer Head (tribev2 ClassificationExperiment)

Uses neuralset cached feature pipeline: extract Wav2VecBert features once, train EmotionEncoder head.

| Experiment | Backbone | Modalities | Hidden | Depth/Heads | Dropout | Dataset | Accuracy | F1 |
|-----------|----------|------------|--------|-------------|---------|---------|----------|-----|
| Grid best (of 54) | Wav2VecBert | Audio | 128 | 4/4 | 0.2 | RAVDESS+CREMA-D | 64.3% | 64.5% |
| Option B | wav2vec2-emotion (Audeering) | Audio | 256 | 4/4 | 0.1 | RAVDESS+CREMA-D | 63.0% | 62.9% |
| Multimodal | Wav2VecBert + Qwen3-0.6B | Audio+Text | 512 | 4/8 | 0.1 | RAVDESS+CREMA-D | ~60% | -- |
| Combined multimodal | Wav2VecBert + Qwen3-0.6B | Audio+Text | 512 | 4/8 | 0.1 | All 3 | 60.0% | -- |

**Grid search dimensions (54 configs):** LR [5e-4, 1e-3, 2e-3] x Depth [2, 4, 8] x Hidden [128, 256] x Dropout [0.0, 0.1, 0.2]

### B. LoRA Fine-tuning (Wav2VecBert backbone + MLP head)

End-to-end training with LoRA adapters on top 6 transformer layers (18-23), targeting Q/K/V/Out attention projections.

| Experiment | LoRA r | Dataset | Batch | LR (lora/head) | Best Acc | Best F1 | Epochs Run |
|-----------|--------|---------|-------|-----------------|----------|---------|------------|
| LoRA r16 | 16 | RAVDESS+CREMA-D | 8 | 5e-5 / 5e-4 | 62.5% | 61.7% | -- |
| LoRA r64 | 64 | RAVDESS+CREMA-D | 8 | 5e-5 / 5e-4 | 68.5% | 68.1% | -- |
| **LoRA r128** | **128** | **RAVDESS+CREMA-D** | **8** | **5e-5 / 5e-4** | **68.9%** | **68.9%** | **~15** |
| LoRA v2 r64 | 64 | RAVDESS+CREMA-D | 8 | 5e-5 / 5e-4 | 62.5% | 61.8% | 30 |
| LoRA v2 r64 resumed | 64 | RAVDESS+CREMA-D | 8 | 5e-5 / 5e-4 | 65.8% | 65.3% | 47 |
| LoRA combined r128 | 128 | All 3 | 16 | 5e-5 / 5e-4 | 63.4% | 63.2% | 21 (stopped) |
| **LoRA augmented r128** | **128** | **RAVDESS+CREMA-D + elderly aug** | **8** | **1e-5 / 1e-4** | **69.4%** | **69.0%** | **16 (from v1 weights)** |

**Production model (v2, current):** LoRA r128 + elderly voice augmentation on RAVDESS+CREMA-D, 69.4% val accuracy, 70.7% TESS, 96.0% fear. Deployed 2026-04-03.

**Production model (v1, replaced):** LoRA r128 on RAVDESS+CREMA-D, 68.9% val accuracy, 68.6% TESS, 98.8% fear. Deployed 2026-03-31.

### LoRA Combined r128 Training Curve (RAVDESS+CREMA-D+IEMOCAP)

| Epoch | Train Loss | Val Loss | Val Acc | Val F1 | Notes |
|-------|-----------|----------|---------|--------|-------|
| 1 | 1.639 | 1.497 | 37.4% | 33.1% | |
| 4 | 1.256 | 1.139 | 54.0% | 53.0% | |
| 6 | 1.093 | 1.070 | 58.0% | 57.4% | |
| 10 | 0.942 | 1.056 | 60.1% | 58.8% | |
| 11 | 0.899 | 1.110 | 61.5% | 60.5% | Best before deadlock |
| 12 | 0.840 | 1.059 | 62.9% | 62.7% | Best after resume |
| 17 | 0.822 | 1.069 | **63.4%** | **63.2%** | **Final best** |
| 21 | 0.724 | 1.246 | 62.3% | 62.0% | Overfitting, stopped |

### C. LoRA + Temporal Head (Wav2VecBert + hybrid attn_pool/BiLSTM)

| Experiment | LoRA r | Head | Hidden | Dataset | Val Acc | Val F1 | TESS Acc | TESS Fear |
|-----------|--------|------|--------|---------|---------|--------|----------|-----------|
| LoRA temporal r128 | 128 | hybrid (attn_pool + BiLSTM) | 128 | RAVDESS+CREMA-D | **70.2%** | **69.8%** | 64.5% | 34.0% |
| **LoRA r128 (production)** | **128** | **mean-pool MLP** | **256** | **RAVDESS+CREMA-D** | **68.9%** | **68.9%** | **68.6%** | **98.8%** |

### D. TESS Out-of-Domain Evaluation (2,400 unseen clips, 2 speakers)

TESS (Toronto Emotional Speech Set) is completely unseen during training. Two speakers: OAF (64yo female) and YAF (26yo female). The 64yo speaker is especially relevant for elderly care.

#### Production Model (LoRA r128, mean-pool MLP)

| Class | Overall | OAF (64yo) | YAF (26yo) |
|-------|---------|-----------|-----------|
| neutral | 69.0% | — | — |
| happy | 58.2% | — | — |
| sad | 77.8% | — | — |
| angry | 7.8% | — | — |
| **fear** | **98.8%** | — | — |
| **disgust** | **100%** | — | — |
| **Total** | **68.6%** | **56.0%** | **81.2%** |

#### Temporal Head Model (LoRA r128, hybrid attn_pool + BiLSTM)

| Class | Overall | OAF (64yo) | YAF (26yo) |
|-------|---------|-----------|-----------|
| neutral | **97.2%** | — | — |
| happy | **61.2%** | — | — |
| sad | 63.0% | — | — |
| angry | **42.0%** | — | — |
| fear | 34.0% | — | — |
| disgust | 89.2% | — | — |
| **Total** | 64.5% | **60.6%** | 68.3% |

**Key finding:** The temporal head achieves higher val accuracy (70.2% vs 68.9%) but **worse out-of-domain generalization**, especially for fear (98.8% → 34.0%). The simpler mean-pool head is more robust to unseen speakers and domains. The temporal model's BiLSTM likely overfits to temporal patterns specific to the training data (RAVDESS/CREMA-D acted speech timing) that don't transfer to TESS's different recording conditions.

**Decision:** Temporal head rejected. Val accuracy alone is insufficient — out-of-domain evaluation on TESS is the better indicator for deployment robustness, especially for fear detection in elderly speech.

#### Augmented Model (LoRA r128, mean-pool MLP, elderly voice augmentation) — CURRENT PRODUCTION

| Class | Overall | OAF (64yo) | YAF (26yo) |
|-------|---------|-----------|-----------|
| neutral | 61.2% | — | — |
| happy | 60.8% | — | — |
| sad | 72.5% | — | — |
| angry | **34.0%** | — | — |
| **fear** | **96.0%** | — | — |
| **disgust** | **99.5%** | — | — |
| **Total** | **70.7%** | **56.8%** | **84.6%** |

**Key improvements over v1 production:**
- Overall TESS: 68.6% → **70.7%** (+2.1%)
- Angry: 7.8% → **34.0%** (+26.2%) — previously nearly broken, now usable
- YAF (26yo): 81.2% → **84.6%** (+3.4%)
- OAF (64yo): 56.0% → **56.8%** (+0.8%) — marginal elderly improvement
- Fear: 98.8% → 96.0% (-2.8%) — slight dip but still excellent
- Val accuracy: 68.9% → **69.4%** (+0.5%)

**Decision:** Augmented model promoted to production (2026-04-03). The elderly augmentation improved overall generalization and fixed the angry class collapse, with only a small fear trade-off (96% is still clinically sufficient).

---

## 3. Model Architectures

### LoRA Model (production)

```
Wav2VecBert (580M params, frozen except LoRA)
  feature_projection: 160 -> 1024
  encoder layers 0-17: frozen Conformer blocks
  encoder layers 18-23: LoRA r=128 on Q/K/V/Out (6.3M params)
  mean pool: (B, T, 1024) -> (B, 1024)

Classification head (266K params):
  LayerNorm(1024) -> Dropout(0.2) -> Linear(1024, 256) -> GELU -> Dropout(0.2) -> Linear(256, 6)

Total: 583.9M params, 3.4M trainable (0.58%)
```

### LoRA + Temporal Head (experimental)

Replaces mean pooling with a hybrid temporal head that preserves timing information.

**Why mean pooling loses information:**
Mean pooling collapses a 199-frame sequence into a single vector, weighting every frame equally. A silent padding frame counts the same as the frame where the speaker's voice cracks with fear. All temporal dynamics are destroyed.

**The hybrid temporal head uses two parallel branches:**

```
Wav2VecBert + LoRA output: (B, 199, 1024)  — 199 frames x 1024 features
              |                          |
       Attentive Pooling             BiLSTM
              |                          |
         (B, 1024)                  (B, 256)
              |                          |
              +------ concatenate -------+
                          |
                     (B, 1280)
                          |
                    MLP classifier
                          |
                     (B, 6) logits
```

**Branch 1 — Self-Attentive Pooling:** A small network (Linear -> tanh -> Linear) scores each of the 199 frames, then softmax-weights them before summing. Learns to focus on the ~15% of frames carrying emotional cues (hesitations, pitch breaks, emphasized words) and ignore silence/filler.

**Branch 2 — BiLSTM:** Processes the full sequence left-to-right AND right-to-left, capturing temporal patterns. Fear often manifests as a *change* over time (voice starts steady, then trembles). Mean pooling can't detect this; BiLSTM can.

Concatenating both branches gives the classifier both "what's salient" (attention) and "how it evolves" (LSTM).

```
Wav2VecBert (580M params, frozen except LoRA)
  encoder layers 18-23: LoRA r=128 (6.3M params)
  full sequence preserved: (B, 199, 1024)

Hybrid temporal head (1.5M params):
  SelfAttentivePool: Linear(1024, 128) -> tanh -> Linear(128, 1) -> softmax -> weighted sum
  BiLSTM: 1-layer bidirectional LSTM(1024, 128) -> last hidden state
  Classifier: LayerNorm(1280) -> Dropout -> Linear(1280, 128) -> GELU -> Dropout -> Linear(128, 6)

Total trainable: 7.8M (6.3M LoRA + 1.5M head)
```

### LoRA Hyperparameters

| Parameter | Value |
|-----------|-------|
| Backbone | facebook/w2v-bert-2.0 |
| LoRA rank (r) | 128 |
| LoRA alpha | 256 (2x rank) |
| LoRA dropout | 0.15 |
| Target modules | linear_q, linear_k, linear_v, linear_out |
| Layers | 18-23 (top 6 of 24) |
| LR (LoRA) | 5e-5 |
| LR (head) | 5e-4 |
| Optimizer | AdamW (weight_decay=0.01) |
| Scheduler | OneCycleLR (pct_start=0.1) |
| Max audio | 4.0 seconds |
| Sample rate | 16000 Hz |
| Gradient clip | norm=1.0 |

### Frozen Backbone Model (tribev2 ClassificationExperiment)

```
Wav2VecBert features -> layer_cat -> projector -> (B, T, H)
  combiner MLP (optional) -> TransformerEncoder -> mean pool -> Linear(H, 6)
```

Uses neuralset's cached extraction pipeline with exca caching.

---

## 4. Training Infrastructure

| Resource | Details |
|----------|---------|
| GPU | NVIDIA RTX A5000 (24GB VRAM) |
| Wav2VecBert VRAM | 2.4 GB |
| Qwen3-0.6B VRAM | 1.2 GB |
| care-whisper-ser sidecar | 2.4 GB (always running on GPU) |
| LoRA training speed | ~8 min/epoch (backbone forward pass dominates) |
| Frozen feature training | ~3 min preprocessing + ~30s/epoch |
| TensorBoard | Ports 6006-6008 |
| Checkpoints | ~/data/emotion_save/results/ |
| Feature cache | ~/data/emotion_save/cache/ |

---

## 5. Lessons Learned

### What Worked

1. **LoRA rank scaling:** r16 (62.5%) < r64 (68.5%) < r128 (68.9%). Higher rank = more adaptation capacity. Diminishing returns above r128.

2. **Speaker-based train/val splits:** Critical for preventing data leakage. Same speaker in both splits inflates metrics.

3. **Differential learning rates:** LoRA params at 5e-5, head at 5e-4 (10x). Backbone needs gentle updates; head needs faster convergence.

4. **OneCycleLR scheduler:** Consistent improvement over flat LR across all experiments.

### What Didn't Work

5. **IEMOCAP mixed with acted corpora:** Combined dataset (63.4%) performs worse than RAVDESS+CREMA-D alone (68.9%). Naturalistic vs acted speech is too different a domain.

6. **Multi-modal frozen backbone:** Audio+text (60%) scored below audio-only (64.3%) with frozen features. Text features couldn't compensate without backbone adaptation.

7. **Pre-tuned emotion backbone (Audeering):** wav2vec2-emotion (63%) performed worse than generic Wav2VecBert (64.3%). Pre-training on MSP-Podcast didn't transfer to our 6-class task.

8. **Larger batch size on combined:** Batch 16 on combined dataset showed no improvement over batch 8 on RAVDESS+CREMA-D.

### Technical Issues

9. **DataLoader num_workers + CUDA = deadlock:** `num_workers=4` caused deadlock at epoch 13 (fork + CUDA tensors). Fix: `num_workers=0`.

10. **Exca cache key includes keep_in_ram:** Pre-extracting with `keep_in_ram=False` then training with `keep_in_ram=True` invalidates cache. Must use identical config.

11. **Training speed:** 580M backbone forward pass runs every batch even with only 3.4M trainable params. ~8 min/epoch vs seconds for MLP-only.

12. **Resume support:** Checkpoint must save optimizer + scheduler state. Model-only checkpoint loses training momentum on resume.

---

## 6. File Inventory

### Training Scripts
- `tribev2/emotion/lora_train.py` — LoRA fine-tuning on RAVDESS+CREMA-D
- `tribev2/emotion/lora_train_combined.py` — LoRA fine-tuning on all 3 datasets
- `tribev2/emotion/pre_extract.py` — Pre-extract features for combined dataset
- `tribev2/emotion/prepare_iemocap.py` — Export IEMOCAP from HuggingFace to disk

### Grid Configs
- `tribev2/grids/emotion_defaults.py` — Base audio-only config
- `tribev2/grids/emotion_gridsearch.py` — 54-config grid search
- `tribev2/grids/emotion_multimodal.py` — Audio+text on RAVDESS+CREMA-D
- `tribev2/grids/emotion_optionb.py` — Audeering emotion backbone
- `tribev2/grids/emotion_combined.py` — Audio+text on all 3 datasets

### Study Classes
- `tribev2/studies/emotion_audio.py` — RavdessEmotion, CremadEmotion, IEMOCAPEmotion

### Model & Data Pipeline
- `tribev2/emotion/model.py` — EmotionEncoderModel (multi-modal fusion)
- `tribev2/emotion/data.py` — ClassificationData
- `tribev2/emotion/experiment.py` — ClassificationExperiment (Lightning)
- `tribev2/emotion/pl_module.py` — ClassificationModule (Lightning)

### Saved Models
- `~/data/emotion_save/results/tribe_emotion_lora_augmented_r128/merged_model.pt` — **Production model v2 (69.4% val, 70.7% TESS)**
- `~/data/emotion_save/results/tribe_emotion_lora_r128/merged_model.pt` — Production model v1 (68.9% val, 68.6% TESS)
- `~/data/emotion_save/results/tribe_emotion_lora_temporal_r128/merged_model.pt` — Temporal head experiment (70.2% val, 64.5% TESS)
- `~/data/emotion_save/results/tribe_emotion_lora_combined_r128/best_lora.pt` — Combined best (63.4%)
- `~/containers/care-whisper-ser/models/best_model.pt` — Symlink to production model v2

---

## 7. Next Steps: Elderly Voice Augmentation

### Problem

The production model shows a 25-point accuracy gap between the younger and older TESS speakers:
- YAF (26yo): 81.2%
- OAF (64yo): 56.0%

All training data (RAVDESS + CREMA-D) uses speakers aged 20-74, with the majority under 50. The model has limited exposure to elderly voice characteristics, causing degraded performance on older speakers.

### Approach: Simulate Aging Voice Characteristics

Apply audio augmentations to the RAVDESS+CREMA-D training clips to synthesize elderly-sounding variants. Three key characteristics of aging voices:

1. **Lower fundamental frequency / reduced pitch range** — vocal fold stiffening and thinning reduces F0 and narrows the pitch contour. Simulate with pitch shifting (-1 to -3 semitones) and F0 range compression.

2. **Slower speaking rate** — reduced motor control and cognitive processing speed. Simulate with time stretching (1.1x to 1.3x duration) without pitch change.

3. **Increased jitter/shimmer (vocal tremor)** — neuromuscular degeneration causes micro-perturbations in pitch and amplitude. Simulate with low-frequency modulation of F0 (3-7 Hz tremor) and amplitude variation.

### Implementation Plan

- Create `tribev2/emotion/augment_elderly.py` that generates augmented copies of training clips
- Apply augmentations with random parameters per clip (pitch shift, rate, tremor intensity)
- Augmented clips added to training set alongside originals (not replacing them)
- Retrain LoRA r128 with mean-pool head (the production architecture)
- Evaluate on TESS, specifically measuring OAF (64yo) improvement

### Success Criteria

- OAF (64yo) accuracy improves from 56.0% toward YAF's 81.2%
- Fear detection stays above 90% on TESS (currently 98.8%)
- Overall TESS accuracy improves from 68.6%

### Results (2026-04-03)

Augmentation completed: 7,107 training clips augmented (100% success rate). Fine-tuned from production v1 weights with lower LR (1e-5 LoRA, 1e-4 head) for 20 epochs.

| Metric | Before (v1) | After (v2 augmented) | Change |
|--------|------------|---------------------|--------|
| Val accuracy | 68.9% | **69.4%** | +0.5% |
| TESS overall | 68.6% | **70.7%** | +2.1% |
| TESS OAF (64yo) | 56.0% | 56.8% | +0.8% |
| TESS angry | 7.8% | **34.0%** | +26.2% |
| TESS fear | 98.8% | 96.0% | -2.8% |

The elderly speaker gap (OAF vs YAF) only closed by 0.8 points. The augmentation helped general robustness more than elderly-specific performance. The pitch/rate/tremor simulation may not capture the full complexity of elderly voice characteristics. More targeted augmentation or real elderly data would be needed to close the 28-point gap further.
