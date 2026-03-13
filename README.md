# Speech Emotion Recognition (RAVDESS)

This project trains a hybrid Speech Emotion Recognition model and serves live predictions in Streamlit.

## What the model predicts

- Emotion class: full RAVDESS set by default
  - `neutral`, `calm`, `happy`, `sad`, `angry`, `fearful`, `disgust`, `surprised`
- Emotion intensity (degree): `normal` vs `strong`

This is multi-task prediction (emotion + intensity), not just `happy/sad/other`.

## RAVDESS naming convention handled in code

Parser uses full filename schema:

- `MM-VC-EE-II-SS-RR-AA`
- `MM`: modality
- `VC`: vocal channel
- `EE`: emotion
- `II`: intensity
- `SS`: statement
- `RR`: repetition
- `AA`: actor

All fields are decoded and tracked in `SampleRecord` metadata.

## Why this version is stronger

- Transformer backbone: `superb/hubert-base-superb-er`
- Hybrid architecture:
  - Transformer embedding branch
  - Engineered acoustic feature branch
  - Fusion before multi-task heads
- Feature engineering (handcrafted acoustic descriptors):
  - ZCR, RMS, spectral centroid/bandwidth/rolloff/flatness
  - MFCC + delta MFCC statistics
  - Log-mel spectrogram + delta + delta-delta statistics
  - Chroma statistics
  - Pitch/voicing statistics
- Augmentation pipeline:
  - Additive noise
  - Time shift
  - Pitch shift
  - Time stretch
  - Random gain
  - Same-label cross-speaker mixing ("merge two humans")
- Training/evaluation:
  - Actor-wise train/val/test split
  - Class-weighted losses (and optional focal emotion loss)
  - Label smoothing
  - Warmup + cosine LR schedule
  - Feature encoder freeze warm-start
  - Optional partial backbone unfreezing (last N transformer layers)
  - Early stopping on composite validation score
  - Emotion, intensity, and joint confusion/report outputs

## Why we did each major choice

- Full emotion labels + intensity:
  - The dataset includes both class and degree, so we model both tasks directly instead of collapsing labels.
- Actor-wise split:
  - Prevents speaker leakage and gives a more realistic estimate of generalization to new speakers.
- Hybrid model (transformer + engineered features):
  - Transformer captures deep contextual speech patterns.
  - Engineered features preserve classic prosody/spectral cues (pitch, energy, timbre) that are important for emotion.
  - Fusion improves robustness when one feature family is weak.
- Speaker-mix augmentation:
  - Simulates mixed speaking conditions and improves robustness to speaker variation.
  - We mix only same-label samples so targets stay consistent.
- Multi-level evaluation:
  - We report emotion, intensity, and joint metrics, plus confusion matrices and streaming latency.
  - This gives both model-quality and real-time app-quality evidence for class presentation.

## What \"best\" means here

- No one can honestly guarantee \"best in the world\" from a single class-project run.
- This repo now uses strong, defensible practices for this dataset and deployment goal:
  - modern pretrained backbone
  - hybrid feature fusion
  - robust augmentation
  - leakage-safe splitting
  - offline + streaming evaluation
- To claim \"best in class\" for your report, compare at least:
  - transformer-only (`--no-use-handcrafted-features`)
  - hybrid (default)
  - and report both offline test metrics + `streaming_metrics.json`

## Project files

- `train_model.py`: hybrid multi-task training + evaluation + export
- `evaluate_streaming.py`: streaming/chunked evaluation with latency metrics
- `ser_pipeline.py`: parsing, labels, augmentation, features, metadata
- `ser_multitask.py`: shared model definition (training + inference)
- `streamlit_app.py`: live + clip inference app
- `scripts/download_data.sh`: download + extract dataset

## Setup

1. Create and activate env:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Download dataset:

```bash
./scripts/download_data.sh
```

## Train (same entrypoint)

```bash
python train_model.py --data-dir actors_speech --output-dir artifacts
```

Quality-focused run:

```bash
python train_model.py \
  --data-dir actors_speech \
  --output-dir artifacts \
  --epochs 30 \
  --batch-size 8 \
  --augment-copies 4 \
  --speaker-mix-prob 0.35
```

Emotion-focused run (focal loss + partial unfreezing):

```bash
python train_model.py \
  --data-dir actors_speech \
  --output-dir artifacts \
  --emotion-loss focal \
  --focal-gamma 2.0 \
  --unfreeze-last-n-layers 4
```

Optional 7-emotion scheme (drops `calm`):

```bash
python train_model.py --emotion-scheme ekman7
```

Optional disable engineered feature branch:

```bash
python train_model.py --no-use-handcrafted-features
```

Optional offline mode (use local cache only, no model download attempt):

```bash
python train_model.py --offline
```

CPU-friendly mode (fast head-only tuning when full backbone fine-tuning is too slow on CPU):

```bash
python train_model.py \
  --freeze-backbone \
  --augment-copies 0 \
  --epochs 6
```

For best final quality, run full fine-tuning (without `--freeze-backbone`) on a GPU-capable machine.

## Evaluate streaming behavior

Runs chunked, rolling-window evaluation on the held-out test actors and reports latency + online metrics:

```bash
python evaluate_streaming.py --artifacts-dir artifacts --data-dir actors_speech
```

Output:

- `artifacts/streaming_metrics.json`

## Saved artifacts

`artifacts/` includes:

- `hf_model/` (fine-tuned backbone + feature extractor)
- `model_state.pt` (multi-task heads + config + weights)
- `metadata.json`
- `metrics.json`
- `emotion_classification_report.json`
- `intensity_classification_report.json`
- `joint_classification_report.json`
- `emotion_confusion_matrix.csv/.npy`
- `intensity_confusion_matrix.csv/.npy`
- `joint_confusion_matrix.csv/.npy`
- `history.json`
- `streaming_metrics.json` (after streaming eval)

## Run Streamlit app

```bash
streamlit run streamlit_app.py
```

In app:

- `Live` tab: rolling-window real-time predictions while speaking
- `Clip` tab: one-shot prediction from recorded/uploaded audio

Live mode auto-trims long speech to the latest chunk and predicts:

- emotion class
- intensity degree (`normal`/`strong`)
