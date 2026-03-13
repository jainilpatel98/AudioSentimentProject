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

Resume behavior:

- If the output directory already contains training artifacts, `train_model.py` resumes automatically from the saved checkpoint.
- To force a fresh run in the same output directory, disable resume explicitly:

```bash
python train_model.py --no-resume-if-exists
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

--------------------------------------------------------------------------
The exact successful full run we completed on macboook was:

source .venv/bin/activate

MPLCONFIGDIR=/tmp python train_model.py \
  --data-dir actors_speech \
  --output-dir artifacts_fullrun_final \
  --epochs 50 \
  --batch-size 8 \
  --patience 2 \
  --duration-seconds 2.5 \
  --augment-copies 0 \
  --feature-stats-max-records 600 \
  --emotion-loss focal \
  --focal-gamma 2.0 \
  --unfreeze-last-n-layers 2 \
  --freeze-feature-encoder-epochs 1 \
  --offline \
  --num-workers 0 \
  --device mps

--------------------------------------
{                                                                                                                                                                   
  "emotion_accuracy": 0.35,
  "emotion_balanced_accuracy": 0.359375,
  "intensity_accuracy": 0.7791666666666667,
  "intensity_balanced_accuracy": 0.76953125,
  "joint_accuracy": 0.23333333333333334,
  "joint_macro_f1": 0.19685323354693626,
  "emotion_macro_f1": 0.3349771293673913,
  "emotion_weighted_f1": 0.33064227132521745,
  "intensity_macro_f1": 0.7703598057446155,
  "intensity_weighted_f1": 0.7733578860585053,
  "best_val_composite_score": 0.7100685733598842,
  "num_train_samples": 960,
  "num_val_samples": 240,
  "num_test_samples": 240,
  "model_name": "superb/hubert-base-superb-er",
  "task_type": "emotion_intensity_multitask",
  "emotion_scheme": "all8",
  "emotion_loss": "focal",
  "focal_gamma": 2.0,
  "unfreeze_last_n_layers": 2,
  "use_handcrafted_features": true,
  "freeze_backbone": false,
  "aux_feature_dim": 489,
  "device": "mps",
  "training_args": {
    "data_dir": "actors_speech",
    "output_dir": "artifacts_fullrun_final",
    "model_name": "superb/hubert-base-superb-er",
    "emotion_scheme": "all8",
    "epochs": 4,
    "batch_size": 8,
    "learning_rate": 1.5e-05,
    "weight_decay": 0.01,
    "warmup_ratio": 0.1,
    "augment_copies": 0,
    "speaker_mix_prob": 0.3,
    "speaker_mix_alpha_min": 0.35,
    "speaker_mix_alpha_max": 0.65,
    "val_size": 0.15,
    "test_size": 0.15,
    "seed": 42,
    "duration_seconds": 2.5,
    "sample_rate": 16000,
    "max_train_records": 0,
    "max_val_records": 0,
    "max_test_records": 0,
    "patience": 2,
    "min_delta": 0.001,
    "freeze_feature_encoder_epochs": 1,
    "label_smoothing": 0.05,
    "emotion_loss": "focal",
    "focal_gamma": 2.0,
    "intensity_label_smoothing": 0.02,
    "intensity_loss_weight": 1.0,
    "intensity_metric_weight": 0.5,
    "head_dropout": 0.2,
    "unfreeze_last_n_layers": 2,
    "freeze_backbone": false,
    "use_handcrafted_features": true,
    "aux_hidden_dim": 128,
    "feature_stats_max_records": 600,
    "grad_clip": 1.0,
    "num_workers": 0,
    "device": "mps",
    "offline": true
  },
  "audio_config": {
    "sample_rate": 16000,
    "duration_seconds": 2.5
  },
  "augment_config": {
    "noise_prob": 0.8,
    "noise_scale": 0.006,
    "shift_prob": 0.7,
    "shift_max_fraction": 0.2,
    "pitch_prob": 0.45,
    "pitch_max_steps": 2.5,
    "stretch_prob": 0.35,
    "stretch_min_rate": 0.85,
    "stretch_max_rate": 1.15,
    "gain_prob": 0.35,
    "gain_db_min": -6.0,
    "gain_db_max": 6.0,
    "speaker_mix_prob": 0.3,
    "speaker_mix_alpha_min": 0.35,
    "speaker_mix_alpha_max": 0.65
  },
  "feature_config": {
    "n_mfcc": 13,
    "n_mels": 64,
    "frame_length": 1024,
    "hop_length": 256,
    "n_fft": 1024,
    "mel_fmin": 20.0,
    "mel_fmax": 8000.0,
    "pitch_fmin": 50.0,
    "pitch_fmax": 500.0
  },
  "per_actor": {
    "1": {
      "num_samples": 60,
      "emotion_accuracy": 0.3333333333333333,
      "intensity_accuracy": 0.7333333333333333,
      "joint_accuracy": 0.21666666666666667
    },
    "9": {
      "num_samples": 60,
      "emotion_accuracy": 0.3333333333333333,
      "intensity_accuracy": 0.8,
      "joint_accuracy": 0.26666666666666666
    },
    "17": {
      "num_samples": 60,
      "emotion_accuracy": 0.3333333333333333,
      "intensity_accuracy": 0.8166666666666667,
      "joint_accuracy": 0.23333333333333334
    },
    "19": {
      "num_samples": 60,
      "emotion_accuracy": 0.4,
      "intensity_accuracy": 0.7666666666666667,
      "joint_accuracy": 0.21666666666666667
    }
  }
}


--------------------------------------
MPLCONFIGDIR=/tmp python evaluate_streaming.py \
  --artifacts-dir artifacts_fullrun_final \
  --data-dir actors_speech \
  --device mps
--------------------------------------
cp -R artifacts_fullrun_final artifacts
--------------------------------------
streamlit run streamlit_app.py
--------------------------------------
MPLCONFIGDIR=/tmp python train_model.py \
  --data-dir actors_speech \
  --output-dir artifacts_fullrun_final \
  --epochs 30 \
  --batch-size 8 \
  --patience 5 \
  --duration-seconds 10 \
  --augment-copies 2 \
  --feature-stats-max-records 600 \
  --emotion-loss focal \
  --focal-gamma 2.0 \
  --unfreeze-last-n-layers 4 \
  --freeze-feature-encoder-epochs 1 \
  --offline \
  --num-workers 0 \
  --device mps
