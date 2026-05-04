# 🎙️ Emotion Recognition from Speech

A machine learning project that classifies human emotions from audio using the RAVDESS dataset and an SVM classifier.

**Group Members:** JoshaLynn · Janil · Diego · Tina

---

## 📋 Overview

This project trains a Support Vector Machine (SVM) to identify 8 emotions from speech recordings. Beyond the core classifier, we run three additional experiments to better understand model behavior:

1. **Intensity Breakdown** — Does the model perform better on strongly-expressed emotions vs. normal ones?
2. **Emotion Grouping** — Does collapsing acoustically similar emotions into broader classes improve accuracy?
3. **Gender Analysis** — Is there a performance gap between male and female speakers?
4. **Unsupervised Exploration** — Do emotions form natural clusters in audio feature space (LDA + KMeans)?

---

## 📁 Dataset

**RAVDESS** — Ryerson Audio-Visual Database of Emotional Speech and Song

- 24 professional actors (12 male, 12 female)
- 8 emotions: `neutral`, `calm`, `happy`, `sad`, `angry`, `fearful`, `disgust`, `surprised`
- 2 intensity levels: `normal` and `strong`
- Clean studio audio recordings
- Labels encoded directly in each filename (e.g. `03-01-05-01-01-01-12.wav`)

> **Note:** The `neutral` emotion only appears at normal intensity in RAVDESS, giving it fewer clips than other classes.

---

## 🔧 Setup & Installation

### Prerequisites

- Python 3.8+
- Google Colab (recommended) or local Jupyter environment

### Install Dependencies

```bash
pip install gdown librosa soundfile scikit-learn pandas numpy matplotlib seaborn
```

### Download the Dataset

The notebook auto-downloads the dataset from Google Drive on first run using `gdown`. Re-running the cell will not re-download.

```python
DATA_ZIP = "actors_speech.zip"
DATA_DIR = "actors_speech"
```

---

## 🗂️ Project Structure

```
emotion_recognition.ipynb   # Main notebook — all code and analysis
ground_truth_labels.csv     # Auto-generated metadata for all audio clips
actors_speech/              # Downloaded RAVDESS dataset
  Actor_01/
  Actor_02/
  ...
  Actor_24/
```

---

## 🔄 Pipeline

```
1. Load & Preprocess Audio
         ↓
2. Exploratory Data Analysis
         ↓
3. Train / Test Split (by Actor)
         ↓
4. Augment Training Data
         ↓
5. Feature Extraction
         ↓
6. Train SVM Classifier
         ↓
7. Evaluate + Experiments
```

---

## 🧹 Audio Preprocessing

Every clip is standardized before feature extraction:

| Step | Detail |
|------|--------|
| Resample | 16 kHz — standard rate for speech processing |
| Normalize | Scales amplitude so volume is consistent across speakers |
| Trim Silence | Strips quiet sections at start/end (`top_db = 20`) |
| Fixed Length | All clips padded or truncated to 2 seconds |

> **Volume Normalization Check:** We verified using RMS energy that after normalization, `fearful` and `sad` clips (which differ ~3× in raw volume) become nearly identical in energy — confirming the model learns emotional cues, not just loudness.

---

## ✂️ Train / Test Split

Split is done **by actor**, not randomly, to prevent data leakage (same voice appearing in both sets).

- **Test set:** 4 male + 4 female actors (gender-balanced)
- **Train set:** Remaining 16 actors
- **Approximate split:** 70% train / 30% test

---

## 🔀 Data Augmentation

Training clips only are augmented to give the model more variety. `MULTIPLY = 2` means each training clip gets 2 extra augmented versions (3× total training data).

Augmentation techniques applied randomly:
- Gaussian noise injection
- Time shift (±0.1s)
- Volume scaling (±20%)

The test set is **never augmented**.

---

## 📊 Feature Extraction

Raw audio is converted to a fixed-length feature vector. Each time-series feature is summarized as `[mean, std, min, max]`.

| Feature | What It Captures |
|---------|-----------------|
| MFCCs | Timbral texture of speech |
| Chroma | Pitch class distribution |
| Spectral Contrast | Energy differences across frequency bands |
| Zero Crossing Rate | Noisiness / voicing |
| RMS Energy | Perceived loudness over time |

---

## 🤖 Model

**Support Vector Machine (SVM)** with an RBF kernel, wrapped in a scikit-learn Pipeline:

```
StandardScaler → PCA (n=50) → SVC (RBF kernel, class_weight='balanced')
```

**Key hyperparameters:**
- `C = 1`
- `gamma = 'scale'`
- `class_weight = 'balanced'` (handles class imbalance from neutral having fewer clips)
- `probability = True` (enables ROC curve generation)

> ⚠️ **Note:** Section 18 of the notebook (the initial SVM definition using `PCA(n_components=100)`) was **not used in the final pipeline** due to severe overfitting and extreme memorization of the training data. The final model uses `n_components=50`, which was found to generalize better to unseen actors.

---

## 📈 Results

### 8-Class Emotion Classification

| Metric | Value |
|--------|-------|
| **Test Accuracy** | ~51% |

> ⚠️ **Overfitting:** The large train-test gap indicates the model memorized training data rather than learning generalizable features. The actor-based split makes this a hard generalization problem.

**ROC AUC highlights:**
- Strong performance: `calm`, `surprised`, `angry`, `disgust` (AUC > 0.90)
- Weaker performance: `sad` (AUC = 0.76), `fearful` / `happy` (~0.86) — these emotions share overlapping acoustic patterns

---

### Experiment 1 — Intensity Breakdown

Strong-intensity clips are slightly easier for the model to classify, consistent with the intuition that more expressive recordings produce clearer audio features.

---

### Experiment 2 — Emotion Grouping (4 Classes)

Acoustically similar emotions were merged into 4 broader groups:

| Group | Emotions |
|-------|---------|
| `anger_disgust` | angry, disgust |
| `sad_calm_neutral` | sad, calm, neutral |
| `fear_surprise` | fearful, surprised |
| `happy` | happy |

The grouped model shows a measurable accuracy improvement over the 8-class model, confirming that acoustically similar emotions are the main source of confusion.

---

### Experiment 3 — Gender Analysis

| Group | Accuracy |
|-------|---------|
| Female voices | Higher (~11% gap) |
| Male voices | Lower |

Female voices benefit from wider pitch range and higher acoustic brightness, giving the model clearer patterns. This represents a measurable **gender bias** in the model — in production, gender-normalized features or separate per-gender models would be needed.

---

### Experiment 4 — Unsupervised Exploration (LDA + KMeans)

- **LDA** projects features down to 2D, maximizing class separation
- **KMeans** clusters that space without using labels

Clusters in LDA space look fairly clean, but don't perfectly align with emotion labels — KMeans groups by acoustic similarity, not human-labelled emotion. Acoustically extreme emotions (`angry`, `calm`) form cleaner clusters. Similar-sounding pairs (`neutral` / `calm`) bleed into each other, matching what the SVM confusion matrix shows.

---

### Experiment 5 — 5-Fold Cross-Validation

GroupKFold cross-validation (by actor) was used to get a more stable accuracy estimate. This confirms performance and prevents any single actor split from skewing results.

---

## 🧰 Dependencies

| Package | Purpose |
|---------|---------|
| `librosa` | Audio loading, feature extraction |
| `soundfile` | Audio I/O |
| `scikit-learn` | SVM, PCA, cross-validation, metrics |
| `numpy` | Numerical operations |
| `pandas` | Data management |
| `matplotlib` | Visualization |
| `seaborn` | Statistical plots |
| `gdown` | Dataset download from Google Drive |

---

## 🚀 Running the Notebook

1. Open `emotion_recognition.ipynb` in Google Colab or Jupyter
2. Run all cells in order — the dataset downloads automatically on first run
3. Results, plots, and metrics will render inline

---

## 📝 Notes & Known Issues

- The notebook uses a fixed random seed (`SEED = 42`) for reproducibility across all runs
- The `load_audio` function includes a debug `print(y)` — this produces verbose output and can be removed
- Augmented clips were previously written to a fake `Actor_99` folder; a cleanup cell removes this if it exists
- The `neutral` emotion has roughly half the clips of other classes since RAVDESS only records it at normal intensity
