# Speech Emotion Recognition Architecture Diagram

This file contains research-project-ready Mermaid diagrams for the model and the training pipeline we used.

## 1. Hybrid Model Architecture

```mermaid
flowchart LR
    A["RAVDESS Speech Clip
2.5 s at 16 kHz = 40,000 samples"] --> B["Audio Preprocessing
Mono + 16 kHz resampling + fixed-length pad/crop"]

    B --> T0["Transformer Branch Input
Waveform [40,000]"]
    B --> H0["Handcrafted Feature Branch Input
Waveform [40,000]"]

    subgraph TB1 ["Transformer Branch"]
        T0 --> T1["AutoFeatureExtractor
Input tensor [B, 40,000]"]
        T1 --> T2["HuBERT Backbone
superb/hubert-base-superb-er
12 layers, hidden size 768"]
        T2 --> T3["Frame-level Hidden States
[B, T, 768]"]
        T3 --> T4["Masked Mean Pooling
[B, 768]"]
        T4 --> T5["Transformer Embedding
768-D"]
    end

    subgraph HB1 ["Handcrafted Feature Branch"]
        H0 --> H1["Temporal / Energy Features
ZCR + RMS
8-D"]
        H0 --> H2["Spectral Shape Features
Centroid + Bandwidth + Rolloff + Flatness
16-D"]
        H0 --> H3["Cepstral Features
MFCC mean/std + delta mean/std
52-D"]
        H0 --> H4["Time-Frequency Features
Log-mel mean/std + delta + delta-delta
384-D"]
        H0 --> H5["Harmonic / Prosodic Features
Chroma 24-D + Pitch/Voicing 5-D
29-D"]
        H1 --> H6["Concatenate Feature Families
489-D engineered vector"]
        H2 --> H6
        H3 --> H6
        H4 --> H6
        H5 --> H6
        H6 --> H7["Train-set Standardization
489-D"]
        H7 --> H8["LayerNorm [489]
Linear 489 to 128
GELU + Dropout"]
        H8 --> H9["Auxiliary Feature Embedding
128-D"]
    end

    T5 --> F["Feature Fusion
Concat 768-D + 128-D = 896-D"]
    H9 --> F

    F --> D["Shared Dropout
896-D"]
    D --> E1["Emotion Head
Linear 896 to 8"]
    D --> E2["Intensity Head
Linear 896 to 2"]

    E1 --> O1["Emotion Output
8 classes: neutral, calm, happy, sad, angry, fearful, disgust, surprised"]
    E2 --> O2["Intensity Output
2 classes: normal, strong"]
```

## 2. Training and Evaluation Pipeline

```mermaid
flowchart TD
    A["RAVDESS Dataset"] --> B["Filename Parsing\nMM-VC-EE-II-SS-RR-AA"]
    B --> C["Metadata Table\nEmotion, intensity, statement, repetition, actor"]
    C --> D["Actor-wise Split\nTrain / Validation / Test"]

    D --> E["Training Split Only\nCompute feature mean/std for engineered features"]
    D --> F["Dataset Builder\nWaveform + engineered features + labels"]
    E --> F

    F --> G["Hybrid Multi-Task Model"]
    G --> H1["Emotion Loss\nFocal loss + label smoothing"]
    G --> H2["Intensity Loss\nCross-entropy + label smoothing"]

    H1 --> I["Joint Optimization\nAdamW + warmup + cosine scheduler"]
    H2 --> I
    I --> J["Validation Monitoring\nMacro F1 + composite early stopping"]
    J --> K["Best Model Checkpoint"]

    K --> L["Offline Test Evaluation\nAccuracy, Macro F1, confusion matrices, per-actor metrics"]
    K --> M["Streaming Evaluation\nChunked rolling-window inference + latency metrics"]
```

## 3. Caption You Can Reuse

Figure: Proposed hybrid speech emotion recognition architecture. A fixed-length speech waveform is processed in parallel by a pretrained HuBERT transformer branch and a handcrafted acoustic-feature branch. The handcrafted branch extracts temporal, spectral, cepstral, time-frequency, and pitch/voicing descriptors, which are normalized and projected through an auxiliary MLP. The transformer embedding and engineered-feature embedding are fused and passed to two task-specific heads for emotion classification and intensity classification.

## 4. Short Explanation for the Research Document

Why we used this architecture:
- The transformer branch learns rich contextual speech representations directly from raw audio.
- The engineered-feature branch preserves interpretable acoustic cues such as energy, timbre, spectral shape, and pitch dynamics.
- Feature fusion helps the model combine deep learned patterns with classic speech descriptors.
- Multi-task learning allows the same model to predict both emotion category and emotion intensity.

## 5. Notes

Feature families used in the engineered branch:
- ZCR, RMS
- Spectral centroid, bandwidth, rolloff, flatness
- MFCC mean/std
- Delta MFCC mean/std
- Log-mel mean/std
- Delta log-mel mean/std
- Delta-delta log-mel mean/std
- Chroma mean/std
- Pitch statistics
- Voiced ratio

Backbone used:
- `superb/hubert-base-superb-er`
- 12 transformer layers
- hidden size: 768
- auxiliary MLP: 489 to 128
- fused representation: 896

Outputs used:
- Emotion classification: 8 classes
- Intensity classification: 2 classes
