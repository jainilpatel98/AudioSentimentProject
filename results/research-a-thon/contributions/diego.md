# Team Member Contribution: Diego Brown

## 🌊 Project Role: Model Architecture & Training

For this project, Diego focused on the design and optimization of the deep learning architecture, combining pretrained speech transformers and handcrafted acoustic features.

### 🤖 Key Technical Contributions:
*   **Backbone Fine-Tuning:** Successfully fine-tuned the `hubert-base-superb-er` pretrained model, unlocking state-of-the-art speech representations for emotion task transfer.
*   **Multi-Task Architecture:** Engineered the specialized classification heads for simultaneous Emotion and Intensity prediction.
*   **Loss Optimization:** Implemented the "Focal Loss" function with alpha/gamma parameters to combat the class imbalance inherent in emotional speech datasets.
*   **Partial Unfreezing Strategy:** Developed the layer-by-layer unfreezing schedule, ensuring the transformer backbone was tuned without catastrophic forgetting.
*   **Training Life-Cycle:** Configured the AdamW optimizer with Cosine Annealing and Warmup, reaching convergence for the offline test split within 50 epochs.

### 📈 Strategic Impact:
Through the fusion of modern Transformers and classical feature engineering, the project established a high-performance, multi-task classifier that outperforms simple architectures.
