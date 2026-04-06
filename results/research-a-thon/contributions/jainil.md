# Team Member Contribution: Jainil Anilkumar Patel

## 🧠 Project Role: Preprocessing & Augmentation

For this project, Jainil focused on the mathematical and statistical transformation of raw human voice signals into highly-dimensional numerical features, a critical pre-modeling stage for Emotion Recognition.

### 🔬 Key Technical Contributions:
*   **Advanced Feature Engineering:** Developed the automated extraction of MFCCs (Mel-Frequency Cepstral Coefficients), Chroma STFTs, and Log-Mel Spectrograms using the `librosa` library.
*   **Acoustic Signal Processing:** Standardized sampling rates (16kHz), mono-channel conversion, and silence cropping across the RAVDESS dataset.
*   **Speaker-Mix Augmentation:** Designed and implemented the "same-label cross-speaker mixing" logic, synthesizing new training samples by blending voices.
*   **Robustness Engineering:** Constructed the noise-injection and pitch-shifting pipeline, essential for improving the model's performance in real-world environments.
*   **Feature Statistics Pipeline:** Engineered the global normalization system (standard scaler) that prevents high-amplitude features from overpowering subtle acoustic traits.

### 📈 Strategic Impact:
By converting chaotic sound waves into structured mathematical arrays, the project established high-fidelity features required for the neural network to identify the subtle "texture" of human emotion.
