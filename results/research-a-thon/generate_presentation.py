import os
from fpdf import FPDF

class UMKCPresentation(FPDF):
    def header(self):
        # Top bar
        self.set_fill_color(0, 102, 204) # UMKC Blue
        self.rect(0, 0, 297, 20, 'F')
        self.set_fill_color(234, 175, 15) # UMKC Gold
        self.rect(0, 19, 297, 2, 'F')
        
    def footer(self):
        # Bottom bar
        self.set_y(-15)
        self.set_font('helvetica', 'I', 8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 10, f'Page {self.page_no()} | Emotion Recognition from Speech | UMKC Research-a-thon', 0, 0, 'C')

    def slide_title(self, label):
        self.set_font('helvetica', 'B', 24)
        self.set_text_color(0, 102, 204) # UMKC Blue
        self.set_y(30)
        self.cell(0, 10, label, 0, 1, 'L')
        # Underline
        self.set_draw_color(234, 175, 15) # UMKC Gold
        self.line(10, 42, 100, 42)
        self.ln(10)

    def slide_content(self, text_list):
        self.set_font('helvetica', '', 14)
        self.set_text_color(40, 40, 40)
        for item in text_list:
            self.multi_cell(0, 8, f'- {item}', 0, 'L')
            self.ln(2)

def generate():
    pdf = UMKCPresentation(orientation='L', unit='mm', format='A4')
    pdf.set_auto_page_break(auto=True, margin=15)

    # --- SLIDE 1: TITLE ---
    pdf.add_page()
    # Large background image if available
    if os.path.exists('results/research-a-thon/assets/hero.png'):
        pdf.image('results/research-a-thon/assets/hero.png', x=10, y=55, w=277)
    
    pdf.set_y(60)
    pdf.set_font('helvetica', 'B', 48)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(0, 20, 'Emotion Recognition', 0, 1, 'C')
    pdf.cell(0, 20, 'from Speech', 0, 1, 'C')
    
    # Subtitle box
    pdf.set_y(150)
    pdf.set_fill_color(234, 175, 15) # Gold
    pdf.rect(60, 145, 177, 25, 'F')
    pdf.set_text_color(0, 102, 204) # Blue
    pdf.set_font('helvetica', 'B', 16)
    pdf.cell(0, 15, 'A Machine Learning Approach to Acoustic Affective Computing', 0, 1, 'C')
    pdf.set_font('helvetica', '', 12)
    pdf.cell(0, 5, 'COMP-SCI 5530: Principles of Data Science | University of Missouri - Kansas City', 0, 1, 'C')

    # --- SLIDE 2: TEAM ---
    pdf.add_page()
    pdf.slide_title('The Research Team')
    team = [
        "Tina Nguyen: Technical Project Director & Systems Integration Lead",
        "Diego Brown: Senior Acoustic Signal Engineer & Data Augmentation Architect",
        "Jainil Anilkumar Patel: Lead Machine Learning Architect & Training Specialist",
        "JoshaLynn Worth: Principal Data Analyst & AI Ethics Strategy Lead"
    ]
    pdf.slide_content(team)
    if os.path.exists('results/research-a-thon/assets/neural_net.png'):
        pdf.image('results/research-a-thon/assets/neural_net.png', x=160, y=90, w=120)

    # --- SLIDE 3: PROBLEM ---
    pdf.add_page()
    pdf.slide_title('The Problem: Decoding Affect')
    points = [
        "Speech is the primary medium for human self-expression, carrying rich paralinguistic cues.",
        "Traditional HCI (Human-Computer Interaction) lacks emotional intelligence.",
        "Goal: Build a robust pipeline to extract emotional states from raw voice waves.",
        "Challenges: Subjective nature of emotion and acoustic frequency overlap."
    ]
    pdf.slide_content(points)

    # --- SLIDE 4: DATASET ---
    pdf.add_page()
    pdf.slide_title('The Foundation: RAVDESS Corpus')
    points = [
        "Ryerson Audio-Visual Database of Emotional Speech and Song.",
        "24 professional actors (12 Male, 12 Female) ensuring demographic balance.",
        "Lexically-matched neutral statements to isolate emotion from semantics.",
        "8 Fundamental Emotions: Neutral, Calm, Happy, Sad, Angry, Fearful, Disgust, Surprised."
    ]
    pdf.slide_content(points)

    # --- SLIDE 5: PREPROCESSING ---
    pdf.add_page()
    pdf.slide_title('Phase I: Signal Processing & Augmentation')
    points = [
        "Feature Extraction: MFCCs, Chroma Features, and Mel Spectrograms.",
        "Standardization: 16kHz Mono audio with silence truncation.",
        "Data Augmentation: White noise injection, Pitch shifting, and Time stretching.",
        "Advanced Technique: Speaker-Mix Augmentation to improve generalization."
    ]
    pdf.slide_content(points)
    if os.path.exists('results/research-a-thon/assets/spectrogram.png'):
        pdf.image('results/research-a-thon/assets/spectrogram.png', x=180, y=45, w=100)

    # --- SLIDE 6: MODEL ---
    pdf.add_page()
    pdf.slide_title('Phase II: Hybrid Multi-Task Architecture')
    points = [
        "Backbone: HUBERT (Hidden-Unit BERT) for deep speech representations.",
        "Engineering Branch: Handcrafted acoustic descriptors for prosody (Pitch, Energy, ZCR).",
        "Fusion: Combining contextual transformer embeddings with interpretable features.",
        "Multi-Task Heads: Simultaneous prediction of Emotion (Class) and Intensity (Degree)."
    ]
    pdf.slide_content(points)

    # --- SLIDE 7: RESULTS ---
    pdf.add_page()
    pdf.slide_title('Evaluation: Quantitative Success')
    points = [
        "High Arousal Robustness: 80-85% specific accuracy for Happy/Angry classes.",
        "Intensity Prediction: Reached ~78% accuracy, demonstrating strong degree classification.",
        "Macro F1 Strategy: Ensuring equal priority across all emotional frequencies.",
        "Per-Actor Variance: Identifying speaker-independent vs. speaker-dependent traits."
    ]
    pdf.slide_content(points)
    
    # --- SLIDE 8: DEPLOYMENT ---
    pdf.add_page()
    pdf.slide_title('Live Inference: The Streamlit Dashboard')
    points = [
        "Real-time Telemetry: Processing microphone streams in rolling windows.",
        "Edge-Ready: Low-latency inference (avg. ~370ms) for interactive use.",
        "Visualization: Dynamic probability trackers for 8 emotion classes.",
        "Challenge: Managing micro-batch noise and instability in live environments."
    ]
    pdf.slide_content(points)

    # --- SLIDE 9: ETHICS ---
    pdf.add_page()
    pdf.slide_title('Ethical AI & Future Responsibility')
    points = [
        "Algorithmic Bias: Mitigating demographic skew through diverse dataset integration.",
        "Privacy & Surveillance: Implementing encrypted, localized on-device processing.",
        "Informed Consent: Ensuring users are aware of emotional tracking protocols.",
        "Transparency: Using engineered features to explain the model's decision boundaries."
    ]
    pdf.slide_content(points)

    # --- SLIDE 10: CONCLUSION ---
    pdf.add_page()
    pdf.slide_title('Conclusion & Future Horizons')
    points = [
        "Established a viable end-to-end Affective Computing pipeline.",
        "Future: Multimodal integration (Computer Vision + Audio) for higher reliability.",
        "Scale: Moving to the edge for mental health and automotive safety applications.",
        "Legacy: Setting a high standard for data-driven emotional intelligence."
    ]
    pdf.slide_content(points)
    
    pdf.set_y(160)
    pdf.set_font('helvetica', 'B', 20)
    pdf.set_text_color(0, 102, 204)
    pdf.cell(0, 10, 'Thank You!', 0, 1, 'C')

    output_path = 'results/research-a-thon/presentation.pdf'
    pdf.output(output_path)
    print(f"Presentation successfully generated at {output_path}")

if __name__ == '__main__':
    generate()
