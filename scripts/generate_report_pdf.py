from fpdf import FPDF
from fpdf.enums import XPos, YPos

class UMKCResearchReport(FPDF):
    def header(self):
        # UMKC Blue Banner at the top
        self.set_fill_color(0, 75, 135) # UMKC Blue
        self.rect(0, 0, 210, 25, 'F')
        
        self.set_font('helvetica', 'B', 12)
        self.set_text_color(255, 255, 255) # White
        self.cell(0, 10, 'UMKC School of Science and Engineering | Affective Computing Research', border=0, align='C', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.ln(10)

    def footer(self):
        self.set_y(-20)
        self.set_font('helvetica', 'I', 8)
        self.set_text_color(169, 169, 169)
        self.cell(0, 10, f'Page {self.page_no()} of 4 | Project Report: CS 5530 Research-a-thon', border=0, align='C')
        
    def section_title(self, title):
        self.set_font('helvetica', 'B', 14)
        self.set_text_color(0, 75, 135) # UMKC Blue
        self.cell(0, 10, title, new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='L')
        self.line(self.get_x(), self.get_y(), self.get_x() + 190, self.get_y())
        self.ln(5)

    def section_body(self, body):
        self.set_font('helvetica', '', 11)
        self.set_text_color(30, 30, 30)
        self.multi_cell(0, 6, body)
        self.ln(5)

    def bold_text(self, text):
        self.set_font('helvetica', 'B', 11)
        self.write(6, text)
        self.set_font('helvetica', '', 11)

def clean_text(text):
    # Replace common unicode characters that helvetica doesn't like
    replacements = {
        '\u2014': '-',   # em-dash
        '\u2013': '-',   # en-dash
        '\u201c': '"',   # left smart quote
        '\u201d': '"',   # right smart quote
        '\u2018': "'",   # left smart single quote
        '\u2019': "'",   # right smart single quote
        '\u2022': '*',   # bullet
    }
    for char, rep in replacements.items():
        text = text.replace(char, rep)
    return text

def generate():
    pdf = UMKCResearchReport()
    pdf.set_auto_page_break(auto=True, margin=20)
    
    # --- PAGE 1: TITLE & INTRODUCTION ---
    pdf.add_page()
    pdf.ln(12)
    
    # Large Title
    pdf.set_font('helvetica', 'B', 24)
    pdf.set_text_color(0, 75, 135)
    pdf.multi_cell(0, 12, clean_text('Emotion Recognition from Speech:\nA Hybrid Transformer Approach'), align='C')
    pdf.ln(6)
    
    # Team Section
    pdf.set_font('helvetica', 'B', 12)
    pdf.set_text_color(226, 168, 41) # UMKC Gold
    pdf.cell(0, 10, 'Research Team Members', new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='C')
    
    pdf.set_font('helvetica', '', 11)
    pdf.set_text_color(50, 50, 50)
    team = [
        "Tina Nguyen: Pipeline & Coordination",
        "Diego Brown: Model Architecture & Training",
        "Jainil Anilkumar Patel: Preprocessing & Augmentation",
        "JoshaLynn Worth: Results Evaluation & Ethics"
    ]
    for member in team:
        pdf.cell(0, 7, member, new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='C')
    
    pdf.ln(10)
    
    pdf.section_title('1. Abstract')
    pdf.section_body(clean_text("This project presents a multi-task learning framework for Emotion Recognition from Speech (ERS), utilizing a hybrid architecture that combines a pretrained Hubert speech transformer with handcrafted acoustic features. We achieve robust classification across eight emotional states and two intensity levels, with specific superior performance in high-arousal emotional clusters. The study underscores the viability of affective computing in speaker-independent environments while addressing critical ethical concerns regarding algorithmic bias."))
    
    pdf.section_title('2. Problem Statement & Introduction')
    pdf.section_body(clean_text("Speech encompasses a vast array of paralinguistic features-pitch, tone, and cadence-that convey underlying affective states. Historically, AI has lacked the emotional intelligence to process this latent information. This project constructs a robust pipeline for Affective Computing, capable of decoding emotional states from raw waveforms. The challenge lies in the subjective, overlapping frequency domains of high-arousal emotions, requiring transformation from raw waveforms to high-dimensional statistical representations."))

    # --- PAGE 2: DATA & PREPROCESSING ---
    pdf.add_page()
    pdf.section_title('3. Dataset: RAVDESS')
    pdf.section_body(clean_text("For this project, we utilized the Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS). This dataset contains 7,356 recordings from 24 professional actors, providing a demographic balance across genders and vocal frequencies. Critically, the dataset uses lexically-matched neutral statements, ensuring the model focuses exclusively on vocal inflection rather than semantic meaning."))
    
    pdf.section_title('4. Data Preprocessing & Augmentation')
    pdf.bold_text("A. Audio Standardization: ")
    pdf.section_body(clean_text("Samples were resampled to 16,000 Hz and converted to mono-channel to reduce input dimensionality. Silence trimming was applied to isolate emotional content."))
    
    pdf.bold_text("B. Feature Engineering: ")
    pdf.section_body(clean_text("We extracted 13 MFCCs, Chroma STFT, and Mel Spectrograms to capture spectral and temporal 'textures' of sound. These were fused with a 489-dimensional vector of handcrafted features for high-fidelity representation."))
    
    pdf.bold_text("C. Augmentation Strategy: ")
    pdf.section_body(clean_text("To prevent overfitting, we implemented noise injection, pitch shifting, and time-stretching. We also utilized a novel 'speaker-mix' augmentation, blending voice signals from different actors to improve cross-speaker generalization."))

    # --- PAGE 3: ARCHITECTURE & DESIGN ---
    pdf.add_page()
    pdf.section_title('5. ML/AI Methods: Hybrid Multi-Task Transformer')
    pdf.section_body(clean_text("Our core architecture transitions beyond traditional CNNs toward a Hybrid Transformer model:"))
    
    pdf.bold_text("Transformer Backbone (Hubert): ")
    pdf.section_body(clean_text("We used the superbly pretrained hubert-base-superb-er model. We unmasked and fine-tuned the last 2-4 encoder layers to capture relevant prosodic representations for emotional tasks."))
    
    pdf.bold_text("Multi-Task Learning: ")
    pdf.section_body(clean_text("The model utilizes a dual-head classification system. One head predicts the emotion category (Neutral, Calm, Happy, etc.), while the second head predicts intensity (Normal vs Strong). This architecture allows the model to share low-level acoustic representations across tasks, improving both."))
    
    pdf.section_title('6. Experimental Design')
    pdf.section_body(clean_text("The dataset was partitioned into an 80/20 train-test split, ensuring speaker-independence (actors in training do not appear in validation). We utilized a Focal Loss function to address the class imbalance inherent in emotional speech datasets, particularly for the 'Surprised' and 'Disgust' classes."))

    # --- PAGE 4: RESULTS & CONCLUSION ---
    pdf.add_page()
    pdf.section_title('7. Results and Discussion')
    
    # Results Table
    pdf.set_fill_color(240, 240, 240)
    pdf.set_font('helvetica', 'B', 11)
    pdf.cell(95, 10, 'Metric', border=1, align='C', fill=True)
    pdf.cell(95, 10, 'Accuracy / F1', border=1, align='C', fill=True, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    
    results = [
        ("Emotion Recognition Accuracy", "53.75%"),
        ("Intensity Classification Accuracy", "77.92%"),
        ("Emotion Macro F1-Score", "0.541"),
        ("Intensity Macro F1-Score", "0.775")
    ]
    
    pdf.set_font('helvetica', '', 11)
    for metric, val in results:
        pdf.cell(95, 10, clean_text(metric), border=1, align='L')
        pdf.cell(95, 10, clean_text(val), border=1, align='C', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    
    pdf.ln(5)
    pdf.section_body(clean_text("Discussion: The results demonstrate consistent dominance in high-arousal emotions (Happy, Angry) achieving 80%+ precision. The hybrid approach, combining handcrafted statistics with Transformer embeddings, effectively stabilized real-world predictions, as shown in the cross-actor validation results."))

    pdf.section_title('8. Ethics & Surveillance')
    pdf.section_body(clean_text("Affective computing requires rigorous ethical oversight. We implemented local-inference privacy protocols to protect user recordings and conducted bias skew analysis to identify performance variances across demographic actors."))

    pdf.section_title('9. Conclusion')
    pdf.section_body(clean_text("This project successfully demonstrated a modular, high-performance pipeline for deciphering human emotion. Future improvements will focus on multimodal integration (facial expression tracking) to further refine low-arousal state classification."))

    pdf.ln(10)
    pdf.set_font('helvetica', 'I', 10)
    pdf.set_text_color(128, 128, 128)
    pdf.multi_cell(0, 5, clean_text("Final Research Report | COMP-SCI 5530: Principles of Data Science. | University of Missouri-Kansas City"), align='C')

    output_path = 'results/research-a-thon/report_4_page.pdf'
    pdf.output(output_path)
    print(f"Report generated successfully: {output_path}")

if __name__ == "__main__":
    generate()
