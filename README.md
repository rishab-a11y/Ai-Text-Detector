# 🤖 AI Text Detector

A machine learning web application that detects whether a piece of text is written by **AI or a Human** using an ensemble of TF-IDF + Logistic Regression and fine-tuned RoBERTa models.

![Python](https://img.shields.io/badge/Python-3.12-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green)
![RoBERTa](https://img.shields.io/badge/Model-RoBERTa-orange)
![Accuracy](https://img.shields.io/badge/Accuracy-98%25-brightgreen)

---

## 🎯 Features

- **Dual Model Ensemble** — Combines TF-IDF and RoBERTa for 98% accuracy
- **File Upload Support** — Accepts PDF, DOCX, and TXT files
- **Sentence-Level Highlighting** — Shows exactly which sentences look AI-written
- **Confidence Scores** — Provides detailed probability breakdown
- **Download Report** — Export analysis results as a text file
- **Beautiful UI** — Clean dark-themed web interface

---

## 🧠 How It Works
```
User Input (Text / File)
        ↓
Text Extraction (PDF/DOCX/TXT)
        ↓
┌─────────────────────────────┐
│  TF-IDF + Logistic          │  Weight: 40%
│  Regression Model           │
└─────────────────────────────┘
        +
┌─────────────────────────────┐
│  Fine-tuned RoBERTa         │  Weight: 60%
│  Transformer Model          │
└─────────────────────────────┘
        ↓
  Ensemble Prediction
        ↓
  Result + Confidence Score
```

---

## 📊 Model Performance

| Model | Accuracy | F1-Score |
|-------|----------|----------|
| TF-IDF + Logistic Regression | 98.79% | 0.99 |
| Fine-tuned RoBERTa | 97.00% | 0.97 |
| **Ensemble (Final)** | **98%** | **0.98** |

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.12 |
| ML Framework | PyTorch, HuggingFace Transformers |
| Classical ML | Scikit-learn |
| Backend | FastAPI |
| Frontend | HTML, CSS, JavaScript |
| Model | RoBERTa-base (fine-tuned) |
| Dataset | DAIGT V2 (44,868 samples) |

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10 or higher
- 4GB RAM minimum
- GPU recommended for RoBERTa training

### Installation

**1. Clone the repository:**
```bash
git clone https://github.com/rishab-a11y/Ai-Text-Detector.git
cd Ai-Text-Detector
```

**2. Create virtual environment:**
```bash
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Mac/Linux
```

**3. Install dependencies:**
```bash
pip install -r requirements.txt
```

**4. Download and prepare data:**
```bash
python download_data.py
python prepare.py
```

**5. Train the TF-IDF model:**
```bash
python train.py
```

**6. Train RoBERTa (recommended on Google Colab):**
```bash
python trainroberta.py
```

**7. Run the web app:**
```bash
python -m uvicorn app.main:app --reload
```

**8. Open in browser:**
```
http://127.0.0.1:8000
```

---

## 📁 Project Structure
```
Ai-Text-Detector/
├── app/
│   ├── main.py              # FastAPI backend
│   ├── model.py             # Model loading & prediction
│   ├── file_handler.py      # File parsing (PDF/DOCX/TXT)
│   └── static/
│       └── index.html       # Frontend UI
├── download_data.py         # Dataset download
├── explore.py               # Data exploration
├── prepare.py               # Data preprocessing
├── train.py                 # TF-IDF model training
├── preparedataroberta.py    # RoBERTa data preparation
├── trainroberta.py          # RoBERTa fine-tuning
├── ensemble.py              # Ensemble model
├── evaluateRoBerta.py       # Model evaluation
├── requirements.txt         # Dependencies
└── README.md
```

---

## 📈 Dataset

- **Name:** DAIGT V2 Train Dataset
- **Source:** Kaggle
- **Size:** 44,868 samples
- **Labels:** 0 = Human Written, 1 = AI Generated
- **Split:** 80% training, 20% testing

---

## 🔍 Limitations

- Model performs best on **essay-style academic text**
- May over-detect AI in formal human writing
- Short texts (< 50 characters) not supported
- RoBERTa requires GPU for fast inference

---

## 🎓 About

This project was developed as a **BTech final year project** by Rishab.

Built with ❤️ using Python, FastAPI, and HuggingFace Transformers.

---

## 📄 License

MIT License — feel free to use and modify!