# 🎬 CineScope — Movie Review Sentiment Analyzer

> A high-performance NLP sentiment analysis platform for movie reviews, powered by classical machine learning and deep learning transformer architectures.

🔗 **Live Demo:** [https://thecinescope.streamlit.app/](https://thecinescope.streamlit.app/)

---

## 📌 Overview

CineScope is a Streamlit-based web application that classifies movie reviews as **positive** or **negative** using two model pipelines:

- **TF-IDF + LinearSVC** — fast classical ML model optimized for production inference
- **DistilBERT Transformer** — deep learning model for rich contextual understanding

The app supports **multilingual input** via automatic translation (Google Translator API), making it accessible to non-English speakers.

---

## ✨ Features

- 🔍 Real-time sentiment analysis on any movie review text
- 🌐 Auto-detection and translation of non-English input
- 📊 Token impact breakdown with word-level sentiment highlighting
- ☁️ Per-review positive and negative word clouds
- 📈 Model performance visualizations (accuracy chart, confusion matrix, feature importance)
- 🧠 Pipeline architecture details and comparative metrics table
- 👥 Team credits page

---

## 🗂️ Project Structure

```
CineText/
├── best_model.pkl              # Saved LinearSVC model (auto-generated after training)
├── best_vectorizer.pkl         # Saved TF-IDF vectorizer (auto-generated after training)
├── requirements.txt            # Unified dependencies for all scripts
│
├── backend_DataNLP/
│   └── nlp_pipeline.py         # Training pipeline: preprocessing, vectorization, model training
│
└── frontend_StreamlitApp/
    └── app.py                  # Streamlit web application
```

---

## ⚙️ Setup & Installation

### Prerequisites
- Python 3.11 (recommended — tested and stable)
- Git

### 1. Clone the repository
```bash
git clone https://github.com/your-repo/CineText.git
cd CineText
```

### 2. Install dependencies
```bash
python -m pip install -r requirements.txt
```

### 3. Train the model
```bash
python backend_DataNLP/nlp_pipeline.py
```
This will:
- Download the IMDB 50k dataset via KaggleHub
- Train all 4 classical model variants
- Evaluate DistilBERT on the test set
- Save `best_model.pkl` and `best_vectorizer.pkl` to the repo root

### 4. Run the app
```bash
python -m streamlit run frontend_StreamlitApp/app.py
```

---

## 🧠 Model Pipeline

### Preprocessing
1. HTML tag stripping via BeautifulSoup
2. Lowercase normalization and punctuation removal
3. Stopword filtering (NLTK English) — negation words (`not`, `never`, `hardly` etc.) are preserved
4. Lemmatization via WordNetLemmatizer

### Classical Models Trained
| Model | Vectorizer | Notes |
|---|---|---|
| Naïve Bayes | Bag-of-Words | Baseline |
| LinearSVC | Bag-of-Words | Baseline |
| Naïve Bayes | TF-IDF (1–3 ngrams) | Improved baseline |
| **LinearSVC** | **TF-IDF (1–3 ngrams)** | **Best classical model** |

### Deep Learning
- **DistilBERT** (`distilbert-base-uncased-finetuned-sst-2-english`) evaluated on raw text (no preprocessing)

### Results
| Model | Accuracy | F1-Score |
|---|---|---|
| BoW + Naïve Bayes | 85.17% | 85.17% |
| BoW + LinearSVC | 84.91% | 84.91% |
| TF-IDF + Naïve Bayes | 87.89% | 87.89% |
| **TF-IDF + LinearSVC** | **90.12%** | **90.12%** |
| DistilBERT | 89.16% | 89.15% |

---

## 📦 Dependencies

See [`requirements.txt`](./requirements.txt) for the full list. Key packages:

```
streamlit==1.32.0
scikit-learn>=1.2.2
torch>=2.0.0
transformers>=4.35.0
nltk>=3.8.1
deep-translator>=1.11.0
st-annotated-text>=4.0.0
```

---

## 👥 Project Team

| Name | Matric ID | Role |
|---|---|---|
| Mun Weng Yann | A24AI0067 | NLP Pipeline Engineer |
| Nur Nadsyuha Bt. Mustafa | A24AI0117 | Frontend Systems Developer |
| Areesha Nabila Bt. Dick Hilmi | A24AI0098 | Data Systems Analyst |
| Faqihah Humaira' Bt. Muhammad Firhat | A24AI0028 | Core Project Lead |

