import re
import joblib
import pandas as pd
import nltk
import seaborn as sns
import matplotlib.pyplot as plt
import kagglehub
import os
import torch
from tqdm import tqdm
import pathlib

from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from deep_translator import GoogleTranslator

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score

# Advanced NLP
from transformers import pipeline

nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")

class NLPipeline:
    def __init__(self):
        self.stop_words = set(stopwords.words("english"))
        self.lemmatizer = WordNetLemmatizer()

        self.bow_vectorizer = CountVectorizer(max_features=10000)
        self.tfidf_vectorizer = TfidfVectorizer(max_features=15000, ngram_range=(1,3))

        self.device = 0 if torch.cuda.is_available() else (-1 if not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available() else "mps")
        print(f"Assigning Deep Learning models to device target: {self.device}")
        
        self.bert_pipeline = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            framework="pt",
            device=self.device
        )

    def clean_text(self, text: str) -> str:
        """Deterministic text normalization pipeline."""
        # Strip HTML tags
        text = BeautifulSoup(str(text), "html.parser").get_text()
        # Enforce uniform lowercasing & clear punctuation/digits
        text = text.lower()
        text = re.sub(r"[^a-z\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        
        # Token extraction, filtering, and structural dictionary lemma reduction
        NEGATION_WORDS = {"not", "no", "nor", "never", "neither", "nobody", "nothing",
                      "nowhere", "hardly", "scarcely", "barely", "without"}
        
        words = text.split()
        cleaned_words = [
            self.lemmatizer.lemmatize(word)
            for word in words
            if (word not in self.stop_words or word in NEGATION_WORDS) and len(word) > 2
        ]
        return " ".join(cleaned_words)
    
    def translate_to_english(self, text: str) -> str:
        """Fault-tolerant translation routing mechanism."""
        try:
            return GoogleTranslator(source="auto", target="en").translate(text)
        except Exception:
            return text

    def run_classical_models(self, X_train_clean, X_test_clean, y_train, y_test):
        """Trains and evaluates statistical Scikit-Learn estimators."""
        print("\n--- Vectorizing Text Data ---")
        X_train_bow = self.bow_vectorizer.fit_transform(X_train_clean)
        X_test_bow = self.bow_vectorizer.transform(X_test_clean)

        X_train_tfidf = self.tfidf_vectorizer.fit_transform(X_train_clean)
        X_test_tfidf = self.tfidf_vectorizer.transform(X_test_clean)

        classical_scenarios = {
            "BoW + Naive Bayes": (MultinomialNB(), X_train_bow, X_test_bow),
            "BoW + SVM": (LinearSVC(max_iter=5000), X_train_bow, X_test_bow),
            "TF-IDF + Naive Bayes": (MultinomialNB(), X_train_tfidf, X_test_tfidf),
            "TF-IDF + SVM": (LinearSVC(max_iter=5000), X_train_tfidf, X_test_tfidf)
        }

        results = []
        trained_models = {}

        for name, (model, Xtr, Xte) in classical_scenarios.items():
            print(f"Training Model Variant: {name}")
            model.fit(Xtr, y_train)
            predictions = model.predict(Xte)

            trained_models[name] = model
            report = classification_report(y_test, predictions, output_dict=True)

            results.append({
                "Model": name,
                "Accuracy": accuracy_score(y_test, predictions),
                "Precision": report["weighted avg"]["precision"],
                "Recall": report["weighted avg"]["recall"],
                "F1-Score": report["weighted avg"]["f1-score"]
            })
            
        return results, trained_models
    
    def evaluate_distilbert(self, X_test_raw, y_test):
        """Evaluates Transformer architecture utilizing batch streams on raw text."""
        print("\n--- Running Inference via DistilBERT Pipeline ---")
        
        # Generator function to support pipeline dataset streaming
        def data_stream():
            for text in X_test_raw:
                yield str(text)

        bert_predictions = []
        # Leverage batch processing size parameters to scale up through GPU acceleration limits
        for output in tqdm(self.bert_pipeline(data_stream(), batch_size=32, truncation=True, max_length=512), total=len(X_test_raw)):
            # Normalize transformer output tokens into alignment with ground-truth classes
            sentiment = output["label"].lower()
            bert_predictions.append(sentiment)

        # Standardizing assessment parameters
        accuracy = accuracy_score(y_test, bert_predictions)
        precision = precision_score(y_test, bert_predictions, pos_label="positive", average="weighted")
        recall = recall_score(y_test, bert_predictions, pos_label="positive", average="weighted")
        f1 = f1_score(y_test, bert_predictions, pos_label="positive", average="weighted")

        print("\nDistilBERT Classification Report:")
        print(classification_report(y_test, bert_predictions))

        return {
            "Model": "DistilBERT",
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1-Score": f1
        }
    
# --- Execution
if __name__ == "__main__":
    # --- Load Dataset ---
    dataset_dir = kagglehub.dataset_download("lakshmi25npathi/imdb-dataset-of-50k-movie-reviews")
    csv_path = os.path.join(dataset_dir, os.listdir(dataset_dir)[0])
    
    df = pd.read_csv(csv_path)
    df = df.rename(columns={"review": "text", "sentiment": "label"})
    
    # Standardize data labeling schemas
    df["label"] = df["label"].str.lower()

    # --- Initialize Custom Pipeline ---
    nlp_eng = NLPipeline()

    # --- Stratified Train-Test Split ---
    # Split raw text first to preserve linguistic semantics for DistilBERT
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        df["text"], 
        df["label"], 
        test_size=0.2, 
        random_state=42, 
        stratify=df["label"]
    )

    # --- Text Engineering (Classical Models Requirement Only) ---
    print("\nExecuting Preprocessing Engine for Statistical Estimators...")
    X_train_clean = X_train_raw.apply(nlp_eng.clean_text)
    X_test_clean = X_test_raw.apply(nlp_eng.clean_text)

    # --- Execute Classical Training Matrix ---
    metrics, models = nlp_eng.run_classical_models(X_train_clean, X_test_clean, y_train, y_test)

    # --- Execute Transformer Inference Pipeline ---
    # Evaluated on raw text variables to preserve transformer context
    bert_metrics = nlp_eng.evaluate_distilbert(X_test_raw, y_test)
    metrics.append(bert_metrics)

    # --- Consolidated System Analysis Matrix ---
    comparison_table = pd.DataFrame(metrics)
    print("\n================ FINAL MATRIX RECOGNITION ================")
    print(comparison_table.to_string(index=False))

    # --- Save Best Statistical Model Configuration ---
    classical_metrics = [m for m in metrics if m["Model"] != "DistilBERT"]
    best_classical_meta = max(classical_metrics, key=lambda x: x["F1-Score"])
    best_classical_name = best_classical_meta["Model"]
    
    best_vectorizer = nlp_eng.bow_vectorizer if "BoW" in best_classical_name else nlp_eng.tfidf_vectorizer
    
    ROOT_DIR = pathlib.Path(__file__).resolve().parent.parent  # goes up from backend_DataNLP to CineText
MAIN_DIR = ROOT_DIR / "main"
MAIN_DIR.mkdir(exist_ok=True)  # creates /main if it doesn't exist

joblib.dump(models[best_classical_name], MAIN_DIR / "best_model.pkl")
joblib.dump(best_vectorizer, MAIN_DIR / "best_vectorizer.pkl")
print(f"\nExported optimal classical pipeline artifact ({best_classical_name}) to {MAIN_DIR}")