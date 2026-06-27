import joblib
import nltk
nltk.download('wordnet')
vec = joblib.load("best_vectorizer.pkl")
model = joblib.load("best_model.pkl")

tests = [
    "i hate this movie",
    "i hate this",
    "hate",
    "this movie is very interesting i like it",
    "worst movie ever complete waste of time",
]

for t in tests:
    from bs4 import BeautifulSoup
    import re
    from nltk.stem import WordNetLemmatizer
    from nltk.corpus import stopwords
    
    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()
    
    text = BeautifulSoup(t, "html.parser").get_text().lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    words = [lemmatizer.lemmatize(w) for w in text.split() if w not in stop_words and len(w) > 2]
    cleaned = " ".join(words)
    
    vec_text = vec.transform([cleaned])
    pred = model.predict(vec_text)[0]
    score = model.decision_function(vec_text)[0]
    print(f"Input: '{t}' → cleaned: '{cleaned}' → {pred} (score: {score:.3f})")