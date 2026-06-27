import joblib
model = joblib.load("best_model.pkl")
vec = joblib.load("best_vectorizer.pkl")

print("Classes:", model.classes_)

# Check "hate" and "excellent"
for word in ["hate", "excellent", "worst", "great","interesting"]:
    if word in vec.vocabulary_:
        idx = vec.vocabulary_[word]
        print(f"{word}: {model.coef_[0][idx]:.4f}")