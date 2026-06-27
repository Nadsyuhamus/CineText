import joblib, pathlib
import numpy as np

ROOT_DIR = pathlib.Path("c:/Users/fasha/OneDrive/Documents/GitHub/CineText")
model = joblib.load(ROOT_DIR / "best_model.pkl")
vec = joblib.load(ROOT_DIR / "best_vectorizer.pkl")

print("Classes:", model.classes_)
for word in ["hate", "interesting", "actor", "excellent", "worst", "not", "like"]:
    if word in vec.vocabulary_:
        idx = vec.vocabulary_[word]
        print(f"{word}: {model.coef_[0][idx]:.4f}")
    else:
        print(f"{word}: NOT IN VOCABULARY")

coef_abs = np.abs(model.coef_[0])
threshold = np.percentile(coef_abs[coef_abs > 0], 80)
print(f"80th percentile threshold: {np.percentile(coef_abs[coef_abs > 0], 80):.4f}")
