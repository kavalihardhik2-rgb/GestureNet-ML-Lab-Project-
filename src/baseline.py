"""Classical ML baseline: HOG features + SVM (and Logistic Regression).
Gives a reference point to show what the CNN adds over hand-crafted features."""
import json
import time

import numpy as np
from skimage.feature import hog
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from config import OUT_DIR, SEED
from data import load_dataset, split_dataset


def hog_features(X):
    gray = X.mean(axis=-1)  # RGB -> grayscale
    return np.stack([
        hog(im, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2),
            block_norm="L2-Hys")
        for im in gray
    ])


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    X, y = load_dataset()
    (Xtr, ytr), (Xva, yva), (Xte, yte) = split_dataset(X, y)
    Ftr, Fva, Fte = hog_features(Xtr), hog_features(Xva), hog_features(Xte)
    print("HOG feature length:", Ftr.shape[1])

    models = {
        "HOG + Logistic Regression": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=3000, C=0.1, random_state=SEED)),
        "HOG + SVM (RBF)": make_pipeline(
            StandardScaler(), SVC(kernel="rbf", C=10, gamma="scale", random_state=SEED)),
    }
    results = {}
    for name, clf in models.items():
        t = time.time()
        clf.fit(Ftr, ytr)
        val_acc = accuracy_score(yva, clf.predict(Fva))
        pred = clf.predict(Fte)
        results[name] = {
            "val_accuracy": round(val_acc, 4),
            "test_accuracy": round(accuracy_score(yte, pred), 4),
            "test_macro_f1": round(f1_score(yte, pred, average="macro"), 4),
            "train_seconds": round(time.time() - t, 1),
        }
        print(name, results[name])

    with open(OUT_DIR / "baseline_results.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
