"""Evaluate the trained GestureNet on the held-out test set and make figures."""
import json
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score)
from tensorflow import keras

from config import GESTURE_ACTIONS, MODEL_DIR, OUT_DIR
from data import load_dataset, split_dataset

plt.rcParams.update({"figure.dpi": 130, "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False})
INK, MUTED, BLUE, ORANGE = "#1f2328", "#6e7781", "#2563eb", "#ea7317"


def plot_history(history):
    ep = np.arange(1, len(history["loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, key, title in [(axes[0], "accuracy", "Accuracy"), (axes[1], "loss", "Loss")]:
        ax.plot(ep, history[key], color=BLUE, lw=2, label="Train")
        ax.plot(ep, history["val_" + key], color=ORANGE, lw=2, label="Validation")
        ax.set_title(title, color=INK, loc="left", fontweight="bold")
        ax.set_xlabel("Epoch", color=MUTED)
        ax.grid(alpha=0.25)
        ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "training_curves.png")
    plt.close(fig)


def plot_confusion(cm):
    fig, ax = plt.subplots(figsize=(5.6, 5))
    ax.imshow(cm, cmap="Blues")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=9,
                    color="white" if cm[i, j] > cm.max() / 2 else INK)
    ax.set_xticks(range(10)); ax.set_yticks(range(10))
    ax.set_xlabel("Predicted gesture", color=MUTED)
    ax.set_ylabel("True gesture", color=MUTED)
    ax.set_title("GestureNet — test confusion matrix", color=INK, loc="left", fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "confusion_matrix.png")
    plt.close(fig)


def plot_samples(X, y, pred, probs, n=12):
    rng = np.random.default_rng(0)
    idx = rng.choice(len(X), n, replace=False)
    fig, axes = plt.subplots(2, n // 2, figsize=(12, 4.6))
    for ax, i in zip(axes.ravel(), idx):
        ax.imshow(X[i]); ax.axis("off")
        ok = pred[i] == y[i]
        ax.set_title(f"true {y[i]} / pred {pred[i]}\n{probs[i].max():.0%}",
                     fontsize=9, color="#15803d" if ok else "#b91c1c")
    fig.suptitle("Sample test predictions", color=INK, fontweight="bold", x=0.02, ha="left")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "sample_predictions.png")
    plt.close(fig)


def plot_dataset_overview(X, y):
    fig, axes = plt.subplots(1, 10, figsize=(13, 1.9))
    for c, ax in enumerate(axes):
        ax.imshow(X[np.where(y == c)[0][0]]); ax.axis("off")
        ax.set_title(f"class {c}", fontsize=9, color=INK)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "dataset_examples.png")
    plt.close(fig)


def plot_model_comparison(cnn_acc):
    path = OUT_DIR / "baseline_results.json"
    if not path.exists():
        return
    base = json.loads(path.read_text())
    names = list(base) + ["GestureNet CNN"]
    accs = [base[k]["test_accuracy"] for k in base] + [cnn_acc]
    fig, ax = plt.subplots(figsize=(7, 2.8))
    colors = [MUTED] * len(base) + [BLUE]
    bars = ax.barh(names, [a * 100 for a in accs], color=colors, height=0.55)
    for b, a in zip(bars, accs):
        ax.text(b.get_width() + 0.5, b.get_y() + b.get_height() / 2, f"{a:.1%}",
                va="center", color=INK, fontsize=9)
    ax.set_xlim(0, 105)
    ax.set_xlabel("Test accuracy (%)", color=MUTED)
    ax.set_title("Model comparison on the same test split", color=INK, loc="left", fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "model_comparison.png")
    plt.close(fig)


def main():
    X, y = load_dataset()
    _, _, (Xte, yte) = split_dataset(X, y)
    model = keras.models.load_model(MODEL_DIR / "gesturenet.keras")

    probs = model.predict(Xte, verbose=0)
    pred = probs.argmax(axis=1)
    acc = accuracy_score(yte, pred)
    f1 = f1_score(yte, pred, average="macro")
    cm = confusion_matrix(yte, pred)
    report = classification_report(yte, pred, digits=4,
                                   target_names=[f"gesture_{i}" for i in range(10)])
    print(f"Test accuracy: {acc:.4f}   Macro F1: {f1:.4f}\n")
    print(report)

    (OUT_DIR / "classification_report.txt").write_text(
        f"Test accuracy: {acc:.4f}\nMacro F1: {f1:.4f}\n\n{report}")
    with open(OUT_DIR / "test_metrics.json", "w") as f:
        json.dump({"test_accuracy": round(acc, 4), "test_macro_f1": round(f1, 4),
                   "test_samples": int(len(yte)),
                   "params": int(model.count_params())}, f, indent=2)

    history_path = OUT_DIR / "history.json"
    if history_path.exists():
        plot_history(json.loads(history_path.read_text()))
    plot_confusion(cm)
    plot_samples(Xte, yte, pred, probs)
    plot_dataset_overview(X, y)
    plot_model_comparison(acc)
    print("Figures saved to", OUT_DIR)


if __name__ == "__main__":
    main()
