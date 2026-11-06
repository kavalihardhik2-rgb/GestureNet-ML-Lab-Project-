"""Classify a single image file and print the mapped HCI command.

Usage:  python predict.py path/to/hand.jpg
"""
import os
import sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
from tensorflow import keras

from config import GESTURE_ACTIONS, MODEL_DIR
from data import load_image


def main():
    if len(sys.argv) < 2:
        sys.exit("Usage: python predict.py <image_path>")
    model = keras.models.load_model(MODEL_DIR / "gesturenet.keras")
    x = load_image(sys.argv[1])[None, ...]
    probs = model.predict(x, verbose=0)[0]
    top = np.argsort(probs)[::-1][:3]
    print(f"Predicted gesture: {top[0]}  ({probs[top[0]]:.1%})")
    print(f"Command:           {GESTURE_ACTIONS[top[0]]}")
    print("Top-3:", ", ".join(f"{c} ({probs[c]:.1%})" for c in top))


if __name__ == "__main__":
    main()
