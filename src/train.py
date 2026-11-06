"""Train GestureNet and save the best model + training history."""
import json
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf
from tensorflow import keras

from config import BATCH_SIZE, EPOCHS, LEARNING_RATE, MODEL_DIR, OUT_DIR, SEED
from data import load_dataset, split_dataset
from model import build_gesturenet


def main():
    keras.utils.set_random_seed(SEED)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    X, y = load_dataset()
    (Xtr, ytr), (Xva, yva), _ = split_dataset(X, y)
    print(f"Train {Xtr.shape}, Val {Xva.shape}")

    model = build_gesturenet()
    model.compile(
        optimizer=keras.optimizers.Adam(LEARNING_RATE),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )
    model.summary()

    callbacks = [
        keras.callbacks.ModelCheckpoint(MODEL_DIR / "gesturenet.keras",
                                        monitor="val_accuracy", save_best_only=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                          patience=5, min_lr=1e-5),
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=14,
                                      restore_best_weights=True),
    ]
    hist = model.fit(Xtr, ytr, validation_data=(Xva, yva), epochs=EPOCHS,
                     batch_size=BATCH_SIZE, callbacks=callbacks, verbose=2)

    history = {k: [float(v) for v in vals] for k, vals in hist.history.items()}
    with open(OUT_DIR / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    print("Best val accuracy:", max(history["val_accuracy"]))


if __name__ == "__main__":
    main()
