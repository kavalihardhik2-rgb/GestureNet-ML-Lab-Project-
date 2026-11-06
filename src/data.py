"""Loading and splitting the Sign Language Digits hand-gesture dataset."""
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

from config import DATA_DIR, IMG_SIZE, SEED, VAL_FRAC, TEST_FRAC


def load_image(path, size=IMG_SIZE):
    """Read an image, convert to RGB, resize and scale pixels to [0, 1]."""
    img = Image.open(path).convert("RGB").resize((size, size), Image.BILINEAR)
    return np.asarray(img, dtype=np.float32) / 255.0


def load_dataset(data_dir=DATA_DIR, size=IMG_SIZE):
    """Return X (N, size, size, 3) and y (N,) where the label is the folder name 0-9."""
    X, y = [], []
    for class_dir in sorted(p for p in data_dir.iterdir() if p.is_dir()):
        label = int(class_dir.name)
        for f in sorted(class_dir.iterdir()):
            if f.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                X.append(load_image(f, size))
                y.append(label)
    return np.stack(X), np.array(y, dtype=np.int64)


def split_dataset(X, y):
    """Stratified 70 / 15 / 15 train / validation / test split (fixed seed)."""
    X_tmp, X_test, y_tmp, y_test = train_test_split(
        X, y, test_size=TEST_FRAC, stratify=y, random_state=SEED)
    val_rel = VAL_FRAC / (1 - TEST_FRAC)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp, test_size=val_rel, stratify=y_tmp, random_state=SEED)
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


if __name__ == "__main__":
    X, y = load_dataset()
    (Xtr, ytr), (Xva, yva), (Xte, yte) = split_dataset(X, y)
    print(f"Total images: {len(X)}  shape: {X.shape[1:]}")
    print(f"Train: {len(Xtr)}  Val: {len(Xva)}  Test: {len(Xte)}")
    print("Per-class counts:", np.bincount(y))
