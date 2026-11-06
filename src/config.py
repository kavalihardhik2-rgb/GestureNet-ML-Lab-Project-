"""Central configuration for GestureNet."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "Sign-Language-Digits-Dataset" / "Dataset"
OUT_DIR = ROOT / "outputs"
MODEL_DIR = ROOT / "models"

IMG_SIZE = 64          # images are resized to IMG_SIZE x IMG_SIZE
NUM_CLASSES = 10
SEED = 42
VAL_FRAC = 0.15
TEST_FRAC = 0.15

BATCH_SIZE = 32
EPOCHS = 60
LEARNING_RATE = 1e-3

# Each hand gesture (ASL digit sign) is mapped to a computer command,
# which is what turns the classifier into a human-computer interface.
GESTURE_ACTIONS = {
    0: "Fist (0)        -> Pause / Stop",
    1: "One (1)         -> Select / Click",
    2: "Two (2)         -> Scroll up",
    3: "Three (3)       -> Scroll down",
    4: "Four (4)        -> Volume up",
    5: "Open palm (5)   -> Play / Resume",
    6: "Six (6)         -> Volume down",
    7: "Seven (7)       -> Next track / slide",
    8: "Eight (8)       -> Previous track / slide",
    9: "Nine (9)        -> Screenshot",
}
