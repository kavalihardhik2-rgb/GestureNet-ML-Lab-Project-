"""GestureNet: a compact CNN for hand-gesture classification."""
from tensorflow import keras
from tensorflow.keras import layers

from config import IMG_SIZE, NUM_CLASSES


def augmentation():
    """On-the-fly augmentation (active only during training).
    No horizontal flip: mirrored hands can change a sign's meaning."""
    return keras.Sequential([
        layers.RandomRotation(0.08),
        layers.RandomZoom(0.12),
        layers.RandomTranslation(0.08, 0.08),
        layers.RandomBrightness(0.15, value_range=(0.0, 1.0)),
        layers.RandomContrast(0.15),
    ], name="augmentation")


def conv_block(x, filters, dropout):
    x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D()(x)
    return layers.Dropout(dropout)(x)


def build_gesturenet(img_size=IMG_SIZE, num_classes=NUM_CLASSES):
    inputs = keras.Input(shape=(img_size, img_size, 3), name="image")
    x = augmentation()(inputs)
    x = conv_block(x, 32, 0.10)   # 64 -> 32
    x = conv_block(x, 64, 0.15)   # 32 -> 16
    x = conv_block(x, 128, 0.20)  # 16 -> 8
    x = conv_block(x, 256, 0.25)  # 8  -> 4
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="gesture")(x)
    return keras.Model(inputs, outputs, name="GestureNet")
