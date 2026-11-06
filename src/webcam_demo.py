"""Real-time HCI demo: classify hand gestures from your webcam.

Hold your hand inside the green box against a plain background (the training
images were taken that way). The predicted gesture and its mapped command are
shown on screen. A command only "fires" after the same gesture is held steadily
for STABLE_FRAMES frames with confidence >= CONF_THRESHOLD, which avoids
flickering between classes.

Usage:  python webcam_demo.py          (press q to quit)
Requires: pip install opencv-python
"""
import os
from collections import deque

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import cv2
import numpy as np
from tensorflow import keras

from config import GESTURE_ACTIONS, IMG_SIZE, MODEL_DIR

CONF_THRESHOLD = 0.80
STABLE_FRAMES = 12
BOX = 300  # side of the square region of interest, in pixels


def main():
    model = keras.models.load_model(MODEL_DIR / "gesturenet.keras")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise SystemExit("Could not open webcam.")
    recent = deque(maxlen=STABLE_FRAMES)
    last_fired = None

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        x0, y0 = w - BOX - 40, (h - BOX) // 2
        roi = frame[y0:y0 + BOX, x0:x0 + BOX]

        rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        x = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE)).astype(np.float32) / 255.0
        probs = model.predict(x[None], verbose=0)[0]
        cls, conf = int(probs.argmax()), float(probs.max())
        recent.append(cls if conf >= CONF_THRESHOLD else -1)

        stable = len(recent) == STABLE_FRAMES and len(set(recent)) == 1 and recent[0] != -1
        if stable and cls != last_fired:
            last_fired = cls
            print("COMMAND:", GESTURE_ACTIONS[cls])  # hook real actions in here

        cv2.rectangle(frame, (x0, y0), (x0 + BOX, y0 + BOX), (0, 200, 0), 2)
        cv2.putText(frame, f"Gesture {cls}  {conf:.0%}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 200, 0), 2)
        if last_fired is not None:
            cv2.putText(frame, GESTURE_ACTIONS[last_fired].split("->")[1].strip(),
                        (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 165, 255), 2)
        cv2.imshow("GestureNet - press q to quit", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
