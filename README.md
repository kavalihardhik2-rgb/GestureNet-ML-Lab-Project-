# GestureNet: Hand Gesture Classification for Human–Computer Interaction

Machine Learning Lab Project · Domain: Image Classification

GestureNet is a convolutional neural network that recognises 10 static hand
gestures from RGB images. Each gesture is mapped to a computer command
(play, pause, scroll, volume and so on), so the classifier can be used as a
touch-free interface. A classical HOG + SVM baseline is included for comparison.

## 1. Dataset

**Sign Language Digits Dataset** (Turkey Ankara Ayrancı Anadolu High School;
project leads Zeynep Dikle and Arda Mavi)

- Source: https://github.com/ardamavi/Sign-Language-Digits-Dataset
- License: Apache 2.0
- 2,062 RGB images at 100 × 100 px; 10 classes (hand signs for digits 0–9); about 205 images per class
- 218 participants contributed 10 images each, taken against a plain background
- Citation: Mavi, A. (2020). *A New Dataset and Proposed Convolutional Neural
  Network Architecture for Classification of American Sign Language Digits.*
  arXiv:2011.08927

The dataset is included in `data/` so the project runs offline.

| Class | Gesture | Mapped HCI command |
|---|---|---|
| 0 | Fist | Pause / Stop |
| 1 | One finger | Select / Click |
| 2 | Two fingers | Scroll up |
| 3 | Three fingers | Scroll down |
| 4 | Four fingers | Volume up |
| 5 | Open palm | Play / Resume |
| 6 | Six | Volume down |
| 7 | Seven | Next track / slide |
| 8 | Eight | Previous track / slide |
| 9 | Nine | Screenshot |

The command mapping is defined in `src/config.py`; change it to suit your application.

**Data split:** stratified 70 / 15 / 15 split (train 1,442 / validation 310 / test 310)
with a fixed seed of 42. The test set is used only once, for the final evaluation.

## 2. Method

**Preprocessing:** images are resized to 64 × 64 and pixel values are scaled to [0, 1].

**Augmentation (training only):** random rotation (±8%), zoom (±12%),
translation (±8%), brightness and contrast changes. Images are not flipped
horizontally, because a mirrored hand can change a sign's meaning.

**GestureNet architecture (about 1.2 M parameters):**

```
Input 64×64×3
→ [Conv3×3-BN-ReLU ×2 → MaxPool → Dropout] × 4   (32, 64, 128, 256 filters)
→ GlobalAveragePooling → Dense(128, ReLU) → Dropout(0.4) → Dense(10, softmax)
```

**Training:** Adam optimiser (learning rate 1e-3) with sparse categorical
cross-entropy and batch size 32, for up to 60 epochs. The learning rate is halved
when validation loss stops improving (ReduceLROnPlateau). Training stops early
when validation loss stops improving (EarlyStopping), and the checkpoint with the
best validation accuracy is kept.

**Baselines:** HOG features (1,764-dimensional) with Logistic Regression, and
HOG features with an RBF SVM.

## 3. Results (held-out test set, 310 images)

| Model | Test accuracy | Macro F1 |
|---|---|---|
| HOG + Logistic Regression | 84.84% | 0.849 |
| HOG + SVM (RBF) | 87.42% | 0.875 |
| **GestureNet CNN (ours)** | **99.68%** | **0.997** |

GestureNet misclassified 1 of 310 test images: a "4" predicted as "5", two
gestures that differ by only one finger. All other classes scored 100%.

Training ran for all 60 epochs on CPU, at about 12 seconds per epoch. During
roughly the first 25 epochs, validation accuracy jumped up and down while the
BatchNorm running statistics settled. It became stable once the learning rate
was reduced. The best checkpoint reached 100% validation accuracy.

Figures in `outputs/`:
- `dataset_examples.png`: one sample image per class
- `training_curves.png`: training and validation accuracy and loss
- `confusion_matrix.png`: test-set confusion matrix
- `sample_predictions.png`: example predictions, coloured green when correct and red when wrong
- `model_comparison.png`: test accuracy of the CNN and the baselines
- `classification_report.txt`: per-class precision, recall and F1

## 4. How to run

```bash
pip install -r requirements.txt
cd src
python data.py          # dataset statistics and split sizes
python baseline.py      # HOG + LR / SVM baselines
python train.py         # trains GestureNet and saves models/gesturenet.keras
python evaluate.py      # test metrics and all figures
python predict.py ../data/Sign-Language-Digits-Dataset/Examples/example_3.JPG
python webcam_demo.py   # live demo (needs a webcam and opencv-python)
```

A trained model is already saved at `models/gesturenet.keras`, so you can run
`evaluate.py`, `predict.py` or `webcam_demo.py` without retraining.

## 5. Project structure

```
GestureNet/
├── data/Sign-Language-Digits-Dataset/   dataset (Apache 2.0)
├── models/gesturenet.keras              trained model
├── outputs/                             metrics, figures, training history
├── src/
│   ├── config.py        paths, hyperparameters, gesture → command map
│   ├── data.py          loading, preprocessing, stratified split
│   ├── model.py         GestureNet CNN and augmentation
│   ├── baseline.py      HOG + Logistic Regression / SVM
│   ├── train.py         training loop with callbacks
│   ├── evaluate.py      test metrics and plots
│   ├── predict.py       classify a single image
│   └── webcam_demo.py   real-time HCI demo with a stability filter
├── requirements.txt
└── README.md
```

## 6. Limitations and future work

- **Controlled backgrounds.** All images were taken against plain light
  backgrounds. Accuracy on the webcam will drop when the background is
  cluttered or the lighting is poor. Larger in-the-wild datasets such as HaGRID,
  or a hand detector such as MediaPipe Hands to crop the hand first, would help.
- **Possible participant overlap.** The file names do not identify participants,
  so the same person can appear in both the training and test sets. Accuracy on
  completely new users may therefore be somewhat lower than reported.
- **Static gestures only.** The model classifies single frames. Dynamic
  gestures such as swipes would need temporal models (3D CNNs or a CNN combined
  with an LSTM).
- **Small dataset.** Fine-tuning a pretrained backbone such as MobileNetV2 is a
  natural next step and usually gives better accuracy on new users.
