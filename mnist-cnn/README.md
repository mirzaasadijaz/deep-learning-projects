# MNIST Handwritten Digit Classification with a CNN

A deep convolutional neural network built with TensorFlow/Keras that classifies handwritten digits (0–9) from the MNIST dataset, using modern training techniques such as batch normalization, label smoothing, AdamW and cosine learning-rate decay.

**Test accuracy: 99.25%**

## Overview

This project improves on the [MNIST ANN baseline](../MNIST_ANN) (97.40%) by using convolutional layers that learn spatial features directly from the images. It also includes a section for predicting your own handwritten digit image.

## Dataset

- **MNIST**, loaded directly with `keras.datasets.mnist` (no manual download needed)
- 60,000 training images and 10,000 test images, 28×28 grayscale, 10 classes
- Pixel values are scaled to [0, 1]

## Model Architecture

Three convolutional blocks followed by a small classifier head:

| Block | Layers |
|-------|--------|
| Augmentation | RandomRotation(0.05), RandomZoom(0.1) |
| Block 1 | 2 × Conv2D(32, 3×3) + BatchNorm → MaxPool → Dropout(0.2) |
| Block 2 | 2 × Conv2D(64, 3×3) + BatchNorm → MaxPool → Dropout(0.3) |
| Block 3 | 2 × Conv2D(128, 3×3) + BatchNorm → MaxPool → Dropout(0.4) |
| Head | Flatten → Dense(128) + BatchNorm → Dropout(0.5) → Dense(10, logits) |

- Activation: Leaky ReLU with He-normal initialization
- L2 regularization (1e-4) on all convolutional and dense kernels
- Horizontal flip is deliberately **not** used, since flipping digits changes their meaning
- Total parameters: **437,610** (436,458 trainable)

## Training Setup

- **Loss:** categorical cross-entropy (from logits) with label smoothing of 0.1
- **Optimizer:** AdamW (weight decay 1e-4) with cosine-decay learning rate starting at 1e-3
- **Batch size:** 32
- **Validation:** 15% of the training set
- **Callbacks:** EarlyStopping on `val_loss` (best weights restored) and ModelCheckpoint (saves `best_model_mnist.keras`)
- Training stopped early after epoch 6; the best epoch was 5 (validation accuracy 99.33%)

Note: the loss values look high (about 0.6) because label smoothing raises the minimum achievable loss. Use accuracy for comparison rather than the raw loss.

## Results

Evaluated on the 10,000-image test set:

| Metric | Value |
|--------|-------|
| Accuracy | 0.9925 |
| Precision / Recall / F1 (macro) | 0.9925 each |

The notebook also reports ROC-AUC, PR-AUC, MCC, a full per-class classification report, a confusion matrix, and ROC / precision-recall curves.

## Predicting Your Own Digit

The last cells load the saved model (`best_model_mnist.keras`) and classify a custom image:

1. Place your image next to the notebook (e.g. `download.jpg`)
2. Set `image_path` to its filename
3. Run the cell to get the predicted digit and confidence

For best results, match the MNIST format: a **white digit on a black background**, centered in the frame. Images that don't match this format can give low-confidence or wrong predictions.

## Getting Started

### Requirements

- Python 3.9+
- tensorflow (2.16+ recommended)
- numpy
- matplotlib
- scikit-learn
- jupyter

```bash
pip install tensorflow numpy matplotlib scikit-learn jupyter
```

### Run

```bash
git clone <your-repo-url>
cd MNIST_CNN
jupyter notebook MNIST_CNN.ipynb
```

Run the cells from top to bottom. A GPU is helpful but not required (training took about 80 seconds per epoch on CPU).

## Project Structure

```
MNIST_CNN/
├── MNIST_CNN.ipynb
└── README.md
```

## Possible Improvements

- Train for more epochs with a higher early-stopping patience (the current patience is 1)
- Ensemble several models
- Add stronger augmentation such as small shifts and elastic distortions
