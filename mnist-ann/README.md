# MNIST Handwritten Digit Classification with an ANN

A simple fully connected neural network (multi-layer perceptron) built with TensorFlow/Keras that classifies handwritten digits (0–9) from the MNIST dataset.

**Test accuracy: 97.40%**

## Overview

This project is a baseline for image classification. Each 28×28 grayscale image is flattened into a vector and passed through dense layers. It is also a useful reference point for the [MNIST CNN project](../MNIST_CNN), which uses convolutional layers on the same dataset.

## Dataset

- **MNIST**, loaded directly with `keras.datasets.mnist` (no manual download needed)
- 60,000 training images and 10,000 test images, 28×28 grayscale, 10 classes (digits 0–9)
- Pixel values are scaled to the range [0, 1]

## Model Architecture

| Layer | Details |
|-------|---------|
| Flatten | 28×28 → 784 |
| Dense | 128 units, ReLU |
| Dropout | 0.2 |
| Dense | 32 units, ReLU |
| Dropout | 0.2 |
| Dense | 10 units, Softmax |

Total parameters: **104,938** (all trainable)

## Training Setup

- **Loss:** sparse categorical cross-entropy
- **Optimizer:** Adam
- **Batch size:** 10
- **Validation:** 20% of the training set (`validation_split=0.2`)
- **Early stopping:** monitors `val_loss`, patience 5 (training stopped at epoch 10)

## Results

Evaluated on the 10,000-image test set:

| Metric | Value |
|--------|-------|
| Accuracy | 0.9740 |
| Precision (macro) | 0.9739 |
| Recall (macro) | 0.9735 |
| F1 score (macro) | 0.9736 |
| ROC-AUC (one-vs-rest) | 0.9994 |
| PR-AUC | 0.9960 |
| Matthews correlation coefficient | 0.9711 |

The notebook also plots training/validation loss and accuracy curves and a confusion matrix.

## Getting Started

### Requirements

- Python 3.9+
- tensorflow
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
cd MNIST_ANN
jupyter notebook MNIST_ANN.ipynb
```

Run the cells from top to bottom. MNIST is downloaded automatically on first run.

## Project Structure

```
MNIST_ANN/
├── MNIST_ANN.ipynb
└── README.md
```

## Possible Improvements

- Use a CNN, which typically reaches about 99%+ on MNIST (see the MNIST CNN project)
- Add data augmentation
- Use a larger batch size to speed up training (batch size 10 is slow)


