# CIFAR-10 Image Classification with a CNN

A convolutional neural network built with TensorFlow/Keras that classifies 32×32 colour images into 10 object categories from the CIFAR-10 dataset, using data augmentation, batch normalization, dropout and L2 regularization.

**Test accuracy: 78.91%** | **Top-3 accuracy: 95.28%** | **Macro ROC-AUC: 0.978**

## Overview

The notebook trains a compact CNN (about 104K parameters) from scratch, evaluates it in depth, and includes cells for predicting your own cat or dog photo.

## Dataset

- **CIFAR-10**, loaded directly with `tensorflow.keras.datasets.cifar10` (no manual download needed)
- 50,000 training images and 10,000 test images, 32×32 RGB
- Classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
- Pixel values are scaled to [0, 1]

## Model Architecture

| Stage | Layers |
|-------|--------|
| Augmentation | RandomFlip (horizontal), RandomRotation(0.05), RandomZoom(0.1), RandomContrast(0.1) |
| Block 1 | Conv2D(32, 3×3) + BatchNorm → MaxPool → BatchNorm → Dropout(0.2) |
| Block 2 | Conv2D(64, 3×3) + BatchNorm → MaxPool → BatchNorm → Dropout(0.2) |
| Block 3 | Conv2D(128, 3×3) + BatchNorm → MaxPool → BatchNorm |
| Head | GlobalAveragePooling → Dense(64) + BatchNorm → Dropout(0.3) → Dense(10, logits) |

- Activation: Leaky ReLU with He-normal initialization
- L2 regularization (1e-4) on all kernels
- Augmentation layers are only active during training; Keras turns them off automatically for evaluation and prediction
- Total parameters: **104,202**

## Training Setup

- **Loss:** sparse categorical cross-entropy (from logits)
- **Optimizer:** Adam
- **Batch size:** 64
- **Epochs:** up to 1000, controlled by callbacks
- **Callbacks:**
  - `EarlyStopping` on `val_loss` (patience 10, best weights restored)
  - `ReduceLROnPlateau` (factor 0.5, patience 4)
  - `ModelCheckpoint` (saves `best_model.keras`)
- Training stopped at epoch 70, with the best weights from epoch 60

## Results

Evaluated on the 10,000-image test set:

| Metric | Value |
|--------|-------|
| Accuracy | 0.7891 |
| Macro precision / recall / F1 | 0.795 / 0.789 / 0.784 |
| Top-3 accuracy | 0.9528 |
| Macro ROC-AUC (one-vs-rest) | 0.9779 |

Per-class notes: the model does best on ships and trucks (F1 about 0.88). The weakest recall is on dogs (0.593), which are often confused with other animals, while frogs have the lowest precision (0.635). The notebook includes the full classification report and a confusion matrix.

## Predicting Your Own Images

The final cells load a custom image, resize it to 32×32, and print the predicted class:

1. Create a `datasets/` folder next to the notebook
2. Add your images (e.g. `datasets/dog.jpg`, `datasets/cat.jpg`)
3. Run the prediction cells

Because CIFAR-10 images are only 32×32, large photos lose a lot of detail when resized, so predictions on real-world photos can be less reliable.

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
cd CNN_image_classification
jupyter notebook CNN_image_classification.ipynb
```

Run the cells from top to bottom. CIFAR-10 is downloaded automatically on first run. Training took roughly one minute per epoch on CPU.

## Project Structure

```
CNN_image_classification/
├── CNN_image_classification.ipynb
├── datasets/            # optional: your own test images
└── README.md
```

## Notes and Possible Improvements

- The test set is also used as the validation set for early stopping and checkpointing, so the reported test accuracy may be slightly optimistic. For a stricter evaluation, hold out a separate validation split from the training data.
- Try a deeper network (e.g. ResNet-style blocks) or transfer learning for higher accuracy.
- Add more augmentation techniques such as CutMix or MixUp.
