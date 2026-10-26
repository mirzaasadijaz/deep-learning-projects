# Cats vs Dogs Binary Classification with Transfer Learning

A binary image classifier built with TensorFlow/Keras that tells cats from dogs using **transfer learning** with a pretrained **MobileNetV2** network, trained in two phases: feature extraction, then fine-tuning.

**Validation accuracy: 98.44%**

## Overview

Instead of training a CNN from scratch, this project reuses MobileNetV2 (pretrained on ImageNet) as a feature extractor and trains a small classification head on top. The top 30 layers of the base network are then unfrozen and fine-tuned at a very low learning rate for a further accuracy gain.

## Dataset

- **Microsoft Cats vs Dogs** from Kaggle (`kagglecatsanddogs_3367a` / `PetImages`)
- 23,390 images used in total, in two classes (Cat = 0, Dog = 1)
- 80/20 split: 18,712 training images and 4,678 validation images (`seed=123`)
- Images are resized to 160×160
- A cleaning step removes corrupt or non-JFIF files (0 were found in this run)

### Getting the data

1. Download the Microsoft "Cats and Dogs" dataset (`kagglecatsanddogs_3367a`) from Kaggle
2. Extract it so the folder structure is:

```
datasets/
└── archive/
    └── kagglecatsanddogs_3367a/
        └── PetImages/
            ├── Cat/
            └── Dog/
```

The dataset is not included in this repository because of its size. If your folder layout differs, change `data_dir` at the top of the notebook.

## Model Architecture

| Stage | Details |
|-------|---------|
| Input | 160×160×3 |
| Augmentation | RandomFlip (horizontal), RandomRotation(0.1), RandomZoom(0.1) |
| Preprocessing | MobileNetV2 `preprocess_input` (scales to [-1, 1]) |
| Base model | MobileNetV2 (ImageNet weights, no top) |
| Head | GlobalAveragePooling → Dropout(0.3) → Dense(1, sigmoid) |

- Total parameters: 2,259,265, of which only **1,281** are trainable in phase 1
- Uses a single sigmoid output for binary classification

## Training Setup

**Phase 1: train the head (base frozen)**
- Adam, learning rate 1e-3, binary cross-entropy
- Up to 15 epochs with EarlyStopping (patience 5); stopped at epoch 12
- Validation accuracy reached about 98.0%

**Phase 2: fine-tuning**
- Last 30 layers of MobileNetV2 unfrozen
- Adam, learning rate 1e-5
- Trained for 10 more epochs (epochs 13–22)
- Best model saved with ModelCheckpoint (`best_cat_dog.keras`)

## Results

| Stage | Validation accuracy |
|-------|--------------------|
| After phase 1 (frozen base) | about 98.0% |
| After phase 2 (fine-tuned) | **98.44%** (val loss 0.0473) |

The notebook plots accuracy and loss across both phases.

## Getting Started

### Requirements

- Python 3.9+
- tensorflow
- matplotlib
- jupyter

```bash
pip install tensorflow matplotlib jupyter
```

### Run

```bash
git clone <your-repo-url>
cd cats_dogs_binary
jupyter notebook cats_dogs_binary.ipynb
```

Run the cells from top to bottom after placing the dataset as described above. Each epoch took about 4 to 5 minutes on CPU, so a GPU is recommended if available.

## Project Structure

```
cats_dogs_binary/
├── cats_dogs_binary.ipynb
├── datasets/            # not included: download from Kaggle
└── README.md
```

## Possible Improvements

- Add a test-set evaluation with a confusion matrix and precision/recall
- Try a larger backbone (EfficientNet, ResNet) or higher resolution
- Add a prediction cell for classifying your own photos
- Export the model for deployment (TFLite or a small web app)
