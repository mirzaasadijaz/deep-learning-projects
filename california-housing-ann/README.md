# California Housing Price Prediction with a Tuned ANN

A regression project that predicts median house values in California using a neural network built with TensorFlow/Keras. The network architecture is tuned with Keras Tuner (Hyperband) and compared against classical baselines, with thorough diagnostics.

**Test R²: 0.7322** | **RMSE: 0.5924** | **MAE: 0.4162** (target is in units of $100,000)

## Overview

The notebook walks through a complete regression workflow:

1. Load and explore the data (health checks, correlation heatmap, target distribution)
2. Split the data and scale features (scaler fit on the training set only, to avoid leakage)
3. Train baseline models for comparison
4. Tune an ANN with Hyperband
5. Evaluate with a wide set of regression metrics
6. Run residual diagnostics and permutation feature importance

## Dataset

- **California Housing**, loaded with `sklearn.datasets.fetch_california_housing` (no manual download needed)
- 20,640 samples, 8 features, no missing values
- Features: `MedInc`, `HouseAge`, `AveRooms`, `AveBedrms`, `Population`, `AveOccup`, `Latitude`, `Longitude`
- Target: `MedHouseVal`, median house value in $100,000s (range 0.15 to 5.00). Values are capped at 5.0, which limits the achievable accuracy.
- 80/20 train/test split (16,512 / 4,128) with `random_state=42`

## Model

Hyperband searched over the number of layers (1–4), units per layer, activation (ReLU / Leaky ReLU / ELU), L2 strength, dropout rate and learning rate (90 trials, up to 60 epochs each).

Best configuration found:

| Setting | Value |
|---------|-------|
| Hidden layers | 2 (12 and 32 units) |
| Activation | ELU |
| L2 regularization | about 5.9e-6 |
| Dropout | 0.0 |
| Learning rate | about 0.0033 |

Each hidden layer is followed by batch normalization, and the output is a single linear unit. The final model was trained with Adam, MSE loss, batch size 10, and EarlyStopping / ReduceLROnPlateau / ModelCheckpoint callbacks (best weights from epoch 5, saved as `best_ann_model.keras`).

## Results

Model comparison on the test set:

| Model | R² | RMSE | MAE |
|-------|-----|------|-----|
| Random Forest | 0.8064 | 0.5036 | 0.3267 |
| **Tuned ANN** | **0.7322** | **0.5924** | **0.4162** |
| Linear Regression | 0.5758 | 0.7456 | 0.5332 |
| Mean baseline | -0.0002 | 1.1449 | 0.9061 |

Additional metrics for the ANN: adjusted R² 0.7317, MAPE 24.43%, median absolute error 0.2969, Pearson r 0.859.

**Key takeaways**

- The tuned ANN clearly beats linear regression, but the **Random Forest is still better** on this tabular dataset. This is common for small tabular data.
- Residuals are right-skewed (skewness 1.12) with heavy tails, so the model makes some large errors, partly due to the capped target values.
- Permutation importance shows that **Latitude, Longitude and MedInc** matter most, meaning location and income drive the predictions. `Population` matters least.

## Getting Started

### Requirements

- Python 3.9+
- tensorflow
- keras-tuner
- scikit-learn
- pandas, numpy, scipy
- matplotlib, seaborn
- jupyter

```bash
pip install tensorflow keras-tuner scikit-learn pandas numpy scipy matplotlib seaborn jupyter
```

### Run

```bash
git clone <your-repo-url>
cd housing_ANN_improved
jupyter notebook housing_ANN_improved.ipynb
```

Run the cells from top to bottom. The dataset downloads automatically. The Hyperband search creates a `kt_dir/` folder; you can add it (and `*.keras` files) to `.gitignore`.

## Project Structure

```
housing_ANN_improved/
├── housing_ANN_improved.ipynb
└── README.md
```

## Notes and Possible Improvements

- The test set is also used as the validation set during tuning and early stopping, so the reported test metrics may be slightly optimistic. A separate validation split would give a cleaner estimate.
- Try gradient boosting (XGBoost / LightGBM), which usually does very well on this dataset.
- Engineer features such as rooms per household or distance to major cities.
