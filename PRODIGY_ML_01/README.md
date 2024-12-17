# Linear Regression Model for House Price Prediction

This repository contains the implementation of a **Linear Regression model** to predict house prices based on various features, including square footage, the number of bedrooms, and bathrooms.

## Index

1. [Project Overview](#project-overview)
2. [Features](#features)
3. [File Structure](#file-structure)
4. [Steps to Run](#steps-to-run)
   - [1. Data Cleaning](#1-data-cleaning)
   - [2. Model Training and Prediction](#2-model-training-and-prediction)
5. [Visualization](#visualization)
6. [Dependencies](#dependencies)
7. [Results](#results)
8. [Dataset](#dataset)
9. [License](#license)

## Project Overview
The project is a part of **Task 1** of the Prodigy InfoTech internship and includes:
- Preprocessing raw data for cleaning and feature preparation.
- Building a Linear Regression model enhanced with **Polynomial Features** to capture non-linear relationships.
- Evaluating the model's performance using metrics like **Mean Squared Error (MSE)** and **R-squared (R²)**.
- Visualizing model accuracy with **scatter plots**, **residual plots**, and **histograms**.
- Saving predictions to a well-structured CSV file.

## Features
- **Data Preprocessing**:
  - Handling missing values and outliers.
  - Log-transforming skewed data for better model performance.
  - Encoding categorical features and scaling numerical features.
- **Model Training**:
  - Using **Linear Regression** with Polynomial Features for non-linear relationships.
- **Visualization**:
  - Plots for evaluating model performance and residual errors.
- **Predictions**:
  - Final house price predictions saved in a CSV file with essential columns.

## File Structure
```
│   data_cleaning.py
│   PredictedPrice.csv
│   task1.py
│
├───cleaned_dataset
│       test_cleaned.csv
│       train_cleaned.csv
│
└───house-prices_dataset
        data_description.txt
        test.csv
        train.csv
```

## Steps to Run

### 1. Data Cleaning

Run the `data_cleaning.py` script to preprocess the raw datasets:

```bash
python data_cleaning.py
```

This will generate cleaned datasets (`train_cleaned.csv` and `test_cleaned.csv`) in the `cleaned_dataset` folder.

### 2. Model Training and Prediction

Run the `task1.py` script to train the Linear Regression model and generate predictions:

```bash
python task1.py
```

The predicted prices will be saved in `PredictedPrice.csv`.

## Visualization

The project includes visualizations to analyze model accuracy:

- **Scatter Plot**: Actual vs. Predicted values.
- **Residual Plot**: Residual errors.
- **Histogram**: Distribution of residuals.

These visualizations are automatically generated when running the `task1.py` script.

## Dependencies

The following Python libraries are required:

- `numpy`
- `pandas`
- `scikit-learn`
- `matplotlib`

Ensure these libraries are installed in your Python environment:

```bash
pip install numpy pandas scikit-learn matplotlib
```

## Results

- The model achieved an R-squared score of `0.XX` on the training data (replace with your result).
- The predictions are saved in `PredictedPrice.csv`, containing only essential columns (`Id` and `SalePrice`).

## Dataset
Kaggle: https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data.

The dataset is available in the `house-prices_dataset` folder, including:

- `train.csv`: Raw training data.
- `test.csv`: Raw test data.
- `data_description.txt`: Dataset documentation.

## License

This project is licensed under the MIT License.
