# Tumor-Identification-with-Logistic-Regression

A machine learning pipeline for binary classification of tumor vs. normal tissue samples using gene expression microarray data from the GEO database (GSE15852).

## Overview

This project demonstrates a complete ML workflow for cancer classification:
- Downloads and processes gene expression data from NCBI GEO
- Applies dimensionality reduction using PCA
- Trains and compares multiple classification models
- Performs hyperparameter tuning
- Provides comprehensive model evaluation and visualization

## Installation & Setup

### Step 1: Clone or Download the Script
```bash
git clone <repository-url>
cd <repository-name>
```

### Step 2: Create a Virtual Environment (Recommended)
```bash
# Create virtual environment
python -m venv venv

# Activate it
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install numpy pandas matplotlib scikit-learn GEOparse
```

### Step 4: Run the Script
```bash
python model.py
```

**What happens when you run it:**
1. Downloads GSE15852 dataset from NCBI GEO (~100MB, first run only)
2. Processes and labels the data
3. Trains all models with progress indicators
4. Displays performance metrics in console
5. Opens multiple visualization windows
6. Outputs top predictive features

**Expected runtime**: 2-5 minutes (longer on first run due to download)

### Troubleshooting

**Issue**: `ModuleNotFoundError: No module named 'GEOparse'`  
**Solution**: Make sure you've installed all dependencies: `pip install GEOparse`

**Issue**: Download fails or times out  
**Solution**: Check internet connection. The GEO database may occasionally be slow. Try running again.

## Dataset

**GEO Series**: GSE15852  
- Microarray gene expression data
- Binary classification: tumor vs. normal tissue
- Automatically downloaded via GEOparse

## Pipeline Steps

### 1. Data Loading & Preprocessing
- Downloads GSE15852 from NCBI GEO database
- Parses sample labels from metadata
- Extracts tumor/normal classifications
- Log2 transformation of expression values
- Removes empty features

### 2. Train/Test Split
- 75/25 stratified split
- Random state: 42 for reproducibility

### 3. Feature Engineering
- **Standardization**: Zero mean, unit variance scaling
- **PCA**: Dimensionality reduction retaining 95% variance
- Visualizes cumulative proportion of variance explained
- Transforms both training and test sets consistently

### 4. Model Training

Three classifiers are trained and compared:

| Model | Description |
|-------|-------------|
| **Logistic Regression** | L2-regularized, balanced class weights |
| **Linear SVM (Calibrated)** | LinearSVC with isotonic calibration for probability estimates |
| **Random Forest** | Ensemble of 300 decision trees |

### 5. Evaluation Metrics

For each model:
- **Accuracy**: Overall classification accuracy
- **ROC-AUC**: Area under the ROC curve
- **PR-AUC**: Area under the Precision-Recall curve
- **CV-AUC**: 5-fold cross-validation AUC on training set
- **Confusion Matrix**: Visual representation of predictions

### 6. Hyperparameter Tuning

**Logistic Regression GridSearchCV**:
- C values: [0.01, 0.1, 1, 3, 10]
- 5-fold cross-validation
- Scoring: ROC-AUC

**Random Forest GridSearchCV**:
- n_estimators: [100, 300, 600]
- max_depth: [None, 10, 20]
- 5-fold cross-validation
- Scoring: ROC-AUC

### 7. Visualizations

The pipeline generates:
- **PCA Variance Plot**: Cumulative explained variance by principal components
- **Confusion Matrices**: One per model (blue heatmap with adaptive text color)
- **Model Comparison Bar Chart**: Accuracy and ROC-AUC side-by-side
- **ROC Curves**: Test set performance for tuned models
- **Precision-Recall Curves**: Test set performance with baseline

### 8. Interpretability

Extracts and displays the top 15 most important principal components from the best Logistic Regression model, showing coefficient magnitudes for tumor prediction.
