
# Class Balancing Techniques Using Mahalanobis-Based Undersampling

This repository contains the code and resources for the article:

**"Enhancing Machine Learning for Credit Card Fraud Detection Using Mahalanobis-Based Undersampling of Extreme Values"**  
Authors: Jorge Saavedra-Garrido, Daira Velandia, Cristian Ubal, Eyleen Spencer, Rodrigo Salas

## 📌 Description

Extreme class imbalance poses a significant challenge for credit card fraud detection. This repository implements and evaluates a novel undersampling strategy based on Mahalanobis distance, which selects extreme observations from both classes to improve class separability and reduce computational cost.

## 🧪 Included Techniques

- **Random Undersampling**
- **NearMiss**
- **SMOTE**
- **Mahalanobis Distance-Based Sampling:**
  - MEUS (Majority-class Extreme-aware Undersampling)
  - FEUS (Full-class Extreme-aware Undersampling)

## 📊 Evaluated Machine Learning Models

- Logistic Regression (LR)
- Support Vector Machines (SVM)
- Decision Trees (DT)
- Random Forests (RF)
- XGBoost
- Artificial Neural Networks (ANN)

## 🔍 Evaluation Scenarios

1. **Scenario 1**: Class balancing performed **before** the train/test split (introduces data leakage).
2. **Scenario 2**: Class balancing performed **after** the split (clean evaluation).

## 📈 Key Findings

FEUS consistently outperformed standard methods (e.g., SMOTE, NearMiss) achieving 97.8% precision, 91.8% F1-score, and 87.2% recall while preserving a realistic and unbiased test set.

## 📂 Repository Structure

```
├── data/                # Original or balanced datasets
├── scripts/             # Balancing techniques implementations
├── models/              # Trained models or configurations
├── results/             # Metrics and visualizations
└── README.md            # This file
```

## 📄 Citation

> Saavedra-Garrido, J.; Velandia, D.; Ubal, C.; Spencer, E.; Salas, R.  
> *Enhancing Machine Learning for Credit Card Fraud Detection Using Mahalanobis-Based Undersampling of Extreme Values*. Mathematics 2024.

## 📥 Data

Dataset available at Kaggle:  
https://www.kaggle.com/mlg-ulb/creditcardfraud
