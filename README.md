# Addressing Class Imbalance Using Mahalanobis-Based Undersampling

This repository contains the official code and resources for the research article:

**"Addressing Class Imbalance in Credit Card Fraud Detection Using Mahalanobis-Based Undersampling of Extreme values"**  
Authors: Jorge Saavedra-Garrido, Daira Velandia, Cristian Ubal, Eyleen Spencer, Rodrigo Salas

---

## 📌 Description

Extreme class imbalance poses a significant challenge for machine learning models in credit card fraud detection, often leading to biased classifiers that overlook fraudulent transactions. This project introduces and evaluates two novel undersampling strategies based on the Mahalanobis distance: **MEUS** and **FEUS**. These methods are designed to generate more informative training sets by intelligently selecting samples, which enhances class separability and model performance while reducing computational costs.

## 🧪 Techniques Implemented

This repository provides implementations for the following class balancing techniques:

- **Random Undersampling (RS)**
- **NearMiss**
- **SMOTE (Synthetic Minority Over-sampling Technique)**

### Proposed Techniques

Our main contributions are two Mahalanobis distance-based undersampling methods:

1.  **MEUS (Majority-class Extreme-aware Undersampling)**
    This technique creates a balanced 1-to-1 dataset by matching each minority class sample with its nearest majority class neighbor in the Mahalanobis space. This preserves local data structure and focuses the model on learning the decision boundary effectively. The distance is calculated as:

    $$
    d_M(x_i, x_j) = \sqrt{(x_i - x_j)^T S^+ (x_i - x_j)}.
    $$

2.  **FEUS (Furthest-point Extreme UnderSampling)**
    This technique takes a global approach by selecting the most extreme samples (outliers) from the entire dataset, regardless of their class. It calculates the Mahalanobis distance of each point from the global data centroid, retaining only the most informative instances. This method excels at improving class separability. The distance is calculated as:

    $$
    d_M(x_i, \\bar{x}) = \sqrt{(x_i - \\bar{x})^T S^+ (x_i - \\bar{x})}.
    $$

## 📊 Evaluated Machine Learning Models

The performance of these techniques was evaluated on the following models:

- Logistic Regression (LR)
- Support Vector Machines (SVM)
- Random Forests (RF)
- XGBoost (XGB)
- Artificial Neural Networks (ANN)

All balancing techniques were correctly applied **after** the train-test split to ensure a robust and unbiased evaluation, avoiding data leakage.

## 📈 Key Findings

- **Superior Performance of FEUS:** The FEUS technique consistently emerged as the most effective strategy, achieving the best balance between precision and recall. It yielded the highest F1-Scores for SVM (93.87%), ANN (92.47%), and XGBoost (93.67%), making it ideal for real-world scenarios where minimizing false positives is critical.

- **Improved Data Separability:** The proposed Mahalanobis-based methods, especially FEUS, generate a more linearly separable feature space. This was confirmed during hyperparameter tuning, where models trained on FEUS data required significantly less regularization (e.g., higher `C` values in SVMs) compared to other techniques.

- **Precision vs. Recall Trade-off:** The study highlights a clear trade-off: SMOTE maximizes recall at the cost of extremely low precision, leading to a high number of false alarms. In contrast, FEUS maximizes precision, providing a more reliable and cost-effective solution for financial institutions.

- **Methodological Rigor:** Our results underscore the critical importance of applying resampling techniques *after* splitting the data into training and testing sets. This prevents data leakage and ensures that the model's performance is evaluated on a truly independent and realistic test set.

## 📂 Repository Structure

```
├── FEUS.ipynb             # Notebook implementing the FEUS technique and model evaluation.
├── MEUS.ipynb             # Notebook implementing the MEUS technique and model evaluation.
├── NEARMISS.ipynb         # Notebook for the NearMiss technique.
├── SMOTE.ipynb            # Notebook for the SMOTE technique.
├── UNDERSAMPLE.ipynb      # Notebook for the Random Undersampling (RS) technique.
├── VISUALITATION.ipynb    # Notebook for generating plots and visualizations from the paper.
├── TEST.ipynb             # Statistical tests.
└── README.md              # This file.
```

## 📄 Citation

If you use this code or our findings in your research, please cite our work.

*Note: This article is currently under review and has not yet been published.*

> Saavedra-Garrido, J.; Velandia, D.; Ubal, C.; Spencer, E.; Salas, R. "Addressing Class Imbalance in Credit Card Fraud Detection Using Mahalanobis-Based Undersampling of Extreme values". *Preprint available upon request*, 2025.

## 📥 Data

The dataset used in this study is publicly available on Kaggle:  
[Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud)
