<h1 align="center">
  Addressing Class Imbalance in Credit Card Fraud Detection<br>
  Using Mahalanobis-Based Undersampling of Extreme Observations
</h1>

<p align="center">
  <em>Source code accompanying the research paper submitted to Heliyon</em>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.10%2B-blue?logo=python&logoColor=white" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow&logoColor=white" alt="TensorFlow">
  <img src="https://img.shields.io/badge/scikit--learn-1.x-f7931e?logo=scikit-learn&logoColor=white" alt="scikit-learn">
  <img src="https://img.shields.io/badge/XGBoost-1.x-006600?logo=xgboost&logoColor=white" alt="XGBoost">
  <img src="https://img.shields.io/badge/imbalanced--learn-0.x-brightgreen" alt="imbalanced-learn">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License: MIT">
</p>

---

## 👥 Authors

**Jorge Saavedra-Garrido**<sup>a</sup>, **Daira Velandia**<sup>a,b,*</sup>, **Cristian Ubal**<sup>c</sup>, **Eyleen Spencer**<sup>d</sup>, **Keyla Canibilo Rojas**<sup>a</sup>, **Rodrigo Salas**<sup>d,e,f</sup>

<sup>a</sup> Statistical Institute, Faculty of Science, Universidad de Valparaíso, Valparaíso, Chile  
<sup>b</sup> Center for Atmospheric Studies and Climate Change (CEACC), Universidad de Valparaíso, Chile  
<sup>c</sup> Department of Data Science, Faculty of Engineering, Universidad de Playa Ancha, Valparaíso, Chile  
<sup>d</sup> Biomedical Engineering School, Faculty of Engineering, Universidad de Valparaíso, Chile  
<sup>e</sup> Center of Interdisciplinary Biomedical and Engineering Research for Health (MEDING), Universidad de Valparaíso, Chile  
<sup>f</sup> Millennium Institute for Intelligent Healthcare Engineering (iHealth), Santiago, Chile

<sup>*</sup> Corresponding author: daira.velandia@uv.cl

---

## 📋 Abstract

Severe class imbalance hinders the detection of card fraud, and applying resampling before partitioning the data inflates performance estimates due to information leakage. Mahalanobis distance has been used previously to detect outliers and in resampling procedures, but the behavior of two geometrically distinct selection rules, one local and one global, under the same metric for fraud detection has not been studied. In this manuscript, we propose and evaluate two subsampling strategies that retain only real transactions: **Majority-class Extreme-aware UnderSampling (MEUS)** matches each minority observation with a close majority candidate according to the Mahalanobis metric, and **Furthest-point Extreme UnderSampling (FEUS)** retains observations farther from the global centroid of the training set. Both use a global covariance matrix estimated from the training set, but apply distinct observation-selection rules. We compare them with five data-level resampling techniques on the European Credit Card Dataset (ECCD) and the IEEE-CIS Fraud Detection dataset using repeated stratified cross-validation, with resampling restricted to the training folds. Under severe imbalance, FEUS offered the most favorable compromise between precision and recall in ECCD and intermediate performance in IEEE-CIS, with low computational cost; MEUS was competitive in ECCD but lost precision in IEEE-CIS. A Euclidean ablation showed that incorporating covariance changes the balance between metrics without being uniformly superior. The effect of covariance undersampling depends on the classifier and the imbalance structure of the data, and we did not observe a method that dominated under all conditions.

**Keywords:** Credit Card Fraud · Class Imbalance · Extreme Observations · Mahalanobis Distance · Machine Learning · Data Leakage

---

## 🔬 Resampling Techniques Evaluated

### Proposed Methods

| Technique | Type | Script | Description |
|-----------|------|--------|-------------|
| **FEUS** | Under-sampling | `new_FEUS.py` | Computes the Mahalanobis distance of each training observation to the global centroid of the training fold and retains the *N<sub>keep</sub>* observations with the largest distances (*N<sub>keep</sub>* = 5000 in the paper). It does not enforce class balance; the resulting class proportion depends on the data geometry. |
| **MEUS** | Under-sampling | `new_MEUS.py` | Retains all minority observations and matches each one, without replacement, with the first unassigned majority observation among its *K* = 100 nearest Mahalanobis neighbors. The result is balanced when every minority observation is matched; otherwise the script logs a warning. |

### Baseline Methods

| Technique | Type | Script | Description |
|-----------|------|--------|-------------|
| **SMOTE** | Over-sampling | `new_SMOTE.py` | Synthetic Minority Over-sampling Technique — generates synthetic minority samples by interpolating between existing instances and their k-nearest neighbors. |
| **NearMiss** | Under-sampling | `new_NEARMISS.py` | Selects majority samples closest to minority instances using k-NN heuristics, focusing retention on decision-boundary regions. |
| **RS** | Under-sampling | `new_UNDERSAMPLE.py` | Random Subsampling — randomly discards majority-class samples until class balance is achieved. |
| **ENN** | Under-sampling | `new_ENN.py` | Edited Nearest Neighbours — removes samples whose class label disagrees with the majority of their *k* nearest neighbors (data cleaning). |
| **Tomek Links** | Under-sampling | `new_TOMEKLINKS.py` | Removes majority-class members of Tomek pairs (nearest-neighbor pairs from opposite classes), cleaning the decision boundary. |

### Additional Scripts

| Script | Description |
|--------|-------------|
| `FEUS_euclidean_mahalanobis.py` | Ablation study comparing **Euclidean vs. Mahalanobis** distance within FEUS on ECCD (*N<sub>keep</sub>* = 5000). |
| `TEST.py` | **Robustness analysis** for the ANN classifier — percentile bootstrap confidence intervals (2,000 resamples), Friedman test, Nemenyi post-hoc comparisons, heatmaps, boxplots, and precision-recall scatter plots. |
| `opentSNE.py` | **t-SNE visualization** of resampled data distributions using [openTSNE](https://opentsne.readthedocs.io/), generating publication-quality figures for both MEUS and FEUS. |

---

## 🤖 Classification Models

Each resampling technique is evaluated with six supervised classifiers, as described in the paper:

| Model | Abbreviation | Framework |
|-------|-------------|-----------|
| Logistic Regression | LR | scikit-learn |
| Support Vector Machine | SVM | scikit-learn |
| Random Forest | RF | scikit-learn |
| Extreme Gradient Boosting | XGBoost | XGBoost |
| Artificial Neural Network (multilayer perceptron) | ANN | TensorFlow / Keras |
| 1-Nearest Neighbor | 1-NN | scikit-learn |

Hyperparameters were tuned in a preliminary stage with stratified 4-fold cross-validation and then kept fixed in all simulations. They differ across resampling techniques and are defined in the `OPTIMAL_*_PARAMS` dictionaries at the top of each script (Supplementary Table S1 of the paper). XGBoost uses early stopping and the ANN uses a hold-out validation split, both drawn from the training fold.

---

## 📊 Evaluation Metrics

All experiments report the following metrics (macro-averaged where applicable):

| Metric | Description |
|--------|-------------|
| **Recall** (Sensitivity) | Ability to correctly identify fraudulent transactions |
| **Precision** | Reliability of positive predictions |
| **F1-Score** | Harmonic mean of precision and recall |
| **Accuracy** | Overall correctness; reported for completeness, with limited diagnostic value under strong imbalance |
| **ROC-AUC** | Threshold-independent overall discrimination capability |
| **PR-AUC** | Precision-Recall trade-off — especially relevant under extreme imbalance |

Class labels are obtained with a fixed decision threshold of 0.5. Fold-level metrics are averaged within each simulation, and results are summarized over the 30 simulation-level estimates with:
- Mean and 95% confidence interval based on the *t* distribution (main results)
- Percentile bootstrap 95% confidence intervals (2,000 resamples), Friedman test, and Nemenyi post-hoc comparisons (robustness analysis, ANN classifier)

---

## 🏗️ Experimental Design

```
┌────────────────────────────────────────────────────────────────┐
│                    FOR EACH SIMULATION (1..30)                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │              STRATIFIED 5-FOLD CROSS-VALIDATION          │  │
│  │  ┌────────────────────────────────────────────────────┐  │  │
│  │  │  FOR EACH FOLD (1..5)                              │  │  │
│  │  │                                                    │  │  │
│  │  │  1. Split → Train / Validation                     │  │  │
│  │  │  2. Scale (MinMaxScaler) — fit on Train ONLY       │  │  │
│  │  │  3. Resample Train ONLY (no data leakage)          │  │  │
│  │  │  4. Train classifiers on resampled Train           │  │  │
│  │  │  5. Evaluate on original (unmodified) Validation   │  │  │
│  │  └────────────────────────────────────────────────────┘  │  │
│  │  Average metrics across 5 folds                          │  │
│  └──────────────────────────────────────────────────────────┘  │
│  Record averaged metrics for this simulation                   │
└────────────────────────────────────────────────────────────────┘
        ↓
   Aggregate statistics over 30 simulations
   (Mean, Std, CI, Min, Max) + Friedman/Nemenyi tests
```

### Key Design Principles

- **No data leakage**: Scaling (`MinMaxScaler`) is fitted exclusively on the training fold; the validation fold is transformed using the same scaler but never participates in fitting. Resampling is applied **after** the train–test split.
- **Resampling isolation**: All resampling methods are applied strictly to the training fold. The validation fold remains unmodified, preserving its statistical independence.
- **Reproducibility**: Each simulation uses a deterministic seed (simulation index) for NumPy, TensorFlow, and all stochastic components across partitioning, shuffling, and model initialization.
- **Hardware acceleration**: Optional Intel Extension for scikit-learn (`sklearnex`) is integrated for transparent CPU acceleration on supported hardware.
- **Hyperparameter optimization**: Conducted via stratified 4-fold cross-validated GridSearchCV with ROC-AUC as the optimization metric (details in Supplementary Table S1 of the paper).

---

## 📂 Dataset

Two public datasets are used.

**European Credit Card Dataset (ECCD)** — [Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud) (Kaggle): **284,807** anonymized transactions made by European cardholders in September 2013.

- **Fraudulent transactions**: 492 (0.17%)
- **Legitimate transactions**: 284,315 (99.83%)
- **Imbalance Ratio (IR)**: 577.88
- **Features**: 28 PCA-transformed components (V1–V28) + Time + Amount
- **Target**: `Class` (1 = Fraud, 0 = Genuine)

**IEEE-CIS Fraud Detection** — [Kaggle competition data](https://www.kaggle.com/competitions/ieee-fraud-detection/data): **590,540** e-commerce transactions, of which 20,663 (3.5%) are fraudulent. The `transaction` and `identity` tables were merged by `TransactionID`, and a fully numeric depurated table was built as described in the paper (transaction amount, temporal and identity-related variables, C and D families, UID-based aggregate descriptors, and a subset of V variables, with median imputation). A balanced version was also built by keeping all fraudulent transactions and randomly sampling the same number of legitimate ones (20,663 per class; 41,326 in total). The scripts expect these tables as `ieee_ready_depurada.csv` and `ieee_balanced.csv`.

---

## ⚙️ Requirements

### Python Dependencies

```
python >= 3.10
numpy
pandas
scipy
scikit-learn
tensorflow >= 2.x
xgboost
imbalanced-learn
joblib
matplotlib
seaborn
scikit-posthocs       # for Nemenyi post-hoc test (TEST.py)
openTSNE              # for t-SNE visualization (opentSNE.py)
sklearnex             # optional: Intel Extension for scikit-learn
```

### Installation

```bash
pip install numpy pandas scipy scikit-learn tensorflow xgboost imbalanced-learn joblib matplotlib seaborn scikit-posthocs openTSNE
```

For Intel acceleration (optional):

```bash
pip install scikit-learn-intelex
```

---

## 📂 Repository Structure

```
Data-Balancing/
│
├── README.md                            # This file
├── LICENSE                              # MIT License
├── .gitignore                           # Git ignore rules
│
│── ── Proposed Methods ──────────────────
├── new_FEUS.py                          # FEUS — Furthest-point Extreme UnderSampling
├── new_MEUS.py                          # MEUS — Majority-class Extreme-aware UnderSampling
├── FEUS_euclidean_mahalanobis.py         # Ablation: Euclidean vs. Mahalanobis in FEUS
│
│── ── Baseline Methods ──────────────────
├── new_SMOTE.py                         # SMOTE (over-sampling baseline)
├── new_NEARMISS.py                      # NearMiss (under-sampling baseline)
├── new_UNDERSAMPLE.py                   # RS — Random Subsampling (baseline)
├── new_ENN.py                           # ENN — Edited Nearest Neighbours (baseline)
├── new_TOMEKLINKS.py                    # TL  — Tomek Links (baseline)
│
│── ── Analysis & Visualization ──────────
├── TEST.py                              # Statistical tests (Friedman/Nemenyi) & plots
└── opentSNE.py                          # t-SNE embedding visualization
```

---

## 🚀 Usage

### Running an Experiment

Each `new_*.py` script is a self-contained experiment. To run, for example, the FEUS experiment:

```bash
python new_FEUS.py
```

> **⚠️ Important:** Each `new_*.py` script contains three configuration blocks at the top (ECCD, IEEE-CIS depurated, IEEE-CIS balanced). Uncomment the one you want and update `CSV_PATH` and `BASE_OUTPUT` to point to your local data file and desired output directory.

### Running Statistical Analysis

After all experiments have completed and CSV summaries are generated:

```bash
python TEST.py
```

> **Note:** Update the CSV file paths inside `TEST.py` to match your output directories.

### Generating t-SNE Visualizations

```bash
python opentSNE.py
```

---

## 📄 Output Files

Each experiment generates:

| File | Content |
|------|---------|
| `ALL_MODELS_<TECHNIQUE>_summary.csv` | Per-simulation metrics for all classifiers |
| `Statistics_summary_<TECHNIQUE>.txt` | Aggregated statistics: mean, std, CI, min, max |
| `run_<experiment>.log` | Detailed execution log per simulation |

---

## 📖 Citation

If you use this code in your research, please cite our paper:

```bibtex
@article{saavedra2026mahalanobis,
  title     = {Addressing Class Imbalance in Credit Card Fraud Detection 
               Using Mahalanobis-Based Undersampling of Extreme Observations},
  author    = {Saavedra-Garrido, Jorge and Velandia, Daira and Ubal, Cristian and Spencer, Eyleen and Canibilo Rojas, Keyla and Salas, Rodrigo},
  journal   = {Heliyon},
  note      = {Under review},
  year      = {2026},
  keywords  = {Credit Card Fraud, Class Imbalance, Extreme Observations, 
               Mahalanobis Distance, Machine Learning, Data Leakage}
}
```

---

## 💰 Funding

This research was funded by:
- **FONDECYT** project N° 1221938
- **ANID Millennium Science Initiative Program** ICN2021_004
- **Beca de Doctorado Nacional** 21231546

The authors acknowledge the support given by the Center of Interdisciplinary Biomedical and Engineering Research for Health (MEDING), Universidad de Valparaíso, Chile.

---

## 📜 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

<p align="center">
  <em>Developed for reproducible machine learning research in imbalanced classification</em>
</p>
