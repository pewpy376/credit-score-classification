# Credit Score Classification & Risk Predictive Modeling

End-to-end machine learning pipeline for multi-class credit risk scoring, featuring leakage-aware grouped cross-validation, comprehensive preprocessing pipelines, model benchmarking, interpretability analysis, and model persistence.

---

## 📌 Project Overview

In credit scoring, errors carry asymmetric financial costs: misclassifying a high-risk borrower as creditworthy leads directly to loan default and severe losses. This project builds a reliable, production-ready classification system to assess customer creditworthiness into three tiers:
- **Poor** (High Risk)
- **Standard** (Moderate Risk)
- **Good** (Low Risk)

### Dataset Key Characteristics
- **Source:** Kaggle Credit Score Classification dataset (`ayushsharma0812/dataset-for-credit-score-classification`).
- **Scale:** 100,000 monthly observations tracking 12,500 unique customers across 8 consecutive months.
- **Features:** 28 raw predictors spanning demographics, income, loan obligations, debt ratios, credit inquiries, and repayment behaviors.
- **Target Distribution:**
  - `Standard`: ~53.2%
  - `Poor`: ~28.0%
  - `Good`: ~18.8%

---

## 🔬 Key Engineering & Modeling Highlights

### 1. Leakage-Aware Validation Strategy
Because each customer is tracked monthly across 8 records, standard random splits would cause severe **data leakage** (training and testing on different months of the same individual). 
- All train/test and cross-validation splits utilize **`StratifiedGroupKFold`** grouped strictly by `customer_id`.
- **Holdout Test Set:** 2,500 unseen customers (20,000 records).
- **Development Set:** 10,000 customers (80,000 records) evaluated via 4-fold grouped cross-validation.
- **Customer overlap between Dev and Test:** Exactly `0`.

### 2. Robust Preprocessing Pipeline (`scikit-learn ColumnTransformer`)
To prevent data snooping, all data transformations and imputations are strictly learned on training folds only:
- **Anomaly Handling:** Implausible negative values in `total_emi_per_month` (0.024% of data) are converted to `NaN` via a custom transformer and imputed inside the pipeline.
- **Skewed Numerical Features:** Yeo-Johnson power transformations (`PowerTransformer`) applied to highly skewed ratio variables (`annual_income`, `monthly_inhand_salary`, `outstanding_debt`, `monthly_balance`, etc.).
- **Ordinal Variables:** Custom ordinal encodings preserving domain order for `credit_mix` (`Bad` < `Standard` < `Good`), `payment_level`, and `payment_volume`.
- **Nominal Features:** One-Hot Encoding (`handle_unknown='ignore'`) for occupations and specific loan types.
- **Feature Engineering:**
  - `debt_to_income`: $\frac{\text{total\_emi\_per\_month}}{\text{monthly\_inhand\_salary}}$
  - `saving_ratio`: $\frac{\text{monthly\_balance}}{\text{monthly\_inhand\_salary}}$
  - Aggregated loan counts by loan category (`auto_loan`, `debt_consolidation_loan`, etc.).

---

## 📊 Model Benchmarking & Selection

Six models were systematically evaluated across 4-fold `StratifiedGroupKFold` on the development set. Because `Standard` dominates ~53% of the distribution, **Macro F1** and **Macro Recall** were designated as primary selection metrics over raw accuracy to ensure strong sensitivity on `Poor` and `Good` classes.

| Model | Accuracy (mean ± std) | Macro F1 (mean ± std) | Macro Precision | Macro Recall | Neg Log-Loss |
| :--- | :---: | :---: | :---: | :---: | :---: |
| 🏆 **LightGBM** | **0.688 ± 0.003** | **0.680 ± 0.003** | **0.669** | **0.722** | **0.723** |
| **Random Forest** | 0.681 ± 0.002 | 0.674 ± 0.002 | 0.667 | 0.724 | 0.760 |
| **XGBoost** | 0.695 ± 0.002 | 0.670 ± 0.005 | 0.670 | 0.670 | 0.698 |
| **Extra Trees** | 0.670 ± 0.005 | 0.664 ± 0.004 | 0.658 | 0.715 | 0.784 |
| **Logistic Regression** | 0.655 ± 0.004 | 0.649 ± 0.004 | 0.643 | 0.699 | 0.820 |
| **Dummy (Most Frequent)** | 0.532 ± 0.000 | 0.231 ± 0.000 | 0.177 | 0.333 | 16.878 |

---

## 🎯 Final Champion Model: LightGBM

The final selected model is **LightGBM Classifier** with `class_weight='balanced'`.

### Holdout Test Set Performance (20,000 Unseen Records)
- **Accuracy:** `0.69`
- **Macro F1 Score:** `0.68`
- **Macro Recall:** `0.72`
- **Multi-Class ROC-AUC (OvR):** `0.8459`

#### Detailed Classification Report:
```
              precision    recall  f1-score   support

        Poor       0.67      0.74      0.70      5798
    Standard       0.83      0.62      0.71     10635
        Good       0.52      0.82      0.63      3567

    accuracy                           0.69     20000
   macro avg       0.67      0.72      0.68     20000
weighted avg       0.73      0.69      0.69     20000
```
> **Observation:** Test metrics match validation CV results (`0.688` accuracy, `0.680` macro F1), confirming zero data leakage and excellent generalization to completely new customers.

---

## 🔍 Feature Importance (Total Gain)

Aggregated total gain across LightGBM splits reveals the primary financial indicators driving creditworthiness:

| Rank | Feature | Share of Total Gain (%) | Cumulative (%) |
| :---: | :--- | :---: | :---: |
| 1 | `outstanding_debt` | **30.0%** | 30.0% |
| 2 | `credit_mix` | **28.9%** | 58.9% |
| 3 | `interest_rate` | **7.6%** | 66.5% |
| 4 | `num_credit_card` | **4.0%** | 70.5% |
| 5 | `changed_credit_limit` | **3.0%** | **73.5%** |
| 6 | `delay_from_due_date` | 2.5% | 76.0% |
| 7 | `credit_history_age` | 2.4% | 78.4% |
| 8 | `num_bank_accounts` | 2.3% | 80.7% |
| 9 | `debt_to_income` | 2.2% | 82.9% |
| 10 | `annual_income` | 2.2% | **85.1%** |

- **Top 2 features** (`outstanding_debt` and `credit_mix`) drive nearly **59%** of model split decisions.
- Engineered ratio **`debt_to_income`** ranks 9th (2.2%), outperforming raw monthly in-hand salary (1.6%).
- Demographics (`age`, `occupation`) contribute marginally (< 2.2% each), confirming findings from EDA.

---

## 📁 Repository Structure

```plaintext
Credit-risk-predictive-modeling/
│
├── 01_eda.ipynb              # EDA, data dictionary, anomaly detection & data cleaning
├── 02_model_pipeline.ipynb    # Feature engineering, leakage-free pipelines, benchmark & export
├── models/
│   └── lightgbm_credit_score_pipeline.joblib  # Serialized production pipeline
├── Dataset/                  # Data directory (ignored in git)
│   ├── Credit_score_cleaned_data.csv          # Raw dataset
│   └── my_data.csv                            # Cleaned data with engineered features
├── README.md                 # Project documentation
└── .gitignore                # Ignored files & datasets
```

---

## 🚀 Quickstart & Inference

### 1. Requirements
Ensure Python 3.10+ is installed with the required libraries:
```bash
pip install numpy pandas scikit-learn lightgbm xgboost matplotlib seaborn joblib
```

### 2. Loading the Pipeline for Inference
The exported pipeline contains the full end-to-end preprocessing, feature encoders, scalers, and classifier:

```python
import joblib
import pandas as pd
import numpy as np

# 1. Define custom transformer function used in preprocessing
def replace_negative_emi(X):
    return np.where(X < 0, np.nan, X)

# 2. Load the trained pipeline
pipeline = joblib.load("models/lightgbm_credit_score_pipeline.joblib")

# 3. Predict directly on raw incoming customer data
# target mapping: 0 = 'Poor', 1 = 'Standard', 2 = 'Good'
label_map = {0: "Poor", 1: "Standard", 2: "Good"}
predictions = pipeline.predict(raw_customer_df)
predicted_labels = [label_map[p] for p in predictions]
```

---

## 📝 License & Attribution
- Dataset provided by [Ayush Sharma on Kaggle](https://www.kaggle.com/datasets/ayushsharma0812/dataset-for-credit-score-classification).
- Developed as an end-to-end credit risk modeling reference architecture.
