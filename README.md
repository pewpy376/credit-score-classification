# Credit Score Classification & Risk Modeling

An end-to-end machine learning pipeline that classifies customers into three credit tiers (**Poor**, **Standard**, **Good**). It uses leakage-aware validation, a reproducible preprocessing pipeline, a six-model benchmark, feature-importance analysis, and a serialized model ready for inference.

---

## 1. Business Context

In credit scoring, mistakes have unequal costs. Approving a high-risk borrower can lead to default and direct financial loss. Rejecting a good borrower means lost revenue. This project builds a classifier that supports credit-risk decisions with a focus on **catching the risky and the reliable customers, not just the average one**.

### Dataset

| Item | Detail |
| :--- | :--- |
| **Source** | [Kaggle: Credit Score Classification](https://www.kaggle.com/datasets/ayushsharma0812/dataset-for-credit-score-classification) |
| **Size** | 100,000 monthly records, 12,500 unique customers, 8 consecutive months each |
| **Predictors** | 28 raw features: demographics, income, loans, debt ratios, credit inquiries, repayment behavior |
| **Target** | `Standard` ≈ 53.2%, `Poor` ≈ 28.0%, `Good` ≈ 18.8% (imbalanced) |

---

## 2. Results at a Glance

Final model: **LightGBM** (`class_weight='balanced'`), evaluated on 20,000 records from 2,500 customers the model has never seen.

| Metric | Holdout score |
| :--- | :---: |
| Accuracy | 0.69 |
| Macro F1 | 0.68 |
| Macro Recall | 0.72 |
| ROC-AUC (OvR) | 0.8459 |

```
              precision    recall  f1-score   support

        Poor       0.67      0.74      0.70      5798
    Standard       0.83      0.62      0.71     10635
        Good       0.52      0.82      0.63      3567

    accuracy                           0.69     20000
   macro avg       0.67      0.72      0.68     20000
weighted avg       0.73      0.69      0.69     20000
```

**How to read this:**
- The model finds **74% of Poor** and **82% of Good** customers, which is the priority in risk screening.
- The cost: only **52% of customers predicted as Good are truly Good**, and **38% of Standard customers are missed** (recall 0.62). Most of these are pushed into Poor or Good.
- Holdout scores (accuracy 0.69, macro F1 0.68) closely match cross-validation (0.688, 0.680). This indicates the model generalizes to new customers and that the validation design is sound.

---

## 3. Methodology

### 3.1 Leakage-Aware Validation

Each customer appears in 8 monthly records. A normal random split would put some months in training and other months of the same person in testing, which inflates scores. To prevent this:

- All splits use **`StratifiedGroupKFold` grouped by `customer_id`**.
- **Development set:** 10,000 customers (80,000 records), evaluated with 4-fold grouped CV.
- **Holdout test set:** 2,500 customers (20,000 records).
- **Customer overlap between development and test sets: 0.**

### 3.2 Preprocessing Pipeline

Built with scikit-learn `ColumnTransformer`. Every transformation is fitted **on training folds only**, so nothing is learned from validation or test data.

| Step | What it does | Why |
| :--- | :--- | :--- |
| **Anomaly handling** | Negative `total_emi_per_month` values (0.024% of rows) become `NaN`, then are imputed | A repayment cannot be negative, so these are data errors |
| **Yeo-Johnson transform** | Applied to skewed numeric features (`annual_income`, `monthly_inhand_salary`, `outstanding_debt`, `monthly_balance`, etc.) | Reduces the effect of extreme values |
| **Ordinal encoding** | `credit_mix` (Bad < Standard < Good), `payment_level`, `payment_volume` | Keeps the natural order of the categories |
| **One-Hot Encoding** | Occupations and loan types, with `handle_unknown='ignore'` | Unseen categories at inference do not crash the pipeline |

### 3.3 Feature Engineering

| Feature | Formula | Intuition |
| :--- | :--- | :--- |
| `debt_to_income` | `total_emi_per_month / monthly_inhand_salary` | Share of income already committed to loan repayments |
| `saving_ratio` | `monthly_balance / monthly_inhand_salary` | How much of income remains as savings |
| Loan-type counts | Count per category (`auto_loan`, `debt_consolidation_loan`, ...) | Shows the mix of borrowing |

---

## 4. Model Benchmark & Selection

Six models were compared with 4-fold grouped CV on the development set. Because `Standard` is ~53% of the data, **Macro F1** and **Macro Recall** were the primary selection metrics rather than accuracy.

| Model | Accuracy | Macro F1 | Macro Precision | Macro Recall | Log-loss ↓ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **LightGBM (selected)** | 0.688 ± 0.003 | **0.680 ± 0.003** | 0.669 | 0.722 | **0.723** |
| Random Forest | 0.681 ± 0.002 | 0.674 ± 0.002 | 0.667 | **0.724** | 0.760 |
| XGBoost | **0.695 ± 0.002** | 0.670 ± 0.005 | **0.670** | 0.670 | 0.698\* |
| Extra Trees | 0.670 ± 0.005 | 0.664 ± 0.004 | 0.658 | 0.715 | 0.784 |
| Logistic Regression | 0.655 ± 0.004 | 0.649 ± 0.004 | 0.643 | 0.699 | 0.820 |
| Dummy (most frequent) | 0.532 ± 0.000 | 0.231 ± 0.000 | 0.177 | 0.333 | 16.878 |

\* Values are taken from the experiment log. Log-loss for XGBoost is lower than LightGBM's in that log, so LightGBM was selected on **Macro F1 and Macro Recall**, not on log-loss.

### Why LightGBM? Trade-offs

| Option | Advantage | Drawback | Decision |
| :--- | :--- | :--- | :--- |
| **LightGBM** | Best Macro F1, strong recall on minority classes, fast | Needs `class_weight` to handle imbalance; less transparent than a linear model | **Selected** |
| XGBoost | Highest accuracy, lowest log-loss | Macro Recall only 0.670: it favors the majority class `Standard` | Rejected: weaker on `Poor`/`Good` |
| Random Forest | Similar recall, simple to tune | Lower Macro F1, higher log-loss, larger model | Runner-up |
| Logistic Regression | Easy to explain and audit | Lowest accuracy among real models (0.655) | Good baseline, not enough alone |

---

## 5. Feature Importance

Share of total gain across LightGBM splits:

| Rank | Feature | Share | Cumulative |
| :---: | :--- | :---: | :---: |
| 1 | `outstanding_debt` | 30.0% | 30.0% |
| 2 | `credit_mix` | 28.9% | 58.9% |
| 3 | `interest_rate` | 7.6% | 66.5% |
| 4 | `num_credit_card` | 4.0% | 70.5% |
| 5 | `changed_credit_limit` | 3.0% | 73.5% |
| 6 | `delay_from_due_date` | 2.5% | 76.0% |
| 7 | `credit_history_age` | 2.4% | 78.4% |
| 8 | `num_bank_accounts` | 2.3% | 80.7% |
| 9 | `debt_to_income` | 2.2% | 82.9% |
| 10 | `annual_income` | 2.2% | 85.1% |

**Takeaways**
- Two features, `outstanding_debt` and `credit_mix`, account for ~59% of the model's decision gain.
- The engineered `debt_to_income` (rank 9, 2.2%) is more informative than raw monthly in-hand salary (1.6%).
- Demographics such as `age` and `occupation` contribute little (each under 2.2%), consistent with the EDA.

---

## 6. Limitations & Next Steps

| Area | Observation | Suggested action |
| :--- | :--- | :--- |
| **Precision on `Good`** | 0.52: about half of "Good" predictions are wrong | Tune decision thresholds, or use probabilities rather than hard labels |
| **Recall on `Standard`** | 0.62: many moderate-risk customers are misclassified | Review costs per error type and choose thresholds to match business policy |
| **Dominant features** | `credit_mix` is a strong, possibly bureau-derived signal; the model leans heavily on it | Check whether it is available at real decision time; test performance without it |
| **Probability quality** | Class weighting shifts predicted probabilities | Apply calibration (e.g. isotonic) before using scores as risk probabilities |
| **Data source** | Public Kaggle dataset, which may not reflect a real lender's population | Re-validate on production data before any real use |
| **Fairness** | No bias audit has been performed | Test performance across demographic groups before deployment |

---

## 7. Repository Structure

```plaintext
Credit-risk-predictive-modeling/
├── 01_eda.ipynb                 # EDA, data dictionary, anomaly detection, cleaning
├── 02_model_pipeline.ipynb      # Feature engineering, pipelines, benchmark, export
├── models/
│   └── lightgbm_credit_score_pipeline.joblib   # Serialized production pipeline
├── Dataset/                     # Data (git-ignored)
│   ├── Credit_score_cleaned_data.csv           # Raw dataset
│   └── my_data.csv                             # Cleaned data + engineered features
├── README.md
└── .gitignore
```

---

## 8. Quickstart

### Requirements

Python 3.10+

```bash
pip install numpy pandas scikit-learn lightgbm xgboost matplotlib seaborn joblib
```

### Run Inference

The saved pipeline already contains preprocessing, encoders, scalers, and the classifier, so you can feed it raw customer data directly.

```python
import joblib
import numpy as np
import pandas as pd

# Custom function used inside the pipeline; it must be defined before loading
def replace_negative_emi(X):
    return np.where(X < 0, np.nan, X)

pipeline = joblib.load("models/lightgbm_credit_score_pipeline.joblib")

# raw_customer_df: a DataFrame with the same raw columns as the training data
label_map = {0: "Poor", 1: "Standard", 2: "Good"}
predictions = pipeline.predict(raw_customer_df)
predicted_labels = [label_map[p] for p in predictions]

# Optional: class probabilities
probabilities = pipeline.predict_proba(raw_customer_df)
```

> **Note:** Use the same scikit-learn and LightGBM versions as in training. Loading a `.joblib` file across different versions can fail or give different results. Only load model files from sources you trust.

---

## 9. License & Attribution

- Dataset: [Ayush Sharma on Kaggle](https://www.kaggle.com/datasets/ayushsharma0812/dataset-for-credit-score-classification). Please follow the dataset's own license terms.
- Developed as a reference architecture for end-to-end credit risk modeling.