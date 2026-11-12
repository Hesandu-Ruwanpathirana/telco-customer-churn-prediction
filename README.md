# Telco Customer Churn Prediction

Binary classification project predicting whether a telecom customer will **churn** (cancel their service), comparing **K-Nearest Neighbours**, **Logistic Regression** and **Random Forest**.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Hesandu-Ruwanpathirana/telco-customer-churn-prediction/blob/main/churn_prediction.ipynb)

## Dataset

[Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) (IBM sample data, via Kaggle)

- 7,043 customers, 21 columns (demographics, account details and services)
- Target: `Churn` (Yes/No), imbalanced at ~26.5% churn

## Project Workflow

| Section | Description |
|---|---|
| 1. Loading | Load the data and inspect column types |
| 2. Cleaning | Fix 11 blank `TotalCharges` values, drop `customerID`, check missing values and duplicates |
| 3. EDA | Distributions, box plots, churn rate by category and correlation heatmap |
| 4. Preprocessing | One-hot encoding, stratified 80/20 train/test split, standard scaling (fitted on training data only) |
| 5–7. Models | Logistic Regression, KNN and Random Forest: baseline and tuned with `GridSearchCV` (stratified 5-fold CV, F1 scoring) |
| 8. Comparison | Results table and ROC curves for all models |
| 9. Ethics | Fairness, explainability, privacy, post-deployment monitoring, limitations |

## Results (test set)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| LR Baseline | 0.806 | 0.659 | 0.559 | 0.605 | 0.842 |
| LR Tuned | 0.743 | 0.511 | 0.783 | 0.618 | 0.841 |
| KNN Baseline | 0.763 | 0.554 | 0.551 | 0.552 | 0.793 |
| KNN Tuned | 0.790 | 0.610 | 0.580 | 0.595 | 0.831 |
| RF Baseline | 0.785 | 0.619 | 0.495 | 0.550 | 0.825 |
| **RF Tuned** | 0.759 | 0.532 | 0.775 | **0.631** | **0.845** |

**Selected model: Tuned Random Forest** (`class_weight="balanced"`, `max_depth=10`, `min_samples_leaf=10`, 200 trees), which achieved the highest F1 and ROC-AUC.

## Key Findings

- Month-to-month contracts, short tenure, fiber optic internet and electronic check payments are the strongest signs of churn.
- Handling class imbalance (`class_weight="balanced"`) had the largest effect of any setting, raising recall from ~0.5 to ~0.78.
- Accuracy was not used to select models, since predicting "No Churn" for everyone would already score 73.5%.
- Logistic Regression weights and Random Forest feature importances agree with the EDA findings.

## How to Run

**Option 1:** click the **Open in Colab** badge above, then **Runtime → Run all**. The dataset loads directly from this repository.

**Option 2 (locally):**
```bash
git clone https://github.com/Hesandu-Ruwanpathirana/telco-customer-churn-prediction.git
cd telco-customer-churn-prediction
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
jupyter notebook churn_prediction.ipynb
```

All results are reproducible (`random_state=42`).

## Tech Stack

Python · pandas · NumPy · Matplotlib · seaborn · scikit-learn · Google Colab

## Author

**Hesandu Ruwanpathirana**, BSc (Hons) Artificial Intelligence and Data Science, Robert Gordon University (IIT Campus, Sri Lanka)

## License

MIT
