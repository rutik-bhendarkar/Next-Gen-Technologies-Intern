# Week 6 — Detailed Implementation Guide

## Practical objective
Convert the raw customer-churn dataset into clean features that can be used by a future machine-learning model.

## Step 1 — Inspect the raw data
Load `data/raw/customer_churn_raw.csv` with `pd.read_csv`. Use `head`, `shape`, `info`, `describe`, `isna().sum` and `duplicated().sum` to understand the dataset.

## Step 2 — Remove duplicates
`drop_duplicates()` removes identical rows. Eight duplicate rows were intentionally added to the 500 generated records. The customer identifier is not a model feature.

## Step 3 — Validate values
Domain checks are used before statistical outlier processing. Age must be 18–100, tenure 0–120 months, monthly charges 0–300, total charges non-negative and support calls 0–30. Values outside these documented ranges are changed to missing and then imputed downstream.

## Step 4 — Missing-value treatment
A median is robust to skew and extreme numeric values, so numeric features use `SimpleImputer(strategy='median')`. Categories use `SimpleImputer(strategy='most_frequent')`. The imputers are fitted on training rows only.

## Step 5 — Outlier detection and treatment
For each numeric column, compute Q1, Q3 and IQR = Q3−Q1. Flag values outside Q1−1.5×IQR and Q3+1.5×IQR. In the pipeline, `IQRClipper` learns bounds on imputed training data, then caps train and test values to those bounds. Clipping is one treatment, not a universal rule; domain context must be considered.

## Step 6 — Encoding
The target `Churn` is mapped to 0/1 (`No`=0, `Yes`=1). Input categories are converted into binary indicator columns with `OneHotEncoder(handle_unknown='ignore')`. `CustomerID` is excluded because it is an identifier, not a meaningful generalizable input.

## Step 7 — Feature scaling
Two separate preprocessing variants are saved:
- StandardScaler: centers and scales numeric variables using training mean and standard deviation.
- MinMaxScaler: scales numeric variables to the training range [0,1].

One-hot columns are not scaled. The numeric pipeline order is impute → IQR cap → scale.

## Step 8 — Train/test split
Use an 80/20 stratified split with `random_state=42`. The split happens before fitting imputers, IQR bounds, one-hot categories and scaling parameters. The test subset is only transformed, never used to fit preprocessing.

## Deliverables
- Raw dataset and dataset card
- Auditable cleaned dataset before imputation
- Standardized and normalized train/test CSV outputs
- Serialized fitted preprocessing objects (`models/*.joblib`)
- Jupyter notebook, source module, report, plots and tests

## Limitations
This is an educational synthetic dataset. The assignment demonstrates data preparation, not real-world churn prediction. Week 6 does not train or evaluate a predictive model.
