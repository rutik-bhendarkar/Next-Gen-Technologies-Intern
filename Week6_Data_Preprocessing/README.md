# Week 6 — Data Preprocessing for Machine Learning

A detailed, reproducible practical assignment for Week 6 of the 24-week AI/ML internship plan.

## Requirements covered
- Missing-value analysis and imputation
- Exact duplicate removal
- Domain validation and outlier detection/treatment
- Target label encoding and categorical one-hot encoding
- Standardization with StandardScaler
- Normalization with MinMaxScaler
- Stratified train/test split
- Export of raw, cleaned, train/test and fitted preprocessing artifacts

## Dataset
The included customer-churn dataset is synthetic and generated with a fixed random seed. It contains 500 generated customers plus 8 exact duplicate rows, missing values and seeded invalid numeric values for practice. See `data/DATASET_CARD.md`.

## Setup (Windows PowerShell / VS Code)
```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Run the complete pipeline
```powershell
python run_analysis.py
```

## Open the notebook
```powershell
python -m notebook
```
Open `notebooks/Week6_Data_Preprocessing.ipynb` and choose **Run All**.

## Run tests
```powershell
python -m unittest discover -s tests -v
```

## Important outputs
- `data/processed/cleaned_dataset_before_imputation.csv`
- `data/processed/train_processed_standard.csv`
- `data/processed/test_processed_standard.csv`
- `data/processed/train_processed_minmax.csv`
- `data/processed/test_processed_minmax.csv`
- `models/preprocessor_standard.joblib`
- `models/preprocessor_minmax.joblib`
- `outputs/preprocessing_summary.json`
- `outputs/figures/`

## Leakage prevention
The train/test split happens before fitting imputation values, IQR clipping bounds, one-hot categories or scaling statistics. All preprocessing steps are fitted on the training partition and applied to the test partition with `transform` only.

## Scope note
This week prepares the data; it does not train a churn prediction model. Model building and evaluation appear in later weeks of the internship plan. Review the notebook and be prepared to explain the decisions before submission.
