# Week 7 — Exploratory Data Analysis (EDA)

A complete, reproducible Week 7 EDA assignment using the synthetic customer-churn dataset continued from Week 6.

## Requirements covered
- End-to-end dataset inspection and data-quality audit
- Univariate, bivariate and multivariate analysis
- Correlation analysis and numeric distributions
- IQR outlier analysis
- Feature relationships and categorical churn-rate analysis
- 19 visualizations (exceeds the plan's minimum of 10)
- Jupyter Notebook, raw/analysis dataset, summary tables and detailed EDA report

## Dataset
The data is artificial educational data with 500 unique generated customers plus 8 duplicated rows in the raw file. It is not real customer information. See `data/DATASET_CARD.md`.

## Windows / VS Code setup
```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Run EDA
```powershell
python run_eda.py
```

## Run notebook
```powershell
python -m notebook
```
Open `notebooks/Week7_Exploratory_Data_Analysis.ipynb` and choose **Run All**.

## Run tests
```powershell
python -m unittest discover -s tests -v
```

## Main outputs
- `data/processed/customer_churn_eda_audited.csv`
- `reports/Week7_EDA_Report.md`
- `outputs/figures/` — all 19 visualizations
- `outputs/tables/numeric_descriptive_statistics.csv`
- `outputs/tables/categorical_frequencies.csv`
- `outputs/tables/iqr_outlier_summary.csv`
- `outputs/tables/churn_rate_by_category.csv`
- `outputs/tables/correlation_matrix.csv`
- `outputs/tables/categorical_chi_square.csv`
- `outputs/summary.json` and `outputs/audit_report.json`

## Analysis choices
Exact duplicate rows are removed from the analysis copy. Invalid numeric values outside the domain limits documented in the Week 6 project are marked missing. Values are not imputed for descriptive EDA, and IQR outliers are flagged instead of automatically deleted.

## Academic integrity and interpretation
Review the notebook and be able to explain the calculations. The dataset is synthetic and its labels were generated using selected features. Results are only for learning and must not be presented as evidence about real customers or as causal findings.
