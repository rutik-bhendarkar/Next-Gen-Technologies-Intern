# Week 7 — Detailed Implementation Guide

## Project purpose
Perform an end-to-end Exploratory Data Analysis (EDA) on the synthetic churn dataset from Week 6.

## Folder structure
```text
Week7_Exploratory_Data_Analysis/
├── data/
│   ├── raw/customer_churn_raw.csv
│   ├── processed/customer_churn_eda_audited.csv
│   └── DATASET_CARD.md
├── notebooks/Week7_Exploratory_Data_Analysis.ipynb
├── src/eda.py
├── reports/Week7_EDA_Report.md
├── reports/tables/
├── outputs/figures/
├── outputs/tables/
├── tests/test_eda.py
├── run_eda.py
├── requirements.txt
└── README.md
```

## EDA stages
1. **Load and inspect:** dimensions, column names, data types, `head`, `tail`, `info`, `describe`, unique counts.
2. **Quality audit:** missing values and exact duplicate rows. Remove exact duplicates only from the analysis copy.
3. **Domain validation:** values outside the declared Week 6 domain bounds are marked missing. Raw file remains unchanged.
4. **Univariate numeric analysis:** count, mean, median, standard deviation, quartiles, skewness, histograms and boxplots.
5. **Univariate categorical analysis:** value counts and row percentages.
6. **Bivariate analysis:** numeric distributions by churn and category-specific churn rates.
7. **Multivariate analysis:** pairplot, scatterplots and Pearson correlation matrix.
8. **Outlier analysis:** IQR fences and counts. Outliers are flagged, not automatically removed.
9. **Key findings:** write data-driven observations and clearly disclose limitations.

## Important formulas
- Mean: `sum(x) / n`
- Sample variance: `sum((x - mean)^2) / (n - 1)`
- Pearson correlation: covariance divided by the product of standard deviations.
- IQR: `Q3 - Q1`
- Lower IQR fence: `Q1 - 1.5 * IQR`
- Upper IQR fence: `Q3 + 1.5 * IQR`
- Churn rate in a category: `count(Churn == Yes) / category_count * 100`

## Why no imputation in EDA?
The goal is first to understand what is actually missing. Filling missing values at the EDA stage would change distributions and could hide quality issues. Week 6 already implements the train-fitted preprocessing workflow; Week 7 reports the observed data and leaves missing values visible.

## Run on Windows / VS Code
```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python run_eda.py
python -m notebook
```
Open `notebooks/Week7_Exploratory_Data_Analysis.ipynb` and select **Run All**.

Run tests:
```powershell
python -m unittest discover -s tests -v
```

## Interpretation precautions
The included dataset is synthetic and generated for learning. Statistical patterns and chi-square results must not be described as verified customer-business findings. Association is not causation. Review IQR outliers rather than automatically deleting them.
