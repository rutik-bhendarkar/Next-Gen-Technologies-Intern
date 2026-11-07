# Week 7 — Exploratory Data Analysis (EDA) Report

## 1. Executive summary

This project performs end-to-end exploratory data analysis on the synthetic customer-churn dataset continued from Week 6. The work includes univariate, bivariate and multivariate analysis; distribution, correlation, outlier and feature-relationship analysis; at least 10 visualizations; and written observations.

- Raw records: **508** across **11 columns**.
- Exact duplicate rows: **8**.
- Unique records after exact deduplication: **500**.
- Customers labelled `Yes` for churn: **107** (21.4% of analyzed records).
- Customers labelled `No`: **393**.
- Generated visualizations: **19**.

## 2. Problem statement and objectives

**Problem statement:** Understand the structure, quality, distributions and relationships in a customer-churn dataset before attempting machine-learning model development.

**Objectives:**
1. Load and inspect the dataset using Pandas.
2. Examine missing values, duplicate records and invalid domain values.
3. Perform univariate analysis of numeric and categorical features.
4. Perform bivariate analysis between features and churn.
5. Perform multivariate and correlation analysis.
6. Identify potential outliers using the IQR rule.
7. Produce visualizations and summarize actionable learning observations.

## 3. Dataset description

The dataset has **11 columns**: an identifier (`CustomerID`), five numeric attributes (`Age`, `TenureMonths`, `MonthlyCharges`, `TotalCharges`, `SupportCalls`), four categorical attributes (`ContractType`, `PaymentMethod`, `InternetService`, `PaperlessBilling`), and the target (`Churn`). It is an **artificially generated educational dataset**, not a downloaded company dataset and not real customer information. Its churn labels are generated using a probabilistic rule involving selected customer features. See `data/DATASET_CARD.md`.

## 4. Methodology and data-quality audit

1. Loaded the raw CSV without modifying it.
2. Counted missing values and exact duplicate rows.
3. Removed **8** exact duplicate rows for the analytical working copy.
4. Applied the same transparent domain checks used in Week 6: Age 18–100, TenureMonths 0–120, MonthlyCharges 0–300, TotalCharges non-negative, SupportCalls 0–30. Values outside these limits were marked missing in the analysis copy, not silently retained as valid measurements.
5. Did **not** impute missing values for descriptive EDA. Every statistic is calculated from available valid observations, and sample counts are shown.
6. Did **not** automatically delete IQR outliers. They are flagged for review because an unusual valid customer value is not necessarily an error.

### Data audit details

- Raw missing values by column: `{"Age": 20, "Churn": 0, "ContractType": 9, "CustomerID": 0, "InternetService": 7, "MonthlyCharges": 16, "PaperlessBilling": 7, "PaymentMethod": 8, "SupportCalls": 14, "TenureMonths": 12, "TotalCharges": 17}`
- Impossible values set to missing: `{"Age": 2, "MonthlyCharges": 1, "SupportCalls": 1, "TenureMonths": 1, "TotalCharges": 1}`
- Remaining missing values in analysis copy: `{"Age": 20, "Churn": 0, "ContractType": 9, "CustomerID": 0, "InternetService": 7, "MonthlyCharges": 16, "PaperlessBilling": 6, "PaymentMethod": 8, "SupportCalls": 15, "TenureMonths": 13, "TotalCharges": 18}`

## 5. Univariate analysis

### 5.1 Numeric descriptive statistics

| feature        |   count |    mean |     50% |     std |   min |     max |   missing_count |   skewness |
|:---------------|--------:|--------:|--------:|--------:|------:|--------:|----------------:|-----------:|
| Age            |     480 |   42.15 |   42    |   13.42 |    18 |    82   |              20 |       0.18 |
| TenureMonths   |     487 |   34.97 |   35    |   21.21 |     0 |    72   |              13 |       0.09 |
| MonthlyCharges |     484 |   71.45 |   73.66 |   32.54 |    17 |   195   |              16 |       0.25 |
| TotalCharges   |     482 | 2502.41 | 1919.28 | 2051.8  |     0 | 10020.9 |              18 |       1.05 |
| SupportCalls   |     485 |    1.95 |    2    |    1.5  |     0 |    14   |              15 |       2.26 |

The histograms and boxplots in `outputs/figures/` show the shape, spread, skew and potential extreme observations of each numeric feature. Mean and median are both reported because skewed distributions can make the mean sensitive to high values.

### 5.2 Categorical distributions

Frequency counts and percentages for contract, payment method, internet service, paperless billing and churn are saved in `outputs/tables/categorical_frequencies.csv`.

## 6. Bivariate analysis

### 6.1 Numeric features vs churn

| Feature | Median for Churn = No | Median for Churn = Yes |
|---|---:|---:|
| TenureMonths | 37.00 | 23.50 |
| MonthlyCharges | 71.31 | 85.50 |

The boxplots and violin plot compare numeric feature distributions across the target classes. These comparisons are descriptive and do not show that a feature causes churn.

### 6.2 Categorical features vs churn

| feature          | category         |   customers |   churned |   churn_rate |
|:-----------------|:-----------------|------------:|----------:|-------------:|
| ContractType     | Month-to-month   |         264 |        78 |        29.55 |
| ContractType     | One year         |         116 |        19 |        16.38 |
| ContractType     | Two year         |         111 |         9 |         8.11 |
| PaymentMethod    | Bank transfer    |         118 |        25 |        21.19 |
| PaymentMethod    | Credit card      |          99 |        22 |        22.22 |
| PaymentMethod    | Electronic check |         187 |        40 |        21.39 |
| PaymentMethod    | Mailed check     |          88 |        18 |        20.45 |
| InternetService  | DSL              |         155 |        32 |        20.65 |
| InternetService  | Fiber optic      |         242 |        56 |        23.14 |
| InternetService  | No internet      |          96 |        18 |        18.75 |
| PaperlessBilling | No               |         127 |        26 |        20.47 |
| PaperlessBilling | Yes              |         367 |        80 |        21.8  |

The highest observed category churn rate among contract types is **Month-to-month (29.5%)**, while the lowest is **Two year (8.1%)**.

Category-level churn rates are computed using records with non-missing values for both the category and target. The number of customers in each category is included in the saved table.

## 7. Multivariate and correlation analysis

Pearson correlations for numeric features and the binary-encoded churn target (`No = 0`, `Yes = 1`) are shown below.

|                |    Age |   TenureMonths |   MonthlyCharges |   TotalCharges |   SupportCalls |   ChurnBinary |
|:---------------|-------:|---------------:|-----------------:|---------------:|---------------:|--------------:|
| Age            |  1     |          0.031 |            0.039 |          0.069 |          0.049 |        -0.026 |
| TenureMonths   |  0.031 |          1     |            0.023 |          0.739 |          0.007 |        -0.171 |
| MonthlyCharges |  0.039 |          0.023 |            1     |          0.565 |          0.008 |         0.141 |
| TotalCharges   |  0.069 |          0.739 |            0.565 |          1     |          0.036 |        -0.003 |
| SupportCalls   |  0.049 |          0.007 |            0.008 |          0.036 |          1     |         0.101 |
| ChurnBinary    | -0.026 |         -0.171 |            0.141 |         -0.003 |          0.101 |         1     |

The selected-feature pairplot and scatter plots help inspect joint distributions and possible patterns that are not visible in one-variable summaries. Correlation describes linear association; it does not establish causation, and a low Pearson value does not rule out non-linear relationships.

## 8. Outlier analysis

The Interquartile Range (IQR) rule defines potential outliers as values below `Q1 - 1.5 × IQR` or above `Q3 + 1.5 × IQR`.

| feature        |     q1 |      q3 |     iqr |   lower_fence |   upper_fence |   potential_outlier_count |   potential_outlier_percent_of_non_missing |   non_missing_count |
|:---------------|-------:|--------:|--------:|--------------:|--------------:|--------------------------:|-------------------------------------------:|--------------------:|
| Age            |  32    |   51    |   19    |          3.5  |         79.5  |                         3 |                                       0.62 |                 480 |
| TenureMonths   |  16    |   52    |   36    |        -38    |        106    |                         0 |                                       0    |                 487 |
| MonthlyCharges |  40.62 |   95.31 |   54.69 |        -41.41 |        177.34 |                         2 |                                       0.41 |                 484 |
| TotalCharges   | 887.04 | 3696.12 | 2809.08 |      -3326.57 |       7909.73 |                         8 |                                       1.66 |                 482 |
| SupportCalls   |   1    |    3    |    2    |         -2    |          6    |                         5 |                                       1.03 |                 485 |

Potential outliers are retained in the EDA dataset. They should be investigated in context before any transformation or deletion. IQR flags are diagnostic indicators, not proof of data errors.

## 9. Categorical association test (supplementary)

A chi-square test of independence and Cramér's V effect-size estimate are included as supplementary descriptive diagnostics for categorical features against churn.

| feature          |   chi_square |   degrees_of_freedom |   p_value |   cramers_v |   n |
|:-----------------|-------------:|---------------------:|----------:|------------:|----:|
| ContractType     |      23.6493 |                    2 |    0      |      0.2195 | 491 |
| PaymentMethod    |       0.0889 |                    3 |    0.9931 |      0.0134 | 492 |
| InternetService  |       0.8831 |                    2 |    0.643  |      0.0423 | 493 |
| PaperlessBilling |       0.0355 |                    1 |    0.8506 |      0.0085 | 494 |

The dataset is synthetic, and the tests are not evidence of real-world population effects. Small sample sizes, missing categories and the artificial label-generation process limit inferential interpretation.

## 10. Key findings

1. The cleaned analytical copy contains **500 distinct rows** after exact duplicate removal.
2. The observed churn share is **21.4%** in this synthetic sample.
3. Contract category churn rates differ in this generated sample: The highest observed category churn rate among contract types is **Month-to-month (29.5%)**, while the lowest is **Two year (8.1%)**.
4. Median tenure is **37.0 months** for `No` and **23.5 months** for `Yes` in the available observations.
5. Median monthly charges are **71.31** for `No` and **85.50** for `Yes`.
6. Correlation heatmap indicates that tenure and total charges have a comparatively strong positive linear relationship in this dataset (Pearson r = **0.739**).

These findings reflect the designed synthetic dataset only. They must not be presented as real telecom/customer-business conclusions.

## 11. Limitations

- Synthetic, fixed-size dataset; not representative of any real customer population.
- Only 500 unique records; some features contain missing values.
- Domain limits are assumptions defined for this educational exercise.
- Churn labels were algorithmically generated from selected feature patterns, so apparent associations may partly reflect the data-generation rule.
- Pairwise deletion means sample size can vary across analyses.
- Correlation and chi-square tests are descriptive and not causal evidence.

## 12. Conclusion

The EDA process provides a structured understanding of the dataset's quality, distributions, class balance, outliers and feature relationships. The analysis produced 19 charts, descriptive-statistics and categorical summaries, a correlation matrix, outlier audit and key findings. The next step in the internship plan is to use suitable train/test procedures for model building in later weeks; no predictive model is trained in this Week 7 assignment.

## 13. Reproducibility

From the project root:

```bash
python -m pip install -r requirements.txt
python run_eda.py
python -m unittest discover -s tests -v
```

Open `notebooks/Week7_Exploratory_Data_Analysis.ipynb` and run all cells to review the complete notebook.
