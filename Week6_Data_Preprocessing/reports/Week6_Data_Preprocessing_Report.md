# Week 6 — Data Preprocessing Report

## 1. Project overview
**Title:** Customer Churn Dataset — Data Cleaning and Preprocessing  
**Objective:** Transform a raw mixed-type dataset into machine-learning-ready training and testing files using Pandas and Scikit-learn.

The assignment implements the Week 6 requirements: missing-value handling, duplicate removal, outlier detection, label encoding, one-hot encoding, feature scaling, normalization and train-test split.

## 2. Dataset description
The included dataset is **synthetic**, generated for this learning exercise with a fixed random seed. It does not contain real customer records. The raw CSV contains 508 rows: 500 generated records plus 8 deliberately duplicated rows. After exact duplicate removal, 500 rows remain.

**Target:** `Churn` (`No` / `Yes`).  
**Identifier:** `CustomerID` (excluded from predictive features).  
**Numeric features:** Age, TenureMonths, MonthlyCharges, TotalCharges, SupportCalls.  
**Categorical features:** ContractType, PaymentMethod, InternetService, PaperlessBilling.

## 3. Initial data-quality findings

### Missing values in raw data
- `Age`: 20 missing cells
- `TenureMonths`: 12 missing cells
- `MonthlyCharges`: 16 missing cells
- `TotalCharges`: 17 missing cells
- `SupportCalls`: 14 missing cells
- `ContractType`: 9 missing cells
- `PaymentMethod`: 8 missing cells
- `InternetService`: 7 missing cells
- `PaperlessBilling`: 7 missing cells

### Duplicate records
- Rows before deduplication: 508
- Exact duplicate rows removed: 8
- Rows after deduplication: 500

### Domain validation
A total of **6** invalid numeric entries were converted to missing values. Rules applied: Age 18–100; TenureMonths 0–120; MonthlyCharges 0–300; TotalCharges non-negative; SupportCalls 0–30.

| Feature | Invalid values changed to missing |
|---|---:|
| Age | 2 |
| TenureMonths | 1 |
| MonthlyCharges | 1 |
| TotalCharges | 1 |
| SupportCalls | 1 |

## 4. Missing-value treatment
Numeric columns use median imputation; categorical columns use most-frequent imputation. Crucially, imputers are fitted on the training subset only. The auditable `cleaned_dataset_before_imputation.csv` preserves missing cells so the original cleaning decisions remain inspectable.

## 5. Outlier detection and handling
IQR = Q3 − Q1. A value is flagged when it is below Q1 − 1.5×IQR or above Q3 + 1.5×IQR. The diagnostic counts below are calculated after domain validation; invalid domain entries have already been marked missing.

| Feature | IQR outlier count |
|---|---:|
| Age | 3 |
| TenureMonths | 0 |
| MonthlyCharges | 2 |
| TotalCharges | 8 |
| SupportCalls | 5 |

For the ML pipeline, the custom IQR clipper learns lower and upper bounds from imputed training features only. The same learned bounds are applied to test features. Outliers are capped rather than blindly deleted to preserve rows for this educational exercise.

## 6. Encoding
- Target label encoding: `No → 0`, `Yes → 1`.
- Categorical inputs: one-hot encoding via `OneHotEncoder(handle_unknown='ignore')`.
- `CustomerID` is excluded from X because it is an identifier rather than a generalizable signal.

The fitted output produced **17 encoded feature columns** (before the target column).

## 7. Feature scaling and normalization
Two output variants are produced:
1. **StandardScaler:** transforms numeric features using training-set mean and standard deviation.
2. **MinMaxScaler:** maps numeric training features to [0, 1].

Categorical one-hot columns are left as binary indicators. Both fitted preprocessing objects are saved under `models/`.

## 8. Train/test split
A stratified 80/20 split is used with `random_state=42`.

| Partition | Rows |
|---|---:|
| Training | 400 |
| Testing | 100 |
| Total | 500 |

Stratification preserves the target class proportion as closely as possible. The test subset is not used to fit imputers, clipping bounds, one-hot categories or scalers.

## 9. Visualizations
Six figures are generated under `outputs/figures/`:
1. Raw missing values by column
2. Target distribution
3. Numeric boxplots before/after domain validation
4. Numeric feature histograms
5. Numeric correlation heatmap
6. Churn proportion by contract type

## 10. Deliverables
- Raw CSV and dataset card
- Auditable cleaned CSV before imputation
- StandardScaler train and test CSV files
- MinMaxScaler train and test CSV files
- Fitted preprocessing joblib artifacts
- Executed Jupyter Notebook
- Python preprocessing source and runner
- Six visualizations
- Automated tests and this report

## 11. Testing
Automated tests cover duplicate removal, domain validation, IQR clipping, end-to-end pipeline outputs, no missing values in processed tables, binary target encoding, and MinMax numeric ranges. Test log: `outputs/test_results.txt`.

## 12. Conclusion
The raw dataset was cleaned and transformed into numeric feature matrices suitable for a later ML model. Missing data is imputed, duplicates are removed, invalid values are flagged, outliers are handled using training-fitted IQR bounds, categories are encoded, numeric features are scaled/normalized, and train/test partitions are exported.

**Scope boundary:** Week 6 prepares data; it does not fit a churn classifier. Predictive model training and evaluation are part of later weeks. Because the dataset is synthetic, the results demonstrate preprocessing techniques rather than real customer behavior.
