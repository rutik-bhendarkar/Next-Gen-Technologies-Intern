# Week 4 – Pandas & Data Visualization

A complete, offline-runnable exploratory data analysis project using Python, Pandas, NumPy, Matplotlib and Seaborn.

## Internship task coverage

This project follows the Week 4 task in the supplied 24-week AI/ML internship plan:

1. Load a real-world dataset.
2. Explore its structure.
3. Handle missing values.
4. Remove duplicates where present.
5. Perform statistical analysis.
6. Create at least eight visualizations (this project creates 12).
7. Write key observations.

## Dataset

The project uses a historical Mauna Loa weekly atmospheric CO₂ dataset distributed through the Statsmodels datasets collection. It contains 2,284 weekly date records from 1958-03-29 through 2001-12-29 and 59 missing CO₂ observations. The CSV is bundled locally so that the notebook does not need to download data when it runs.

- Raw input: `data/raw/mauna_loa_co2_weekly.csv`
- Cleaned output: `data/processed/mauna_loa_co2_weekly_cleaned.csv`
- Measurement column: `co2_ppm` (parts per million)
- Date column: `date`

The missing readings are time-interpolated in a separate cleaned DataFrame. The original raw file is never overwritten. No duplicate rows or duplicate dates were present in this dataset. IQR outliers are reviewed but not automatically deleted.

## Detailed learning material

A complete step-by-step guide is included at `docs/Week4_Detailed_Implementation_Guide.md`. It explains each Pandas operation, missing-value handling, duplicate checks, feature engineering, statistics, IQR screening, all 12 charts, expected observations, common errors, viva questions and a suggested daily internship report entry. A compact submission checklist is available at `submission/Week4_Submission_Checklist.md`.

## Project structure

```text
Week4_Pandas_Data_Visualization/
├── data/
│   ├── raw/
│   │   └── mauna_loa_co2_weekly.csv
│   └── processed/
│       └── mauna_loa_co2_weekly_cleaned.csv
├── notebooks/
│   ├── Week4_Pandas_Data_Visualization.ipynb
│   └── Week4_Pandas_Data_Visualization_Source.ipynb
├── outputs/
│   ├── figures/                  # 12 PNG visualizations
│   ├── tables/                   # Annual, monthly, decade and missing summaries
│   ├── summary_metrics.json
│   └── cleaning_audit.json
├── reports/
│   └── Week4_EDA_Analysis_Report.md
├── docs/
│   └── Week4_Detailed_Implementation_Guide.md
├── submission/
│   └── Week4_Submission_Checklist.md
├── src/
│   └── analysis.py               # Reusable cleaning, analysis and chart pipeline
├── tests/
│   └── test_analysis.py
├── requirements.txt
├── run_analysis.py
└── README.md
```

## Setup in Windows / VS Code

### 1. Extract and open

Extract `Week4_Pandas_Data_Visualization.zip`. In VS Code, select **File → Open Folder** and open the extracted `Week4_Pandas_Data_Visualization` folder.

### 2. Create a virtual environment (recommended)

Open the VS Code terminal (PowerShell) in the project folder:

```powershell
py -3.12 -m venv .venv
```

Activate it:

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation in the current terminal session:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 4. Run the complete analysis script

```powershell
python run_analysis.py
```

This reads the bundled raw CSV, checks/cleans the data, produces summary tables and creates all 12 charts.

### 5. Open and run the Jupyter Notebook

```powershell
python -m notebook
```

Open `notebooks/Week4_Pandas_Data_Visualization.ipynb`, select the `.venv` Python kernel if prompted, and choose **Run → Run All Cells**. The notebook contains explanations, code, results and embedded visualizations. A clean source copy is also included in `notebooks/Week4_Pandas_Data_Visualization_Source.ipynb`.

### 6. Run automated tests

```powershell
python -m unittest discover -s tests -v
```

The test suite checks dataset shape, missing-value detection and cleaning, duplicate-date handling, time-feature creation, summary tables and IQR output structure.

## Expected results

When run with the bundled dataset:

- 2,284 rows and 2 columns in the raw CSV.
- 59 missing `co2_ppm` observations before cleaning.
- 0 exact duplicate rows and 0 duplicate dates.
- 0 missing `co2_ppm` values after interpolation in the cleaned copy.
- 12 PNG visualizations in `outputs/figures/`.
- Annual, monthly, decade and missing-value summary tables in `outputs/tables/`.
- Eight automated tests pass in the included test suite.

## Visualization list

1. Raw weekly CO₂ trend with missing gaps.
2. Missing-value counts.
3. CO₂ distribution histogram.
4. CO₂ boxplot by decade.
5. Annual mean trend.
6. Average by calendar month.
7. Weekly values with 52-week rolling mean.
8. Year-month heatmap.
9. Year-over-year change in annual mean.
10. Year versus CO₂ scatter plot and trend line.
11. Numeric correlation heatmap.
12. Monthly boxplot.

## Important interpretation notes

- The dataset ends in December 2001; it is historical and is not a current CO₂ dataset.
- Interpolated values are estimates, not direct measurements.
- The raw CSV stays unchanged; cleaned values are stored separately.
- There were no duplicate rows in the bundled dataset. The code includes duplicate-removal logic for other input files.
- The IQR rule did not flag observations in this dataset. This does not guarantee there are no domain-specific anomalies.
- Pooled monthly averages can contain both long-term trend and seasonal effects; this project does not claim they isolate seasonality.

## Data references

- Statsmodels, Mauna Loa Weekly Atmospheric CO₂ Data: https://www.statsmodels.org/stable/datasets/generated/co2.html
- NOAA Global Monitoring Laboratory, Trends in Atmospheric CO₂ – Data: https://www.gml.noaa.gov/ccgg/trends/data.html
- Historical citation: Keeling, C.D. and T.P. Whorf (2004), *Atmospheric CO₂ concentrations derived from flask air samples at sites in the SIO network*, in *Trends: A Compendium of Data on Global Change*, CDIAC, Oak Ridge National Laboratory.

## Submission checklist

- [x] Python source code
- [x] Jupyter Notebook
- [x] Dataset (raw copy)
- [x] Cleaned dataset
- [x] At least 8 visualizations (12 included)
- [x] Statistical analysis
- [x] EDA report and key observations
- [x] README and requirements
- [x] Automated tests

Before submission, run the notebook on your own computer, review the generated charts, and make sure you can explain the missing-value and outlier decisions to your mentor/evaluator.
