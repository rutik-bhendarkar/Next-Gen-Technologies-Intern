# WEEK 4 — PANDAS & DATA VISUALIZATION
## Detailed implementation and learning guide

**Internship phase:** Phase 1 — Python & Data Science Fundamentals  
**Assignment:** Data Analysis & Visualization  
**Dataset:** Historical Mauna Loa weekly atmospheric CO₂ dataset  
**Tools:** Python, Pandas, NumPy, Matplotlib, Seaborn, Jupyter Notebook  

---

## 1. Assignment requirements and coverage

The supplied 24-week internship plan asks the student to choose a real-world dataset and:

1. Load the dataset.
2. Explore its structure.
3. Handle missing values.
4. Remove duplicates.
5. Perform statistical analysis.
6. Create at least eight visualizations.
7. Write key observations.

The project fulfils these requirements with a reproducible Python pipeline and an executed Jupyter Notebook. It produces twelve visualizations, exported summary tables, a cleaned CSV, a cleaning audit and a written EDA report.

## 2. Problem statement

Explore historical weekly atmospheric carbon dioxide measurements from Mauna Loa Observatory. The analysis should identify the dataset's structure and quality, clean missing observations transparently, calculate descriptive statistics, study changes over time and communicate important patterns through charts.

This is a historical data-analysis exercise. The bundled data ends on **29 December 2001** and must not be described as current CO₂ measurements.

## 3. Dataset overview

The locally bundled CSV is distributed through the Statsmodels datasets collection. It has two columns:

| Column | Meaning | Data type after loading |
|---|---|---|
| `date` | Date of the weekly observation | `datetime64[ns]` |
| `co2_ppm` | Atmospheric CO₂ concentration in parts per million | `float64` |

Dataset facts from the supplied CSV:

- Rows: **2,284**
- Columns: **2**
- Date range: **1958-03-29 to 2001-12-29**
- Missing `co2_ppm` readings: **59**
- Exact duplicate rows: **0**
- Duplicate dates: **0**

The raw dataset is stored at `data/raw/mauna_loa_co2_weekly.csv`. It is preserved as-is; the cleaning pipeline writes a separate processed file.

## 4. Tools and their purpose

| Tool | Purpose in this assignment |
|---|---|
| Python | Main programming language |
| Pandas | Read CSV, inspect data, clean values, group records, calculate summaries and export tables |
| NumPy | Numerical calculations and array support |
| Matplotlib | Build and customize charts |
| Seaborn | Statistical plots such as histograms, boxplots, line plots and heatmaps |
| Jupyter Notebook | Combine code, output, charts and explanations in one submission |

## 5. Step 1 — Import libraries

```python
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")
```

**Explanation:**

- `Path` creates paths that work across operating systems.
- `pandas` is imported as `pd`, the standard alias.
- `numpy` is imported as `np`.
- `matplotlib.pyplot` is imported as `plt`.
- `seaborn` is imported as `sns`.
- `set_theme()` applies a consistent chart style.

## 6. Step 2 — Load the CSV dataset

```python
raw_df = pd.read_csv(
    "data/raw/mauna_loa_co2_weekly.csv",
    parse_dates=["date"]
)

print(raw_df.head())
```

**Important methods:**

- `pd.read_csv()` loads a comma-separated file into a DataFrame.
- `parse_dates=["date"]` asks Pandas to parse the date column as dates rather than plain text.
- `head()` displays the first five rows by default.

Other useful inspection commands:

```python
raw_df.tail()          # Last five rows
raw_df.shape           # (number of rows, number of columns)
raw_df.columns         # Column labels
raw_df.info()          # Data types and non-null counts
raw_df.dtypes          # Data type of each column
raw_df.describe()      # Descriptive statistics for numeric columns
```

**Expected result:** the DataFrame contains 2,284 rows and two columns. Dates should be parsed as datetime values and CO₂ values as numeric values.

## 7. Step 3 — Explore the dataset structure

```python
print("Shape:", raw_df.shape)
print("Columns:", raw_df.columns.tolist())
print("Data types:\n", raw_df.dtypes)
print("Information:")
raw_df.info()
print("Numeric summary:")
print(raw_df["co2_ppm"].describe())
```

**Why this matters:** inspection helps identify incorrect column names, unexpected types, missing values and unusual ranges before cleaning or plotting. A date column accidentally loaded as text can make time-series operations fail or sort incorrectly.

## 8. Step 4 — Find missing values

```python
missing_count = raw_df.isna().sum()
missing_percent = raw_df.isna().mean() * 100

missing_report = pd.DataFrame({
    "missing_count": missing_count,
    "missing_percent": missing_percent
})
print(missing_report)
```

**Pandas methods:**

- `isna()` returns `True` for missing cells.
- `sum()` counts the `True` values in each column.
- `mean()` on Boolean values returns the missing fraction; multiplying by 100 gives a percentage.

**Observed result:** the `co2_ppm` field has 59 missing values (approximately 2.58% of 2,284 rows). The date column has no missing values.

## 9. Step 5 — Check and remove duplicates

```python
exact_duplicate_count = raw_df.duplicated().sum()
duplicate_date_count = raw_df["date"].duplicated().sum()

clean_df = raw_df.drop_duplicates(subset=["date"], keep="first")
```

- `duplicated()` detects repeated rows or values.
- `drop_duplicates(subset=["date"])` keeps one observation per date.
- `keep="first"` retains the first row where repeated dates exist.

**Observed result:** this dataset contains zero exact duplicate rows and zero duplicate dates, so no rows are removed. The code still includes duplicate-handling logic so it can be reused with another CSV.

## 10. Step 6 — Handle missing values appropriately

A time series is ordered by date. For missing readings between valid dates, linear interpolation estimates a value using the surrounding observations. In this project the date is used as the index so Pandas can account for time spacing.

```python
clean_df = raw_df.copy()
clean_df["date"] = pd.to_datetime(clean_df["date"])
clean_df = clean_df.sort_values("date")

clean_df = clean_df.set_index("date")
clean_df["co2_ppm"] = clean_df["co2_ppm"].interpolate(
    method="time",
    limit_direction="both"
)
clean_df = clean_df.reset_index()
```

**Explanation:**

1. `copy()` protects the original in-memory DataFrame.
2. `sort_values("date")` ensures chronological order.
3. `set_index("date")` makes the date the time index.
4. `interpolate(method="time")` estimates values in missing gaps using time intervals.
5. `reset_index()` returns the date to a regular column.

**Data-integrity decision:** the raw CSV is not overwritten. Interpolated values are estimates, not direct observations. The cleaned file is stored separately at `data/processed/mauna_loa_co2_weekly_cleaned.csv`. In a real scientific workflow, the analyst should also consult domain guidance before deciding whether interpolation is suitable for a particular analysis.

After cleaning, confirm the result:

```python
print(clean_df.isna().sum())
print("Rows before:", len(raw_df))
print("Rows after:", len(clean_df))
```

The processed file contains 2,284 rows and zero missing `co2_ppm` values.

## 11. Step 7 — Create useful date features

Date components make time-based grouping and comparison easier.

```python
clean_df["year"] = clean_df["date"].dt.year
clean_df["month"] = clean_df["date"].dt.month
clean_df["month_name"] = clean_df["date"].dt.month_name()
clean_df["quarter"] = clean_df["date"].dt.quarter
clean_df["decade"] = (clean_df["year"] // 10) * 10
clean_df["week_number"] = clean_df["date"].dt.isocalendar().week.astype(int)
```

A 52-week rolling average is also useful for smoothing short-term movement:

```python
clean_df["rolling_52_week_mean"] = (
    clean_df["co2_ppm"].rolling(window=52, min_periods=26).mean()
)
```

`rolling(window=52)` calculates a moving statistic across a one-year-sized window of weekly records. `min_periods=26` allows a value to appear when at least half the window is available near the beginning of the series.

## 12. Step 8 — Perform statistical analysis

### 12.1 Descriptive statistics

```python
co2 = raw_df["co2_ppm"].dropna()

print("Count:", co2.count())
print("Mean:", co2.mean())
print("Median:", co2.median())
print("Mode:", co2.mode().tolist())
print("Variance:", co2.var())
print("Standard deviation:", co2.std())
print("Minimum:", co2.min())
print("Maximum:", co2.max())
print("Range:", co2.max() - co2.min())
print("Quartiles:\n", co2.quantile([0.25, 0.50, 0.75]))
```

**Meaning of the measures:**

- **Count:** number of non-missing observations.
- **Mean:** arithmetic average.
- **Median:** middle value after sorting.
- **Mode:** most frequently occurring value(s).
- **Variance:** average squared deviation from the mean (sample variance in Pandas by default).
- **Standard deviation:** square root of variance; expresses spread in the same unit as the data.
- **Minimum / maximum:** smallest / largest observed value.
- **Range:** maximum minus minimum.
- **Quartiles:** divide sorted data into four parts.

The report's observed-value summary (excluding missing readings) is approximately:

| Statistic | Value |
|---|---:|
| Count | 2,225 |
| Mean | 340.14 ppm |
| Median | 338.30 ppm |
| Standard deviation | 17.00 ppm |
| Minimum | 313.00 ppm |
| Maximum | 373.90 ppm |

These values describe the packaged historical series only.

### 12.2 Grouped analysis with `groupby()`

Annual average:

```python
annual_summary = (
    clean_df.groupby("year", as_index=False)
    .agg(annual_mean_ppm=("co2_ppm", "mean"),
         annual_median_ppm=("co2_ppm", "median"),
         observations=("co2_ppm", "size"))
)
```

Monthly average:

```python
monthly_summary = (
    clean_df.groupby("month", as_index=False)
    .agg(mean_co2_ppm=("co2_ppm", "mean"),
         median_co2_ppm=("co2_ppm", "median"),
         observations=("co2_ppm", "size"))
)
```

Decade summary:

```python
decade_summary = (
    clean_df.groupby("decade", as_index=False)
    .agg(mean_co2_ppm=("co2_ppm", "mean"),
         min_co2_ppm=("co2_ppm", "min"),
         max_co2_ppm=("co2_ppm", "max"),
         observations=("co2_ppm", "size"))
)
```

`groupby()` divides rows into groups, and `agg()` calculates one or more summaries for each group. These tables are exported to `outputs/tables/`.

## 13. Step 9 — Detect possible outliers with IQR

The Interquartile Range is:

`IQR = Q3 - Q1`

The usual screening boundaries are:

- Lower boundary = `Q1 - 1.5 × IQR`
- Upper boundary = `Q3 + 1.5 × IQR`

```python
q1 = clean_df["co2_ppm"].quantile(0.25)
q3 = clean_df["co2_ppm"].quantile(0.75)
iqr = q3 - q1
lower = q1 - 1.5 * iqr
upper = q3 + 1.5 * iqr

possible_outliers = clean_df[
    (clean_df["co2_ppm"] < lower) |
    (clean_df["co2_ppm"] > upper)
]
print(possible_outliers)
```

The project reports the IQR result but does not automatically delete observations. A global IQR rule can be inappropriate for a time series with a long-term upward trend. Any flagged point should be reviewed against dates, neighboring observations and domain knowledge.

For this bundled dataset, the global IQR screen flags zero observations.

## 14. Step 10 — Create and interpret visualizations

The assignment requires at least eight visualizations. The project includes twelve:

| No. | Chart | What it helps answer |
|---:|---|---|
| 1 | Raw weekly time-series line plot | How do the weekly values change over time, and where are the raw gaps? |
| 2 | Missing-values bar chart | Which column contains missing observations, and how many? |
| 3 | Histogram with density curve | What is the distribution of observed CO₂ values? |
| 4 | Boxplot by decade | How do the median, spread and range shift across decades? |
| 5 | Annual mean line plot | What is the long-term direction after aggregating by year? |
| 6 | Average by calendar month | How do pooled calendar-month means differ? |
| 7 | Weekly values with 52-week rolling mean | What trend is visible after smoothing short-term variation? |
| 8 | Year-month heatmap | How do the values vary across both year and month? |
| 9 | Year-over-year annual mean change | How much does the annual average change from one year to the next? |
| 10 | Year versus CO₂ scatter plot | How are individual weekly readings distributed over calendar time? |
| 11 | Numeric correlation heatmap | What linear associations exist among selected numeric features? |
| 12 | Month-wise boxplot | How do monthly distributions and spreads compare? |

**How to interpret common chart types:**

- **Line plot:** connected points emphasize order and trend. Dates should be sorted.
- **Histogram:** bars show the frequency of values within numeric intervals (bins).
- **Boxplot:** the centre line is the median; the box covers Q1 to Q3; whiskers and points help reveal spread and possible unusual values according to the plotting rule.
- **Scatter plot:** each point is one observation; a fitted line summarizes a simple linear pattern but does not prove causation.
- **Heatmap:** color encodes the magnitude of a value in a two-dimensional table.
- **Correlation heatmap:** values near +1 indicate positive linear association, values near -1 indicate negative linear association and values near 0 indicate weak linear association. Correlation does not establish cause and effect.

All twelve PNG files are located in `outputs/figures/` and are also displayed in the executed notebook.

## 15. Key observations from this dataset

1. The packaged historical series contains 2,284 weekly records spanning March 1958 to December 2001.
2. There are 59 missing CO₂ readings (about 2.58%) in the raw file. The date column is complete.
3. There are no exact duplicate rows or repeated dates in this dataset.
4. The cleaning step interpolates missing values in a separate processed copy; it does not overwrite raw data.
5. The observed-value mean is approximately 340.14 ppm and the median is approximately 338.30 ppm.
6. The annual mean is approximately 315.19 ppm in 1958 and 370.87 ppm in 2001. This is a historical endpoint comparison within this file, not a current reading.
7. The annual trend, decade boxplots and rolling mean display a rising long-term pattern in this historical period.
8. The global IQR screening rule flags zero observations. This does not prove that the dataset has no possible measurement anomalies.
9. Monthly pooled averages describe month-wise differences across the whole time span. Because the overall level changes over time, these averages should not be presented as a pure seasonal effect.
10. Twelve visualizations are included, exceeding the minimum of eight.

## 16. Output files and why they are included

| Path | Purpose |
|---|---|
| `data/raw/mauna_loa_co2_weekly.csv` | Original bundled observations |
| `data/processed/mauna_loa_co2_weekly_cleaned.csv` | Cleaned and feature-engineered copy |
| `notebooks/Week4_Pandas_Data_Visualization.ipynb` | Executed notebook with code, explanations and displayed charts |
| `notebooks/Week4_Pandas_Data_Visualization_Source.ipynb` | Notebook source version |
| `src/analysis.py` | Reusable loading, cleaning, statistics and chart functions |
| `run_analysis.py` | Command-line entry point |
| `outputs/figures/` | Twelve saved charts |
| `outputs/tables/` | Descriptive, annual, monthly, decade and year-month tables |
| `outputs/summary_metrics.json` | Machine-readable summary of key results |
| `outputs/cleaning_audit.json` | Before/after cleaning counts |
| `reports/Week4_EDA_Analysis_Report.md` | Formal written report |
| `tests/` | Automated checks for important analysis behavior |

## 17. Run the project in VS Code on Windows

Open PowerShell in the extracted `Week4_Pandas_Data_Visualization` project folder.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Run the complete reproducible pipeline:

```powershell
python run_analysis.py
```

Run the automated tests:

```powershell
python -m unittest discover -s tests -v
```

Launch Jupyter Notebook:

```powershell
python -m notebook
```

Open `notebooks/Week4_Pandas_Data_Visualization.ipynb` and use **Run → Run All Cells**. If VS Code asks for a kernel, select the Python interpreter inside `.venv`.

## 18. Common errors and fixes

| Error / symptom | Likely reason | Fix |
|---|---|---|
| `ModuleNotFoundError: No module named 'pandas'` | Dependencies are not installed in the active environment | Activate `.venv`, then run `python -m pip install -r requirements.txt` |
| Notebook cannot find the CSV | Notebook opened with an unexpected working directory | Open the project root as the VS Code folder and run the notebook there; inspect the path cell |
| Date `.dt` accessor error | Date column is still text | Run `pd.to_datetime(df["date"])` before using `.dt` |
| Charts do not display | Matplotlib backend or notebook display issue | Run the notebook in Jupyter; ensure chart cells execute and figures are saved under `outputs/figures/` |
| `jupyter` command not recognized | Jupyter is not installed in the active environment | Run `python -m pip install notebook ipykernel` and `python -m notebook` |

## 19. Viva / evaluation questions and answers

**Q1. What is Pandas?**  
Pandas is a Python library used for working with structured/tabular data. Its main structures are Series and DataFrame.

**Q2. What is a DataFrame?**  
A DataFrame is a two-dimensional labeled data structure with rows and columns.

**Q3. What is the difference between `head()` and `tail()`?**  
`head()` displays the first rows; `tail()` displays the last rows.

**Q4. How did you identify missing values?**  
I used `isna().sum()` to count missing cells in each column and `isna().mean() * 100` to calculate their percentages.

**Q5. How did you handle missing CO₂ readings?**  
I sorted by date, set the date as the index and used time-based linear interpolation. I retained the untouched raw file and saved the cleaned copy separately.

**Q6. Why did you check duplicates?**  
Repeated dates or rows can double-count observations and distort summaries. I checked exact duplicate rows and duplicate dates. This dataset had none.

**Q7. What does `groupby()` do?**  
It splits records into groups, applies calculations to each group and combines the results. I used it for annual, monthly and decade summaries.

**Q8. What is a histogram?**  
A histogram groups numeric values into intervals and displays how many observations fall in each interval.

**Q9. What does a boxplot show?**  
It summarizes a distribution using quartiles, median and whiskers, and helps identify possible outliers.

**Q10. What is IQR?**  
IQR is Q3 minus Q1. The project uses 1.5 times IQR beyond the quartiles as a simple outlier screening boundary.

**Q11. Why did you not automatically delete IQR outliers?**  
Time-series values can change legitimately over time. A global statistical rule should flag values for investigation, not automatically remove valid observations.

**Q12. What is a rolling mean?**  
A rolling mean calculates the average over a moving window. A 52-week window smooths weekly variation in this dataset.

**Q13. What is correlation?**  
Correlation measures the strength and direction of a linear association. It does not establish causation.

**Q14. Why use both Matplotlib and Seaborn?**  
Matplotlib provides detailed figure and axis control; Seaborn offers convenient statistical plots and works on top of Matplotlib.

**Q15. What is the main limitation of this dataset?**  
It ends in December 2001, so it is suitable for a historical EDA exercise, not for reporting current atmospheric CO₂ levels.

## 20. Submission checklist

- [ ] Extract the ZIP file.
- [ ] Open the project in VS Code.
- [ ] Install the dependencies in a virtual environment.
- [ ] Run `python run_analysis.py` successfully.
- [ ] Open the notebook and run all cells.
- [ ] Run the automated tests.
- [ ] Review all twelve figures.
- [ ] Read the EDA report and understand the cleaning decisions.
- [ ] Add your own name, internship ID and date to any college-specific cover page if required.
- [ ] Submit the source code, notebook, data/source note, cleaned data, report, figures and README as required by your internship coordinator.

## 21. Suggested daily internship report entry

**Week 4 — Pandas & Data Visualization:** Loaded and explored a historical real-world weekly atmospheric CO₂ dataset using Pandas. Inspected the DataFrame structure, column types and descriptive statistics. Identified 59 missing CO₂ readings, checked exact duplicate rows and duplicate dates, and handled missing values through time-based interpolation in a separate cleaned dataset while preserving the raw data. Created date-based features and performed descriptive, annual, monthly and decade-level statistical analysis. Reviewed possible outliers using the IQR method without automatically deleting valid time-series observations. Generated twelve visualizations using Matplotlib and Seaborn, including time-series plots, histograms, boxplots, scatter plots and heatmaps. Exported the cleaned dataset, summary tables, figures and EDA report. Executed the notebook and ran the automated tests.

## 22. References and attribution

- Statsmodels documentation, *Mauna Loa Weekly Atmospheric CO₂ Data*: https://www.statsmodels.org/stable/datasets/generated/co2.html
- NOAA Global Monitoring Laboratory, *Trends in Atmospheric Carbon Dioxide — Data*: https://www.gml.noaa.gov/ccgg/trends/data.html
- Historical citation listed in the Statsmodels dataset documentation: Keeling, C.D. and T.P. Whorf (2004), *Atmospheric CO₂ concentrations derived from flask air samples at sites in the SIO network*, in *Trends: A Compendium of Data on Global Change*, Carbon Dioxide Information Analysis Center, Oak Ridge National Laboratory.

**Academic integrity:** Understand and be able to explain the work. Cite the dataset and libraries. Follow any additional formatting or submission instructions provided by the internship coordinator.
