# Week 4 – Pandas & Data Visualization
## Exploratory Data Analysis of Mauna Loa Weekly Atmospheric CO₂

**Internship phase:** Phase 1 – Python & Data Science Fundamentals  
**Tools:** Python, Pandas, NumPy, Matplotlib, Seaborn and Jupyter Notebook  
**Dataset period:** 29 March 1958 to 29 December 2001  
**Dataset type:** Historical observational time series  
**Measurement:** Atmospheric carbon dioxide concentration, parts per million (ppm)

---

## 1. Abstract

This project performs exploratory data analysis (EDA) on historical weekly atmospheric carbon dioxide measurements from Mauna Loa Observatory, Hawaii. Pandas is used to load, inspect, clean, transform and summarize the data. NumPy supports numerical calculations, while Matplotlib and Seaborn create visualizations. The raw CSV contains 2,284 weekly date records and 59 missing CO₂ readings. The cleaning workflow preserves the original file, checks for duplicate records and dates, and uses time-based linear interpolation to fill missing values in a separate cleaned dataset. Twelve visualizations are generated, exceeding the internship requirement of at least eight.

The historical series shows a clear long-term increase: the annual mean is approximately 315.19 ppm in 1958 and 370.87 ppm in 2001. This report describes patterns in the packaged historical dataset; it is not a current atmospheric monitoring report and does not make causal claims.

## 2. Introduction

Exploratory Data Analysis is the process of examining a dataset to understand its structure, quality, distributions, relationships and potential patterns before building a model or making decisions. A useful EDA workflow begins with data inspection, proceeds through appropriate cleaning, and ends with statistical summaries and visual explanations.

A time-series dataset requires special care because rows are ordered by date. Missing readings should be considered in relation to their neighboring observations, and valid long-term changes should not be removed merely because they appear far from the overall median.

## 3. Problem statement

Analyze the historical weekly Mauna Loa CO₂ dataset using Pandas and visualization libraries. The analysis must:

1. Load and inspect the dataset.
2. identify missing values and handle them transparently.
3. Detect and remove duplicate records where present.
4. Calculate descriptive statistics.
5. Generate at least eight visualizations.
6. Record key observations and limitations.

## 4. Objectives

- Practice CSV loading and DataFrame inspection with Pandas.
- Understand column types, missing values and duplicate records.
- Clean the data without modifying the raw source file.
- Create useful date-based features.
- Calculate descriptive statistics and grouped summaries.
- Use Matplotlib and Seaborn to communicate the findings.
- Export cleaned data, summary tables, charts and documentation.

## 5. Dataset description and source

The packaged data is the historical **Mauna Loa Weekly Atmospheric CO₂** series distributed with the Statsmodels datasets collection. It describes atmospheric CO₂ from continuous air samples at Mauna Loa Observatory, Hawaii, for the period March 1958 through December 2001. The dataset documentation describes weekly averages and identifies the measurement variable as CO₂ concentration in ppmv. The CSV in this project is a local snapshot so the project can run without downloading the dataset again.

| Field | Description |
|---|---|
| `date` | Date of the weekly observation |
| `co2_ppm` | Atmospheric CO₂ concentration in parts per million |

**Source documentation:** Statsmodels, *Mauna Loa Weekly Atmospheric CO₂ Data* (historical data description and source citation). NOAA Global Monitoring Laboratory also publishes Mauna Loa CO₂ trend data and describes the available weekly, monthly and annual records.

## 6. Technologies used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| Pandas | Loading, cleaning, time features, grouping and export |
| NumPy | Numerical operations |
| Matplotlib | Line, bar and customized charts |
| Seaborn | Statistical plots and heatmaps |
| Jupyter Notebook | Interactive and documented analysis |
| unittest | Basic automated checks |

## 7. Methodology

### 7.1 Data loading

The CSV is loaded with `pandas.read_csv()`. The `date` column is parsed as a datetime field so the records can be sorted and analyzed by year, month and week.

### 7.2 Initial inspection

The notebook checks:

- Number of rows and columns.
- Column names and data types.
- First and last records.
- Non-null counts and memory usage.
- Descriptive statistics.
- Missing-value counts and percentages.

### 7.3 Duplicate handling

The analysis checks exact duplicate rows and duplicate dates. The packaged raw dataset contains **0 exact duplicate rows and 0 duplicate dates**. The cleaning function nevertheless includes duplicate-removal logic (`drop_duplicates`) so it can handle duplicate records if a similar file is used later. Since there were no duplicates, no rows were removed for duplication.

### 7.4 Missing-value handling

The raw data contains **59 missing `co2_ppm` values**, which is approximately **2.58%** of the 2,284 rows. The `date` field is complete.

Because this is an ordered weekly time series, the project uses Pandas time-based linear interpolation:

```python
df_indexed = df_work.set_index("date")
df_indexed["co2_ppm"] = df_indexed["co2_ppm"].interpolate(
    method="time", limit_direction="both"
)
df_clean = df_indexed.reset_index()
```

The interpolation estimates missing values between dated observations. It does not create new measurements. The raw CSV remains unchanged, and the cleaned result is saved separately at `data/processed/mauna_loa_co2_weekly_cleaned.csv`.

### 7.5 Feature engineering

The cleaned dataset is extended with:

- `year`
- `month`
- `month_name`
- `quarter`
- `decade`
- `week_number`
- `rolling_52_week_mean`

These features support calendar-based summaries and smoothing.

### 7.6 Statistical analysis

The project calculates count, mean, standard deviation, minimum, 25th percentile, median, 75th percentile and maximum. It also calculates variance, range, skewness and kurtosis. Grouped summaries are created by year, month and decade.

### 7.7 Outlier analysis

The IQR rule is applied:

- `IQR = Q3 - Q1`
- Lower bound = `Q1 - 1.5 × IQR`
- Upper bound = `Q3 + 1.5 × IQR`

The analysis flags values outside these limits for review. It does not automatically delete them. In a time series with a sustained trend, a value that is distant from the overall distribution may still be a valid observation.

## 8. Results

### 8.1 Data-quality summary

| Check | Result |
|---|---:|
| Raw rows | 2,284 |
| Raw columns | 2 |
| Missing CO₂ readings before cleaning | 59 |
| Missing percentage | 2.58% |
| Exact duplicate rows | 0 |
| Duplicate dates | 0 |
| Duplicate rows removed | 0 |
| Missing CO₂ readings after cleaning | 0 |
| Rows in cleaned dataset | 2,284 |

### 8.2 Descriptive statistics

The following statistics are calculated from the original observed (non-missing) CO₂ values, not from the interpolated estimates.

| Statistic | CO₂ (ppm) |
|---|---:|
| Count of observed readings | 2,225 |
| Mean | 340.14 |
| Median | 338.30 |
| Standard deviation | 17.00 |
| Minimum | 313.00 |
| Maximum | 373.90 |

### 8.3 Long-term change

The annual mean was **315.19 ppm in 1958** and **370.87 ppm in 2001** in the packaged dataset. The difference between these endpoint annual means is approximately **55.68 ppm**, or **17.67%** relative to the 1958 annual mean. This is a comparison of the first and last annual averages in this historical file; it is not an estimate of the present-day concentration.

### 8.4 Calendar-month profile

Across the full historical period, the highest average by calendar month is approximately **342.19 ppm in May**, and the lowest is approximately **336.71 ppm in October**. Because these are pooled averages across a period in which the overall level rises, they are descriptive summaries and should not be interpreted as an isolated seasonal effect. The month-wise plots help inspect the recurring pattern, while a dedicated seasonal decomposition would be required for a more rigorous separation of trend and seasonality.

### 8.5 IQR outlier check

| Measure | Result (ppm) |
|---|---:|
| Q1 | 323.98 |
| Q3 | 354.50 |
| IQR | 30.53 |
| Lower bound | 278.19 |
| Upper bound | 400.29 |
| IQR-flagged observations | 0 |

No observations fall outside these IQR boundaries. No data points were removed during outlier review.

## 9. Visualizations

Twelve visualizations are saved in `outputs/figures/`.

| No. | File | Purpose / observation supported |
|---:|---|---|
| 1 | `01_raw_weekly_co2_trend.png` | Raw weekly time series; missing gaps remain visible. |
| 2 | `02_missing_values.png` | Missing-value count before cleaning. |
| 3 | `03_co2_distribution.png` | Distribution of observed CO₂ concentrations. |
| 4 | `04_co2_by_decade_boxplot.png` | Comparison of CO₂ distributions across decades. |
| 5 | `05_annual_mean_trend.png` | Long-term annual mean trend. |
| 6 | `06_monthly_seasonal_profile.png` | Average concentration by calendar month. |
| 7 | `07_rolling_52_week_mean.png` | Weekly values compared with the 52-week rolling mean. |
| 8 | `08_year_month_heatmap.png` | Concentration pattern by year and month. |
| 9 | `09_annual_mean_change.png` | Year-over-year change in annual mean. |
| 10 | `10_year_vs_co2_scatter.png` | Individual weekly values against year with a linear trend line. |
| 11 | `11_numeric_correlation_heatmap.png` | Correlations among CO₂ and derived numeric time features. |
| 12 | `12_monthly_boxplot.png` | Distribution and spread by calendar month. |

## 10. Key observations

1. The historical file covers weekly readings from March 1958 through December 2001.
2. The raw dataset contains 59 missing CO₂ readings; interpolation fills these in the cleaned copy while preserving the original CSV.
3. No exact duplicate records or duplicate dates were found in the supplied dataset.
4. The annual average concentration increases between the first and last years in the file. The decade boxplots and annual trend make this long-term pattern visible.
5. The 52-week rolling mean smooths weekly variations and emphasizes the long-term direction.
6. Monthly averages and month-wise boxplots show differences among calendar months. A more advanced seasonal decomposition would be needed to isolate seasonal variation from the long-term trend.
7. The IQR method flagged no observations using the calculated global bounds. This does not prove that every observation is error-free; domain and time-context checks remain useful.
8. Twelve visualizations were produced, exceeding the requirement of at least eight.

## 11. Limitations

- The bundled series ends on 29 December 2001 and must not be presented as current data.
- Interpolated values are estimates, not direct observations.
- Pooled monthly averages can mix long-term trend and seasonality.
- Correlation among a time variable and concentration does not establish causation.
- The IQR method is a simple global rule and is not a complete time-series anomaly detector.

## 12. Conclusion

The Week 4 task has been implemented using Pandas, Matplotlib and Seaborn. The work includes data loading and structural inspection, missing-value analysis and treatment, duplicate detection, statistical summaries, time-based feature creation, IQR outlier review, twelve visualizations, exported CSV tables and written observations. The project keeps raw and cleaned data separate and documents the assumptions used in cleaning and interpretation.

## 13. References

1. Statsmodels documentation, *Mauna Loa Weekly Atmospheric CO₂ Data*: https://www.statsmodels.org/stable/datasets/generated/co2.html (historical dataset description; source attribution to Keeling and Whorf, 2004).
2. NOAA Global Monitoring Laboratory, *Trends in Atmospheric Carbon Dioxide – Data*: https://www.gml.noaa.gov/ccgg/trends/data.html (official information about Mauna Loa CO₂ trend datasets and data formats).
3. Keeling, C.D. and T.P. Whorf (2004). *Atmospheric CO₂ concentrations derived from flask air samples at sites in the SIO network*. In *Trends: A Compendium of Data on Global Change*, Carbon Dioxide Information Analysis Center, Oak Ridge National Laboratory, U.S. Department of Energy.
