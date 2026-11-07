"""Week 7 end-to-end Exploratory Data Analysis for the synthetic churn dataset.

This is a descriptive analysis. The input data is synthetic; results are not
business evidence and do not establish causal relationships.
"""
from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency

ID_COLUMN = "CustomerID"
TARGET = "Churn"
NUMERIC = ["Age", "TenureMonths", "MonthlyCharges", "TotalCharges", "SupportCalls"]
CATEGORICAL = ["ContractType", "PaymentMethod", "InternetService", "PaperlessBilling"]
DOMAIN_LIMITS = {
    "Age": (18, 100),
    "TenureMonths": (0, 120),
    "MonthlyCharges": (0, 300),
    "TotalCharges": (0, None),
    "SupportCalls": (0, 30),
}
PALETTE = "Set2"


def load_data(path: str | Path) -> pd.DataFrame:
    """Load CSV and validate required columns."""
    frame = pd.read_csv(path)
    required = {ID_COLUMN, TARGET, *NUMERIC, *CATEGORICAL}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    return frame


def audit_and_prepare(raw: pd.DataFrame):
    """Remove exact duplicates and mark domain-impossible numeric values missing.

    No statistical imputation is performed here. Missingness remains visible so
    the EDA describes observed information instead of silently filling it.
    """
    duplicate_count = int(raw.duplicated().sum())
    data = raw.drop_duplicates().reset_index(drop=True).copy()
    original_missing = raw.isna().sum().astype(int)
    invalid_counts = {}
    for col, (low, high) in DOMAIN_LIMITS.items():
        invalid = data[col].notna()
        if low is not None:
            invalid &= data[col] < low
        if high is not None:
            invalid |= data[col].notna() & (data[col] > high)
        invalid_counts[col] = int(invalid.sum())
        data.loc[invalid, col] = np.nan
    return data, {
        "raw_rows": int(len(raw)),
        "raw_columns": int(raw.shape[1]),
        "exact_duplicate_rows": duplicate_count,
        "rows_after_exact_deduplication": int(len(data)),
        "raw_missing_by_column": {k: int(v) for k, v in original_missing.items()},
        "invalid_domain_values_marked_missing": invalid_counts,
        "analysis_missing_by_column_after_domain_validation": {k: int(v) for k, v in data.isna().sum().items()},
    }


def descriptive_statistics(data: pd.DataFrame) -> pd.DataFrame:
    """Return numeric descriptive statistics with explicit missing handling."""
    stats = data[NUMERIC].describe().T
    stats["median"] = data[NUMERIC].median(numeric_only=True)
    stats["missing_count"] = data[NUMERIC].isna().sum()
    stats["missing_percent"] = data[NUMERIC].isna().mean().mul(100)
    stats["skewness"] = data[NUMERIC].skew(numeric_only=True)
    stats["iqr"] = stats["75%"] - stats["25%"]
    return stats.reset_index(names="feature")


def categorical_frequency(data: pd.DataFrame) -> pd.DataFrame:
    records = []
    for col in CATEGORICAL + [TARGET]:
        counts = data[col].value_counts(dropna=False)
        total = len(data)
        for category, count in counts.items():
            label = "MISSING" if pd.isna(category) else str(category)
            records.append({"feature": col, "category": label, "count": int(count), "percent_of_rows": float(count / total * 100)})
    return pd.DataFrame(records)


def iqr_outlier_summary(data: pd.DataFrame) -> pd.DataFrame:
    records = []
    for col in NUMERIC:
        x = data[col].dropna()
        if x.empty:
            continue
        q1, q3 = x.quantile([0.25, 0.75])
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        count = int(((x < lower) | (x > upper)).sum())
        records.append({
            "feature": col, "q1": float(q1), "q3": float(q3), "iqr": float(iqr),
            "lower_fence": float(lower), "upper_fence": float(upper),
            "potential_outlier_count": count,
            "potential_outlier_percent_of_non_missing": float(count / len(x) * 100),
            "non_missing_count": int(len(x)),
        })
    return pd.DataFrame(records)


def churn_rate_by_category(data: pd.DataFrame) -> pd.DataFrame:
    records = []
    for col in CATEGORICAL:
        subset = data[[col, TARGET]].dropna()
        grouped = subset.groupby(col, observed=False)[TARGET].agg(
            customers="count", churned=lambda s: (s == "Yes").sum(),
            churn_rate=lambda s: (s == "Yes").mean() * 100,
        ).reset_index()
        grouped.insert(0, "feature", col)
        grouped = grouped.rename(columns={col: "category"})
        records.extend(grouped.to_dict(orient="records"))
    return pd.DataFrame(records)


def categorical_chi_square(data: pd.DataFrame) -> pd.DataFrame:
    records = []
    for col in CATEGORICAL:
        table = pd.crosstab(data[col], data[TARGET]).reindex(columns=["No", "Yes"], fill_value=0)
        if table.shape[0] < 2 or table.values.sum() == 0:
            continue
        chi2, p, dof, _ = chi2_contingency(table)
        n = table.values.sum()
        denom = max(1, min(table.shape[0]-1, table.shape[1]-1))
        cramers_v = float(np.sqrt(chi2 / (n * denom))) if n else 0.0
        records.append({"feature": col, "chi_square": float(chi2), "degrees_of_freedom": int(dof), "p_value": float(p), "cramers_v": cramers_v, "n": int(n)})
    return pd.DataFrame(records)


def build_summary(data: pd.DataFrame, audit: dict) -> dict:
    target_counts = data[TARGET].value_counts(dropna=False)
    numeric_target = data[NUMERIC].copy()
    numeric_target["ChurnBinary"] = data[TARGET].map({"No": 0, "Yes": 1})
    corr = numeric_target.corr(numeric_only=True)
    rates = churn_rate_by_category(data)
    return {
        "dataset": "Synthetic Customer Churn dataset continued from Week 6",
        "dataset_origin": "Artificially generated educational data; not real customer data",
        "audit": audit,
        "target_distribution": {str(k): int(v) for k, v in target_counts.items()},
        "target_churn_rate_percent": float((data[TARGET] == "Yes").mean() * 100),
        "numeric_correlations": corr.round(6).replace({np.nan: None}).to_dict(),
        "churn_rates_by_category": rates.to_dict(orient="records"),
        "interpretation_warning": "Patterns are descriptive and reflect synthetic label-generation logic; they are not causal or production evidence.",
    }


def _save(fig, path: Path):
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def create_visualizations(data: pd.DataFrame, raw: pd.DataFrame, outdir: str | Path) -> list[str]:
    out = Path(outdir); out.mkdir(parents=True, exist_ok=True)
    sns.set_theme(context="notebook", style="whitegrid")
    generated = []

    def save(fig, name):
        _save(fig, out/name); generated.append(name)

    # 01 target count
    fig, ax = plt.subplots(figsize=(7, 4.5))
    order = [x for x in ["No", "Yes"] if x in data[TARGET].dropna().unique()]
    sns.countplot(data=data, x=TARGET, order=order, hue=TARGET, palette=PALETTE, legend=False, ax=ax)
    ax.set_title("Target Distribution — Customer Churn"); ax.set_xlabel("Churn"); ax.set_ylabel("Number of customers")
    save(fig, "01_target_distribution_count.png")

    # 02 target percentage
    fig, ax = plt.subplots(figsize=(6, 5))
    shares = data[TARGET].value_counts(normalize=True).reindex(order)
    ax.pie(shares.values, labels=shares.index, autopct="%1.1f%%", startangle=90, colors=sns.color_palette(PALETTE, len(shares)))
    ax.set_title("Customer Churn Class Share")
    save(fig, "02_target_class_share.png")

    # 03 missingness raw
    fig, ax = plt.subplots(figsize=(10, 5))
    miss = raw.isna().sum().sort_values(ascending=False)
    sns.barplot(x=miss.index, y=miss.values, hue=miss.index, palette="crest", legend=False, ax=ax)
    ax.set_title("Missing Values by Feature — Raw Dataset"); ax.set_xlabel("Feature"); ax.set_ylabel("Missing rows")
    ax.tick_params(axis="x", rotation=35)
    save(fig, "03_missing_values_by_feature.png")

    # 04 numeric histogram panel
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    for ax, col in zip(axes.flat, NUMERIC):
        sns.histplot(data=data, x=col, kde=True, bins=25, color=sns.color_palette(PALETTE)[0], ax=ax)
        ax.set_title(f"Distribution of {col}")
    axes.flat[-1].axis("off")
    fig.suptitle("Univariate Analysis — Numeric Feature Distributions", y=1.02, fontsize=15)
    save(fig, "04_numeric_feature_distributions.png")

    # 05 numeric boxplot panel
    fig, ax = plt.subplots(figsize=(11, 5.5))
    long = data[NUMERIC].melt(var_name="Feature", value_name="Value")
    sns.boxplot(data=long, x="Feature", y="Value", hue="Feature", palette="Set3", legend=False, ax=ax)
    ax.set_title("Potential Outliers Across Numeric Features"); ax.tick_params(axis="x", rotation=25)
    save(fig, "05_numeric_boxplots.png")

    # 06 category frequencies
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for ax, col in zip(axes.flat, CATEGORICAL):
        order_cat = data[col].value_counts().index
        sns.countplot(data=data, y=col, order=order_cat, hue=col, palette="Set2", legend=False, ax=ax)
        ax.set_title(f"Frequency — {col}"); ax.set_xlabel("Number of customers"); ax.set_ylabel("")
    fig.suptitle("Univariate Analysis — Categorical Feature Frequencies", y=1.02, fontsize=15)
    save(fig, "06_categorical_feature_frequencies.png")

    # 07 numeric distributions by target
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for ax, col in zip(axes.flat, NUMERIC):
        sns.boxplot(data=data, x=TARGET, y=col, hue=TARGET, palette="Set2", legend=False, ax=ax)
        ax.set_title(f"{col} by Churn")
    axes.flat[-1].axis("off")
    fig.suptitle("Bivariate Analysis — Numeric Features vs Churn", y=1.02, fontsize=15)
    save(fig, "07_numeric_features_by_churn.png")

    # 08 monthly distribution by churn
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.violinplot(data=data, x=TARGET, y="MonthlyCharges", hue=TARGET, palette="Set2", legend=False, inner="quartile", ax=ax)
    ax.set_title("Monthly Charges Distribution by Churn")
    save(fig, "08_monthly_charges_violin_by_churn.png")

    # 09 monthly charges by contract and churn (three-way feature relationship)
    fig, ax = plt.subplots(figsize=(9, 5.2))
    sns.boxplot(data=data, x="ContractType", y="MonthlyCharges", hue=TARGET, palette="Set2", ax=ax)
    ax.set_title("Monthly Charges by Contract Type and Churn")
    ax.set_xlabel("Contract type")
    ax.set_ylabel("Monthly charges")
    ax.tick_params(axis="x", rotation=15)
    save(fig, "09_monthly_charges_by_contract_and_churn.png")

    # 10 contract rate
    rates = churn_rate_by_category(data)
    for i, col in enumerate(CATEGORICAL, start=10):
        sub = rates[rates.feature == col].sort_values("churn_rate", ascending=False)
        fig, ax = plt.subplots(figsize=(8, 4.8))
        sns.barplot(data=sub, x="category", y="churn_rate", hue="category", palette="Set2", legend=False, ax=ax)
        ax.set_title(f"Churn Rate by {col}"); ax.set_xlabel(col); ax.set_ylabel("Churn rate (%)")
        ax.tick_params(axis="x", rotation=20)
        ax.set_ylim(0, max(35, float(sub.churn_rate.max()) * 1.18 if not sub.empty else 35))
        save(fig, f"{i:02d}_churn_rate_by_{col.lower()}.png")

    # 14 tenure vs total charges
    fig, ax = plt.subplots(figsize=(8, 5.5))
    sns.scatterplot(data=data, x="TenureMonths", y="TotalCharges", hue=TARGET, alpha=0.75, palette="Set1", ax=ax)
    ax.set_title("Tenure vs Total Charges — Colored by Churn")
    save(fig, "14_tenure_vs_total_charges.png")

    # 15 monthly charges vs support calls
    fig, ax = plt.subplots(figsize=(8, 5.5))
    sns.scatterplot(data=data, x="MonthlyCharges", y="SupportCalls", hue=TARGET, alpha=0.75, palette="Set1", ax=ax)
    ax.set_title("Monthly Charges vs Support Calls — Colored by Churn")
    save(fig, "15_monthly_charges_vs_support_calls.png")

    # 16 correlation heatmap
    corr_data = data[NUMERIC].copy(); corr_data["ChurnBinary"] = data[TARGET].map({"No": 0, "Yes": 1})
    corr = corr_data.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(8, 6.5))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="vlag", center=0, vmin=-1, vmax=1, square=True, ax=ax)
    ax.set_title("Multivariate Analysis — Numeric Feature Correlation")
    save(fig, "16_correlation_heatmap.png")

    # 17 pairwise feature view
    pair_data = data[NUMERIC + [TARGET]].dropna().copy()
    if len(pair_data) > 250:
        pair_data = pair_data.sample(250, random_state=42)
    g = sns.pairplot(pair_data, vars=["Age", "TenureMonths", "MonthlyCharges", "SupportCalls"], hue=TARGET, diag_kind="hist", corner=True, plot_kws={"alpha": 0.65, "s": 24})
    g.fig.suptitle("Multivariate Relationships Among Selected Features", y=1.02)
    g.fig.savefig(out/"17_multivariate_pairplot.png", dpi=145, bbox_inches="tight")
    plt.close(g.fig); generated.append("17_multivariate_pairplot.png")

    # 18 age vs tenure relationship
    fig, ax = plt.subplots(figsize=(8, 5.5))
    sns.scatterplot(data=data, x="Age", y="TenureMonths", hue=TARGET, alpha=0.75, palette="Set1", ax=ax)
    ax.set_title("Age vs Tenure — Colored by Churn")
    save(fig, "18_age_vs_tenure.png")

    # 19 heatmap of category churn rates (crosstab)
    category_rate = pd.crosstab(data["ContractType"], data[TARGET], normalize="index") * 100
    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.heatmap(category_rate, annot=True, fmt=".1f", cmap="YlGnBu", vmin=0, vmax=100, ax=ax)
    ax.set_title("Contract Type Composition by Churn (%)"); ax.set_xlabel("Churn")
    save(fig, "19_contract_churn_crosstab_heatmap.png")

    return generated


def generate_report(data: pd.DataFrame, raw: pd.DataFrame, audit: dict, stats: pd.DataFrame,
                    outliers: pd.DataFrame, rates: pd.DataFrame, chi: pd.DataFrame,
                    summary: dict, report_path: str | Path):
    """Write a data-driven Markdown EDA report."""
    target_counts = data[TARGET].value_counts()
    churn_pct = summary["target_churn_rate_percent"]
    stats_table = stats[["feature", "count", "mean", "50%", "std", "min", "max", "missing_count", "skewness"]].copy()
    stats_table = stats_table.round(2).to_markdown(index=False)
    out_table = outliers.round(2).to_markdown(index=False)
    rate_table = rates.round(2).to_markdown(index=False)
    chi_table = chi.round(4).to_markdown(index=False) if not chi.empty else "No valid categorical tests."
    corr = pd.DataFrame(summary["numeric_correlations"]).round(3)
    corr_table = corr.to_markdown()
    strongest_rate = rates[rates.feature == "ContractType"].sort_values("churn_rate", ascending=False)
    contract_obs = "No contract categories were available for comparison."
    if len(strongest_rate):
        highest = strongest_rate.iloc[0]
        lowest = strongest_rate.iloc[-1]
        contract_obs = (f"The highest observed category churn rate among contract types is **{highest.category} ({highest.churn_rate:.1f}%)**, "
                        f"while the lowest is **{lowest.category} ({lowest.churn_rate:.1f}%)**.")
    tenure_stats = data.groupby(TARGET)["TenureMonths"].median().to_dict()
    charge_stats = data.groupby(TARGET)["MonthlyCharges"].median().to_dict()
    report = f"""# Week 7 — Exploratory Data Analysis (EDA) Report

## 1. Executive summary

This project performs end-to-end exploratory data analysis on the synthetic customer-churn dataset continued from Week 6. The work includes univariate, bivariate and multivariate analysis; distribution, correlation, outlier and feature-relationship analysis; at least 10 visualizations; and written observations.

- Raw records: **{len(raw):,}** across **{raw.shape[1]} columns**.
- Exact duplicate rows: **{audit['exact_duplicate_rows']}**.
- Unique records after exact deduplication: **{len(data):,}**.
- Customers labelled `Yes` for churn: **{int(target_counts.get('Yes', 0))}** ({churn_pct:.1f}% of analyzed records).
- Customers labelled `No`: **{int(target_counts.get('No', 0))}**.
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

The dataset has **{raw.shape[1]} columns**: an identifier (`CustomerID`), five numeric attributes (`Age`, `TenureMonths`, `MonthlyCharges`, `TotalCharges`, `SupportCalls`), four categorical attributes (`ContractType`, `PaymentMethod`, `InternetService`, `PaperlessBilling`), and the target (`Churn`). It is an **artificially generated educational dataset**, not a downloaded company dataset and not real customer information. Its churn labels are generated using a probabilistic rule involving selected customer features. See `data/DATASET_CARD.md`.

## 4. Methodology and data-quality audit

1. Loaded the raw CSV without modifying it.
2. Counted missing values and exact duplicate rows.
3. Removed **{audit['exact_duplicate_rows']}** exact duplicate rows for the analytical working copy.
4. Applied the same transparent domain checks used in Week 6: Age 18–100, TenureMonths 0–120, MonthlyCharges 0–300, TotalCharges non-negative, SupportCalls 0–30. Values outside these limits were marked missing in the analysis copy, not silently retained as valid measurements.
5. Did **not** impute missing values for descriptive EDA. Every statistic is calculated from available valid observations, and sample counts are shown.
6. Did **not** automatically delete IQR outliers. They are flagged for review because an unusual valid customer value is not necessarily an error.

### Data audit details

- Raw missing values by column: `{json.dumps(audit['raw_missing_by_column'], sort_keys=True)}`
- Impossible values set to missing: `{json.dumps(audit['invalid_domain_values_marked_missing'], sort_keys=True)}`
- Remaining missing values in analysis copy: `{json.dumps(audit['analysis_missing_by_column_after_domain_validation'], sort_keys=True)}`

## 5. Univariate analysis

### 5.1 Numeric descriptive statistics

{stats_table}

The histograms and boxplots in `outputs/figures/` show the shape, spread, skew and potential extreme observations of each numeric feature. Mean and median are both reported because skewed distributions can make the mean sensitive to high values.

### 5.2 Categorical distributions

Frequency counts and percentages for contract, payment method, internet service, paperless billing and churn are saved in `outputs/tables/categorical_frequencies.csv`.

## 6. Bivariate analysis

### 6.1 Numeric features vs churn

| Feature | Median for Churn = No | Median for Churn = Yes |
|---|---:|---:|
| TenureMonths | {tenure_stats.get('No', float('nan')):.2f} | {tenure_stats.get('Yes', float('nan')):.2f} |
| MonthlyCharges | {charge_stats.get('No', float('nan')):.2f} | {charge_stats.get('Yes', float('nan')):.2f} |

The boxplots and violin plot compare numeric feature distributions across the target classes. These comparisons are descriptive and do not show that a feature causes churn.

### 6.2 Categorical features vs churn

{rate_table}

{contract_obs}

Category-level churn rates are computed using records with non-missing values for both the category and target. The number of customers in each category is included in the saved table.

## 7. Multivariate and correlation analysis

Pearson correlations for numeric features and the binary-encoded churn target (`No = 0`, `Yes = 1`) are shown below.

{corr_table}

The selected-feature pairplot and scatter plots help inspect joint distributions and possible patterns that are not visible in one-variable summaries. Correlation describes linear association; it does not establish causation, and a low Pearson value does not rule out non-linear relationships.

## 8. Outlier analysis

The Interquartile Range (IQR) rule defines potential outliers as values below `Q1 - 1.5 × IQR` or above `Q3 + 1.5 × IQR`.

{out_table}

Potential outliers are retained in the EDA dataset. They should be investigated in context before any transformation or deletion. IQR flags are diagnostic indicators, not proof of data errors.

## 9. Categorical association test (supplementary)

A chi-square test of independence and Cramér's V effect-size estimate are included as supplementary descriptive diagnostics for categorical features against churn.

{chi_table}

The dataset is synthetic, and the tests are not evidence of real-world population effects. Small sample sizes, missing categories and the artificial label-generation process limit inferential interpretation.

## 10. Key findings

1. The cleaned analytical copy contains **{len(data)} distinct rows** after exact duplicate removal.
2. The observed churn share is **{churn_pct:.1f}%** in this synthetic sample.
3. Contract category churn rates differ in this generated sample: {contract_obs}
4. Median tenure is **{tenure_stats.get('No', float('nan')):.1f} months** for `No` and **{tenure_stats.get('Yes', float('nan')):.1f} months** for `Yes` in the available observations.
5. Median monthly charges are **{charge_stats.get('No', float('nan')):.2f}** for `No` and **{charge_stats.get('Yes', float('nan')):.2f}** for `Yes`.
6. Correlation heatmap indicates that tenure and total charges have a comparatively strong positive linear relationship in this dataset (Pearson r = **{summary['numeric_correlations']['TenureMonths']['TotalCharges']:.3f}**).

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
"""
    Path(report_path).write_text(report, encoding='utf-8')


def run_eda(project_root: str | Path, make_figures: bool = True) -> dict:
    root = Path(project_root)
    raw = load_data(root / "data/raw/customer_churn_raw.csv")
    data, audit = audit_and_prepare(raw)
    data.to_csv(root / "data/processed/customer_churn_eda_audited.csv", index=False)

    stats = descriptive_statistics(data)
    cats = categorical_frequency(data)
    outliers = iqr_outlier_summary(data)
    rates = churn_rate_by_category(data)
    chi = categorical_chi_square(data)
    summary = build_summary(data, audit)

    (root / "outputs").mkdir(exist_ok=True)
    (root / "outputs/summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    (root / "outputs/audit_report.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    stats.to_csv(root / "outputs/tables/numeric_descriptive_statistics.csv", index=False)
    cats.to_csv(root / "outputs/tables/categorical_frequencies.csv", index=False)
    outliers.to_csv(root / "outputs/tables/iqr_outlier_summary.csv", index=False)
    rates.to_csv(root / "outputs/tables/churn_rate_by_category.csv", index=False)
    chi.to_csv(root / "outputs/tables/categorical_chi_square.csv", index=False)
    corr = data[NUMERIC].copy(); corr["ChurnBinary"] = data[TARGET].map({"No": 0, "Yes": 1})
    corr.corr(numeric_only=True).to_csv(root / "outputs/tables/correlation_matrix.csv")

    figures = create_visualizations(data, raw, root / "outputs/figures") if make_figures else []
    generate_report(data, raw, audit, stats, outliers, rates, chi, summary, root / "reports/Week7_EDA_Report.md")

    result = {
        "raw_rows": len(raw),
        "analysis_rows": len(data),
        "columns": len(raw.columns),
        "duplicate_rows_removed": audit["exact_duplicate_rows"],
        "churn_rate_percent": summary["target_churn_rate_percent"],
        "visualizations": figures,
        "visualization_count": len(figures),
        "summary": summary,
    }
    (root / "outputs/run_result.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    return result
