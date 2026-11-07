"""Week 4 Pandas and data visualization analysis of Mauna Loa weekly CO2 data."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA = PROJECT_ROOT / "data" / "raw" / "mauna_loa_co2_weekly.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"
TABLE_DIR = OUTPUT_DIR / "tables"

sns.set_theme(style="whitegrid", context="notebook")


def load_raw_data(path: Path | str = RAW_DATA) -> pd.DataFrame:
    """Load the bundled CSV and parse its date column."""
    data = pd.read_csv(path, parse_dates=["date"])
    required = {"date", "co2_ppm"}
    missing_columns = required.difference(data.columns)
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing_columns)}")
    data["co2_ppm"] = pd.to_numeric(data["co2_ppm"], errors="coerce")
    return data


def clean_data(raw: pd.DataFrame) -> tuple[pd.DataFrame, Dict[str, int]]:
    """Remove duplicate records and interpolate missing weekly readings by time."""
    data = raw.copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data["co2_ppm"] = pd.to_numeric(data["co2_ppm"], errors="coerce")

    audit = {
        "rows_before": int(len(data)),
        "duplicate_rows_before": int(data.duplicated().sum()),
        "duplicate_dates_before": int(data["date"].duplicated().sum()),
        "invalid_dates_before": int(data["date"].isna().sum()),
        "missing_co2_before": int(data["co2_ppm"].isna().sum()),
    }

    data = data.dropna(subset=["date"])
    audit["invalid_date_rows_removed"] = int(audit["rows_before"] - len(data))
    rows_before_duplicate_removal = len(data)
    data = data.drop_duplicates(subset=["date"], keep="first")
    audit["duplicate_dates_removed"] = int(rows_before_duplicate_removal - len(data))
    data = data.sort_values("date").set_index("date")

    missing_before = int(data["co2_ppm"].isna().sum())
    # Time interpolation estimates gaps between adjacent weekly observations.
    data["co2_ppm"] = data["co2_ppm"].interpolate(method="time", limit_direction="both")
    # Defensive fallback in case a different input file has an all-missing series.
    if data["co2_ppm"].isna().any():
        raise ValueError("CO2 values still contain missing data after interpolation; inspect the input file.")

    data = data.reset_index()
    audit.update({
        "rows_after": int(len(data)),
        "duplicate_rows_removed": int(audit["duplicate_dates_removed"]),
        "missing_co2_filled": missing_before,
        "missing_co2_after": int(data["co2_ppm"].isna().sum()),
    })
    return data, audit


def add_time_features(data: pd.DataFrame) -> pd.DataFrame:
    """Create useful time-based columns for grouped analysis."""
    result = data.copy()
    result["date"] = pd.to_datetime(result["date"])
    result["year"] = result["date"].dt.year
    result["month"] = result["date"].dt.month
    result["month_name"] = result["date"].dt.month_name()
    result["quarter"] = result["date"].dt.quarter
    result["decade"] = (result["year"] // 10) * 10
    result["week_number"] = result["date"].dt.isocalendar().week.astype(int)
    result = result.sort_values("date").reset_index(drop=True)
    result["rolling_52_week_mean"] = result["co2_ppm"].rolling(window=52, min_periods=26).mean()
    return result


def compute_tables(data: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Create descriptive, annual, monthly, and decade summary tables."""
    numeric_summary = data[["co2_ppm"]].describe().T
    annual = data.groupby("year", as_index=False).agg(
        annual_mean_ppm=("co2_ppm", "mean"),
        annual_median_ppm=("co2_ppm", "median"),
        annual_std_ppm=("co2_ppm", "std"),
        observations=("co2_ppm", "size"),
    )
    annual["year_over_year_change_ppm"] = annual["annual_mean_ppm"].diff()
    annual["year_over_year_change_percent"] = annual["annual_mean_ppm"].pct_change() * 100

    monthly = data.groupby("month", as_index=False).agg(
        mean_co2_ppm=("co2_ppm", "mean"),
        median_co2_ppm=("co2_ppm", "median"),
        observations=("co2_ppm", "size"),
    )
    decade = data.groupby("decade", as_index=False).agg(
        mean_co2_ppm=("co2_ppm", "mean"),
        min_co2_ppm=("co2_ppm", "min"),
        max_co2_ppm=("co2_ppm", "max"),
        observations=("co2_ppm", "size"),
    )
    return {"numeric_summary": numeric_summary, "annual_summary": annual,
            "monthly_summary": monthly, "decade_summary": decade}


def detect_iqr_outliers(series: pd.Series) -> Dict[str, float | int]:
    """Report IQR outlier boundaries and counts without automatically deleting points."""
    q1 = float(series.quantile(0.25))
    q3 = float(series.quantile(0.75))
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    count = int(((series < lower) | (series > upper)).sum())
    return {"q1": q1, "q3": q3, "iqr": iqr,
            "lower_bound": lower, "upper_bound": upper, "outlier_count": count}


def generate_visualizations(data: pd.DataFrame, figure_dir: Path | str = FIGURE_DIR, raw_data: pd.DataFrame | None = None) -> List[Path]:
    """Generate and save 12 EDA visualizations."""
    figure_dir = Path(figure_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)
    figures: List[Path] = []
    raw_for_quality = raw_data.copy() if raw_data is not None else data.copy()
    raw_for_quality["date"] = pd.to_datetime(raw_for_quality["date"], errors="coerce")

    def save(fig: plt.Figure, filename: str) -> None:
        path = figure_dir / filename
        fig.tight_layout()
        fig.savefig(path, dpi=160, bbox_inches="tight")
        figures.append(path)
        plt.close(fig)

    # 1. Raw trend with gaps visible.
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(raw_for_quality["date"], raw_for_quality["co2_ppm"], linewidth=1.1)
    ax.set(title="Weekly Atmospheric CO2 at Mauna Loa (Raw Observations; gaps retained)", xlabel="Date", ylabel="CO2 concentration (ppm)")
    save(fig, "01_raw_weekly_co2_trend.png")

    # 2. Missing-value counts.
    fig, ax = plt.subplots(figsize=(7, 4))
    missing_counts = raw_for_quality.isna().sum().sort_values(ascending=False)
    ax.bar(missing_counts.index.astype(str), missing_counts.values)
    ax.set(title="Missing Values by Column (Before Cleaning)", xlabel="Column", ylabel="Missing row count")
    save(fig, "02_missing_values.png")

    # 3. Distribution.
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(data=data, x="co2_ppm", bins=35, kde=True, color="steelblue", ax=ax)
    ax.set(title="Distribution of Weekly CO2 Concentrations", xlabel="CO2 concentration (ppm)", ylabel="Frequency")
    save(fig, "03_co2_distribution.png")

    # 4. Box plot by decade.
    fig, ax = plt.subplots(figsize=(10, 5))
    decade_order = sorted(data["decade"].unique())
    sns.boxplot(data=data, x="decade", y="co2_ppm", order=decade_order, color="steelblue", ax=ax)
    ax.set(title="CO2 Distribution by Decade", xlabel="Decade beginning", ylabel="CO2 concentration (ppm)")
    save(fig, "04_co2_by_decade_boxplot.png")

    # 5. Annual average trend.
    annual = data.groupby("year", as_index=False)["co2_ppm"].mean()
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.lineplot(data=annual, x="year", y="co2_ppm", marker="o", markersize=3, color="steelblue", ax=ax)
    ax.set(title="Annual Mean CO2 Concentration", xlabel="Year", ylabel="Annual mean CO2 (ppm)")
    save(fig, "05_annual_mean_trend.png")

    # 6. Seasonal monthly profile.
    month_order = list(range(1, 13))
    monthly = data.groupby("month", as_index=False)["co2_ppm"].mean()
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.lineplot(data=monthly, x="month", y="co2_ppm", marker="o", color="steelblue", ax=ax)
    ax.set_xticks(month_order)
    ax.set(title="Average CO2 by Calendar Month", xlabel="Month", ylabel="Mean CO2 (ppm)")
    save(fig, "06_monthly_seasonal_profile.png")

    # 7. 52-week rolling mean.
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(data["date"], data["co2_ppm"], alpha=0.30, linewidth=0.8, label="Weekly reading")
    rolling = data["co2_ppm"].rolling(52, min_periods=26).mean()
    ax.plot(data["date"], rolling, linewidth=2, label="52-week rolling mean")
    ax.set(title="Weekly CO2 and 52-Week Rolling Mean", xlabel="Date", ylabel="CO2 concentration (ppm)")
    ax.legend()
    save(fig, "07_rolling_52_week_mean.png")

    # 8. Year-month heatmap.
    pivot = data.pivot_table(index="year", columns="month", values="co2_ppm", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(13, 10))
    sns.heatmap(pivot, cmap="viridis", cbar_kws={"label": "Mean CO2 (ppm)"}, ax=ax)
    ax.set(title="Year-Month Heatmap of Mean CO2", xlabel="Month", ylabel="Year")
    save(fig, "08_year_month_heatmap.png")

    # 9. Annual change.
    annual["annual_change"] = annual["co2_ppm"].diff()
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(annual.dropna()["year"].astype(str), annual.dropna()["annual_change"])
    ax.set(title="Year-over-Year Change in Annual Mean CO2", xlabel="Year", ylabel="Change in annual mean (ppm)")
    ax.tick_params(axis="x", labelrotation=90)
    save(fig, "09_annual_mean_change.png")

    # 10. Scatter: time and concentration.
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.scatterplot(data=data, x="year", y="co2_ppm", alpha=0.45, s=18, color="steelblue", ax=ax)
    sns.regplot(data=data, x="year", y="co2_ppm", scatter=False, ci=None, color="red", ax=ax)
    ax.set(title="CO2 Concentration versus Year", xlabel="Year", ylabel="CO2 concentration (ppm)")
    save(fig, "10_year_vs_co2_scatter.png")

    # 11. Correlation heatmap for numeric features.
    corr_columns = ["co2_ppm", "year", "month", "quarter", "week_number"]
    corr = data[corr_columns].corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set(title="Correlation Matrix of Numeric Time Features")
    save(fig, "11_numeric_correlation_heatmap.png")

    # 12. Month-wise boxplot.
    fig, ax = plt.subplots(figsize=(11, 5))
    sns.boxplot(data=data, x="month_name", y="co2_ppm",
                order=[pd.Timestamp(2000, m, 1).month_name() for m in range(1, 13)], color="steelblue", ax=ax)
    ax.set(title="Weekly CO2 Distribution by Month", xlabel="Month", ylabel="CO2 concentration (ppm)")
    ax.tick_params(axis="x", labelrotation=35)
    save(fig, "12_monthly_boxplot.png")

    return figures


def run_analysis() -> Dict[str, object]:
    """Run data loading, cleaning, summaries, charts, and save the outputs."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    raw = load_raw_data()
    clean, audit = clean_data(raw)
    featured = add_time_features(clean)
    tables = compute_tables(featured)
    outliers = detect_iqr_outliers(featured["co2_ppm"])
    figures = generate_visualizations(featured, raw_data=raw)

    featured.to_csv(PROCESSED_DIR / "mauna_loa_co2_weekly_cleaned.csv", index=False)
    for name, table in tables.items():
        table.to_csv(TABLE_DIR / f"{name}.csv")
    featured.groupby(["year", "month"], as_index=False)["co2_ppm"].mean().to_csv(TABLE_DIR / "year_month_means.csv", index=False)

    raw_nonnull = raw["co2_ppm"].dropna()
    annual = tables["annual_summary"]
    first = float(annual.iloc[0]["annual_mean_ppm"])
    last = float(annual.iloc[-1]["annual_mean_ppm"])
    summary: Dict[str, object] = {
        "dataset": "Mauna Loa weekly atmospheric CO2",
        "source_period": f"{featured['date'].min().date()} to {featured['date'].max().date()}",
        "rows_raw": int(len(raw)),
        "columns_raw": int(raw.shape[1]),
        "raw_missing_co2": int(raw["co2_ppm"].isna().sum()),
        "raw_duplicate_rows": int(raw.duplicated().sum()),
        "raw_duplicate_dates": int(raw["date"].duplicated().sum()),
        "rows_cleaned": int(len(featured)),
        "missing_after_cleaning": int(featured["co2_ppm"].isna().sum()),
        "co2_mean_ppm": float(raw_nonnull.mean()),
        "co2_median_ppm": float(raw_nonnull.median()),
        "co2_std_ppm": float(raw_nonnull.std()),
        "co2_min_ppm": float(raw_nonnull.min()),
        "co2_max_ppm": float(raw_nonnull.max()),
        "first_annual_mean_ppm": first,
        "last_annual_mean_ppm": last,
        "annual_mean_change_ppm": last - first,
        "annual_mean_change_percent": ((last - first) / first) * 100,
        "iqr_outlier_analysis": outliers,
        "figures_created": len(figures),
        "figure_files": [p.name for p in figures],
        "cleaning_audit": audit,
        "note": "IQR outliers are reported, not removed, because valid long-term concentration changes should not be discarded automatically.",
    }
    (OUTPUT_DIR / "summary_metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (OUTPUT_DIR / "cleaning_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    result = run_analysis()
    print(json.dumps(result, indent=2))
