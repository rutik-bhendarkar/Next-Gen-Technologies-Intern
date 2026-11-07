from pathlib import Path
import json
from src.eda import run_eda

if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    result = run_eda(root, make_figures=True)
    print("\nWEEK 7 — EXPLORATORY DATA ANALYSIS")
    print("=" * 48)
    print(f"Raw rows: {result['raw_rows']}")
    print(f"Analysis rows after exact duplicate removal: {result['analysis_rows']}")
    print(f"Duplicate rows removed: {result['duplicate_rows_removed']}")
    print(f"Churn rate: {result['churn_rate_percent']:.2f}%")
    print(f"Visualizations generated: {result['visualization_count']}")
    print("Report: reports/Week7_EDA_Report.md")
    print("Tables: outputs/tables/")
    print("Figures: outputs/figures/")
