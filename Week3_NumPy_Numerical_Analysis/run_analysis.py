from pathlib import Path
import json
import csv
import numpy as np
from src.numpy_analysis import load_dataset, calculate_analysis

ROOT = Path(__file__).resolve().parent
data = load_dataset(ROOT / "data" / "retail_sales.csv")
result = calculate_analysis(data)
out = ROOT / "outputs"
out.mkdir(exist_ok=True)
summary = {
    "dataset": "Synthetic retail sales dataset",
    "record_count": result["record_count"],
    "total_units_sold": int(np.sum(data["units_sold"])),
    "total_net_revenue_inr": float(np.sum(result["net_revenue"])),
    "overall_return_rate": float(np.sum(data["returns"]) / np.sum(data["units_sold"])),
    "statistics": result["statistics"],
    "highest_revenue_month": int(np.argmax(result["monthly_revenue"]) + 1),
    "lowest_revenue_month": int(np.argmin(result["monthly_revenue"]) + 1),
}
(out / "summary_metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
with (out / "monthly_revenue.csv").open("w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["month", "net_revenue_inr", "month_over_month_change_inr", "cumulative_revenue_inr"])
    for i, value in enumerate(result["monthly_revenue"]):
        change = "" if i == 0 else round(float(result["monthly_change"][i-1]), 2)
        writer.writerow([i+1, round(float(value), 2), change, round(float(result["cumulative_revenue"][i]), 2)])
np.savetxt(out / "correlation_matrix.csv", np.corrcoef(np.column_stack([data[k] for k in ("units_sold", "unit_price_inr", "discount_pct", "marketing_spend_inr")]), rowvar=False), delimiter=",", fmt="%.6f")
print("WEEK 3 NUMPY ANALYSIS COMPLETED")
print(f"Records: {result['record_count']}")
print(f"Total net revenue: INR {np.sum(result['net_revenue']):,.2f}")
print(f"Average net revenue per record: INR {result['statistics']['mean_net_revenue_inr']:,.2f}")
print(f"Outputs saved to: {out}")
