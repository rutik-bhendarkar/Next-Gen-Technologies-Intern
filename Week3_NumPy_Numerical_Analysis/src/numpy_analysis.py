"""NumPy analysis functions for the Week 3 internship assignment."""
from __future__ import annotations
import csv
from pathlib import Path
import numpy as np

NUMERIC_FIELDS = ("month", "units_sold", "unit_price_inr", "discount_pct",
                  "marketing_spend_inr", "returns", "customer_rating")

def load_dataset(path: str | Path) -> dict[str, np.ndarray]:
    """Load CSV columns into NumPy arrays without pandas."""
    columns = {name: [] for name in NUMERIC_FIELDS}
    with Path(path).open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = set(NUMERIC_FIELDS)
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Dataset is missing columns: {sorted(missing)}")
        for row in reader:
            for name in NUMERIC_FIELDS:
                columns[name].append(float(row[name]))
    if not columns["month"]:
        raise ValueError("Dataset is empty")
    return {name: np.asarray(values, dtype=float) for name, values in columns.items()}

def calculate_analysis(data: dict[str, np.ndarray]) -> dict:
    """Calculate required numerical, statistical, and matrix results."""
    units = data["units_sold"]
    price = data["unit_price_inr"]
    discount = data["discount_pct"]
    returns = data["returns"]
    month = data["month"].astype(int)
    marketing = data["marketing_spend_inr"]
    rating = data["customer_rating"]

    gross = units * price
    discount_amount = gross * discount
    net_revenue = gross - discount_amount
    revenue_per_unit = np.divide(net_revenue, units, out=np.zeros_like(net_revenue), where=units != 0)
    return_rate = np.divide(returns, units, out=np.zeros_like(returns), where=units != 0)
    z_scores = (net_revenue - np.mean(net_revenue)) / np.std(net_revenue)
    unit_minmax = (units - np.min(units)) / (np.max(units) - np.min(units))
    sales_band = np.where(net_revenue >= np.percentile(net_revenue, 75), "High",
                 np.where(net_revenue >= np.percentile(net_revenue, 25), "Medium", "Low"))
    monthly_revenue = np.bincount(month, weights=net_revenue, minlength=13)[1:13]
    monthly_change = np.diff(monthly_revenue)
    cumulative_revenue = np.cumsum(monthly_revenue)
    rating_clipped = np.clip(rating, 1, 5)
    monthly_factor = np.linspace(0.95, 1.06, 12)
    adjusted_revenue = net_revenue * monthly_factor[month - 1]

    # Five statistical calculations requested in the internship brief.
    stats = {
        "mean_net_revenue_inr": float(np.mean(net_revenue)),
        "median_net_revenue_inr": float(np.median(net_revenue)),
        "variance_net_revenue_inr_squared": float(np.var(net_revenue)),
        "std_dev_net_revenue_inr": float(np.std(net_revenue)),
        "90th_percentile_net_revenue_inr": float(np.percentile(net_revenue, 90)),
        "minimum_net_revenue_inr": float(np.min(net_revenue)),
        "maximum_net_revenue_inr": float(np.max(net_revenue)),
    }

    # Matrix calculations: construct, transpose, multiply, covariance/determinant/inverse.
    features = np.column_stack((units, price, discount, marketing))
    feature_mean = np.mean(features, axis=0)
    feature_std = np.std(features, axis=0)
    feature_std_safe = np.where(feature_std == 0, 1, feature_std)
    standardized = (features - feature_mean) / feature_std_safe  # broadcasting across columns
    transposed = standardized.T
    gram = transposed @ standardized
    covariance = np.cov(standardized, rowvar=False)
    regularized_covariance = covariance + np.eye(covariance.shape[0]) * 1e-8
    determinant = float(np.linalg.det(regularized_covariance))
    inverse = np.linalg.inv(regularized_covariance)
    coefficients = np.array([0.40, 0.25, -0.10, 0.25])
    weighted_activity_index = standardized @ coefficients
    solved = np.linalg.solve(regularized_covariance, np.ones(regularized_covariance.shape[0]))

    return {
        "record_count": int(len(units)),
        "net_revenue": net_revenue,
        "gross_revenue": gross,
        "discount_amount": discount_amount,
        "revenue_per_unit": revenue_per_unit,
        "return_rate": return_rate,
        "z_scores": z_scores,
        "units_minmax": unit_minmax,
        "sales_band": sales_band,
        "monthly_revenue": monthly_revenue,
        "monthly_change": monthly_change,
        "cumulative_revenue": cumulative_revenue,
        "rating_clipped": rating_clipped,
        "adjusted_revenue": adjusted_revenue,
        "statistics": stats,
        "feature_shape": tuple(features.shape),
        "transpose_shape": tuple(transposed.shape),
        "gram_matrix": gram,
        "covariance_matrix": covariance,
        "regularized_determinant": determinant,
        "inverse_matrix": inverse,
        "weighted_activity_index": weighted_activity_index,
        "linear_system_solution": solved,
    }
