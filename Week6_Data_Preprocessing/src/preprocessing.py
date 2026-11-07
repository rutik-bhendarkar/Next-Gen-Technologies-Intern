"""Week 6 data-preprocessing pipeline.

The split occurs before any statistic-learning transformations. Imputation,
IQR clipping thresholds, scaling and one-hot categories are fit on training
features only to avoid test-data leakage.
"""
from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
import joblib

ID_COLUMN = "CustomerID"
TARGET_COLUMN = "Churn"
NUMERIC_FEATURES = ["Age", "TenureMonths", "MonthlyCharges", "TotalCharges", "SupportCalls"]
CATEGORICAL_FEATURES = ["ContractType", "PaymentMethod", "InternetService", "PaperlessBilling"]
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET_MAP = {"No": 0, "Yes": 1}
DOMAIN_LIMITS = {
    "Age": (18, 100),
    "TenureMonths": (0, 120),
    "MonthlyCharges": (0, 300),
    "TotalCharges": (0, None),
    "SupportCalls": (0, 30),
}

class IQRClipper(BaseEstimator, TransformerMixin):
    """Cap numeric columns at training-set Q1 - 1.5*IQR and Q3 + 1.5*IQR."""
    def __init__(self, factor: float = 1.5):
        self.factor = factor
    def fit(self, X, y=None):
        arr = np.asarray(X, dtype=float)
        self.lower_bounds_ = np.nanquantile(arr, 0.25, axis=0) - self.factor * (np.nanquantile(arr, 0.75, axis=0) - np.nanquantile(arr, 0.25, axis=0))
        self.upper_bounds_ = np.nanquantile(arr, 0.75, axis=0) + self.factor * (np.nanquantile(arr, 0.75, axis=0) - np.nanquantile(arr, 0.25, axis=0))
        return self
    def transform(self, X):
        arr = np.asarray(X, dtype=float)
        return np.clip(arr, self.lower_bounds_, self.upper_bounds_)
    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features if input_features is not None else [f"x{i}" for i in range(len(self.lower_bounds_))], dtype=object)


def remove_exact_duplicates(frame: pd.DataFrame):
    before = len(frame)
    result = frame.drop_duplicates().reset_index(drop=True)
    return result, {"rows_before": int(before), "rows_after": int(len(result)), "duplicates_removed": int(before-len(result))}


def validate_domain_values(frame: pd.DataFrame):
    """Replace impossible numeric values with NaN; subsequent imputation is train-fit only."""
    result = frame.copy()
    report = {}
    for col, (low, high) in DOMAIN_LIMITS.items():
        invalid = pd.Series(False, index=result.index)
        if low is not None:
            invalid |= result[col].notna() & (result[col] < low)
        if high is not None:
            invalid |= result[col].notna() & (result[col] > high)
        report[col] = int(invalid.sum())
        result.loc[invalid, col] = np.nan
    return result, report


def iqr_outlier_summary(frame: pd.DataFrame):
    """Descriptive IQR audit. The actual clipping thresholds are learned on train only."""
    records = []
    for col in NUMERIC_FEATURES:
        values = pd.to_numeric(frame[col], errors="coerce").dropna()
        if values.empty:
            records.append({"feature": col, "q1": None, "q3": None, "lower_bound": None, "upper_bound": None, "outlier_count": 0})
            continue
        q1, q3 = values.quantile([0.25, 0.75])
        iqr = q3-q1
        lo, hi = q1-1.5*iqr, q3+1.5*iqr
        count = int(((values < lo) | (values > hi)).sum())
        records.append({"feature": col, "q1": float(q1), "q3": float(q3), "lower_bound": float(lo), "upper_bound": float(hi), "outlier_count": count})
    return records


def build_preprocessor(scaling: str = "standard"):
    if scaling not in {"standard", "minmax"}:
        raise ValueError("scaling must be 'standard' or 'minmax'")
    scaler = StandardScaler() if scaling == "standard" else MinMaxScaler()
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("iqr_clip", IQRClipper(factor=1.5)),
        ("scaler", scaler),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("numeric", numeric_pipe, NUMERIC_FEATURES),
        ("categorical", categorical_pipe, CATEGORICAL_FEATURES),
    ], remainder="drop", verbose_feature_names_out=True)


def _to_processed_frame(matrix, feature_names, target, index):
    out = pd.DataFrame(matrix, columns=feature_names, index=index)
    out[TARGET_COLUMN] = target.values
    return out


def make_plots(raw_frame, valid_frame, clean_frame, outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)
    # Missingness by column, before cleaning.
    missing = raw_frame.isna().sum().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(10, 5))
    missing.plot(kind="bar", ax=ax)
    ax.set_title("Missing Values by Column — Raw Dataset")
    ax.set_ylabel("Missing cell count")
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout(); fig.savefig(outdir/"01_missing_values_raw.png", dpi=150); plt.close(fig)
    # Target balance.
    fig, ax = plt.subplots(figsize=(6, 4))
    clean_frame[TARGET_COLUMN].value_counts().reindex(["No", "Yes"]).plot(kind="bar", ax=ax)
    ax.set_title("Churn Target Distribution"); ax.set_xlabel("Churn"); ax.set_ylabel("Customers")
    fig.tight_layout(); fig.savefig(outdir/"02_target_distribution.png", dpi=150); plt.close(fig)
    # Numeric boxplots before and after domain validation.
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    raw_frame[NUMERIC_FEATURES].boxplot(ax=axes[0], rot=35); axes[0].set_title("Before Domain Validation")
    valid_frame[NUMERIC_FEATURES].boxplot(ax=axes[1], rot=35); axes[1].set_title("After Invalid Values Set to Missing")
    fig.tight_layout(); fig.savefig(outdir/"03_numeric_boxplots.png", dpi=150); plt.close(fig)
    # Histograms.
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    for ax, col in zip(axes.flat, NUMERIC_FEATURES):
        clean_frame[col].hist(ax=ax, bins=25)
        ax.set_title(col)
    axes.flat[-1].axis("off")
    fig.suptitle("Numeric Feature Distributions After Cleaning Rules")
    fig.tight_layout(); fig.savefig(outdir/"04_numeric_distributions.png", dpi=150); plt.close(fig)
    # Correlation matrix.
    corr = clean_frame[NUMERIC_FEATURES].corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)), corr.columns, rotation=40, ha="right")
    ax.set_yticks(range(len(corr.index)), corr.index)
    for i in range(len(corr.index)):
        for j in range(len(corr.columns)):
            ax.text(j, i, f"{corr.iloc[i,j]:.2f}", ha="center", va="center", fontsize=8)
    ax.set_title("Numeric Feature Correlation Matrix")
    fig.colorbar(image, ax=ax, label="Correlation")
    fig.tight_layout(); fig.savefig(outdir/"05_correlation_heatmap.png", dpi=150); plt.close(fig)
    # Contract vs churn.
    ct = pd.crosstab(clean_frame["ContractType"], clean_frame[TARGET_COLUMN], normalize="index")
    fig, ax = plt.subplots(figsize=(8, 5)); ct.plot(kind="bar", stacked=True, ax=ax)
    ax.set_title("Churn Proportion by Contract Type"); ax.set_ylabel("Proportion")
    fig.tight_layout(); fig.savefig(outdir/"06_contract_churn.png", dpi=150); plt.close(fig)


def run_preprocessing(project_root: str | Path, make_figures: bool = True):
    root = Path(project_root)
    raw_path = root/"data"/"raw"/"customer_churn_raw.csv"
    raw = pd.read_csv(raw_path)
    dedup, duplicate_report = remove_exact_duplicates(raw)
    validated, domain_report = validate_domain_values(dedup)
    outlier_report = iqr_outlier_summary(validated)
    # Keep an auditable, pre-imputation cleaned source. Missing cells remain visible here.
    processed_dir = root/"data"/"processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    validated.to_csv(processed_dir/"cleaned_dataset_before_imputation.csv", index=False)
    # No target leakage: separate target, then split, then fit all learned preprocessing on X_train only.
    if validated[TARGET_COLUMN].isna().any():
        validated = validated.dropna(subset=[TARGET_COLUMN]).copy()
    X = validated[FEATURE_COLUMNS]
    y = validated[TARGET_COLUMN].map(TARGET_MAP)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    outputs = {}
    for scaling in ["standard", "minmax"]:
        preprocessor = build_preprocessor(scaling)
        train_matrix = preprocessor.fit_transform(X_train)
        test_matrix = preprocessor.transform(X_test)
        feature_names = preprocessor.get_feature_names_out()
        train_frame = _to_processed_frame(train_matrix, feature_names, y_train, X_train.index)
        test_frame = _to_processed_frame(test_matrix, feature_names, y_test, X_test.index)
        train_name = f"train_processed_{scaling}.csv"
        test_name = f"test_processed_{scaling}.csv"
        train_frame.to_csv(processed_dir/train_name, index=False)
        test_frame.to_csv(processed_dir/test_name, index=False)
        joblib.dump(preprocessor, root/"models"/f"preprocessor_{scaling}.joblib")
        outputs[scaling] = {"train_rows": int(len(train_frame)), "test_rows": int(len(test_frame)), "feature_count": int(len(feature_names)), "train_file": train_name, "test_file": test_name}
    # A compact target mapping and metadata file make the outputs reusable.
    metadata = {
        "dataset": "Synthetic customer churn dataset for educational preprocessing practice",
        "random_seed": 20261105,
        "target_mapping": TARGET_MAP,
        "id_column_excluded": ID_COLUMN,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "rows_after_duplicate_removal": int(len(validated)),
        "duplicate_report": duplicate_report,
        "invalid_values_replaced_with_missing": domain_report,
        "iqr_outlier_audit": outlier_report,
        "split": {"method": "stratified train_test_split", "test_size": 0.20, "random_state": 42, "train_rows": int(len(X_train)), "test_rows": int(len(X_test))},
        "preprocessing_outputs": outputs,
        "leakage_prevention": "Imputers, IQR clipping bounds, category vocabularies and scalers are fitted on training data only. Test data is transform-only.",
    }
    (root/"outputs"/"preprocessing_summary.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    if make_figures:
        make_plots(raw, validated, validated, root/"outputs"/"figures")
    return {"raw": raw, "validated": validated, "X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test, "duplicate_report": duplicate_report, "domain_report": domain_report, "outlier_report": outlier_report, "outputs": outputs, "metadata": metadata}
