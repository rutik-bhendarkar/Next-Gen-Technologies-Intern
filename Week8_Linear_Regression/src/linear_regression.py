"""Reusable linear-regression workflow for the Week 8 internship assignment."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

FEATURES = ["area_sqft", "bedrooms", "bathrooms", "property_age_years",
            "distance_to_city_km", "parking_spaces"]
TARGET = "price_lakhs"

def load_data(path):
    df = pd.read_csv(path)
    required = FEATURES + [TARGET]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    return df[required].copy()

def build_pipeline():
    # Imputation/scaling are fitted on training data only; scaling is not
    # mathematically required by ordinary least squares, but included for a
    # consistent preprocessing workflow. Coefficients are therefore in
    # standardized feature units.
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LinearRegression())
    ])

def evaluate(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "MSE": float(mse),
        "RMSE": float(np.sqrt(mse)),
        "R2": float(r2_score(y_true, y_pred))
    }

def train_and_evaluate(df, random_state=42, test_size=0.2):
    X = df[FEATURES]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)
    return pipeline, X_train, X_test, y_train, y_test, predictions, evaluate(y_test, predictions)

def save_artifacts(pipeline, metrics, output_dir):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, out / "linear_regression_pipeline.joblib")
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
