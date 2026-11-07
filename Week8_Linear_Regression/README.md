# Week 8 – Linear Regression: House Price Prediction

## Assignment alignment
This project follows Week 8 of the supplied 24-week AI/ML plan:
- Feature selection
- Train-test split
- Model training
- Prediction
- MAE, MSE, RMSE and R² evaluation
- Notebook, dataset, model results and prediction examples

## Dataset
`data/house_prices_synthetic.csv` is a **synthetic educational dataset** of 600 properties. It is not real market data and must not be represented as actual property listings or valuation evidence. Target: `price_lakhs` (₹ lakh).

Features: area (sq ft), bedrooms, bathrooms, property age, distance to city centre, parking spaces.

## Setup (Windows PowerShell)
```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python run_analysis.py
```
Open the notebook:
```powershell
python -m notebook
```
Then open `notebooks/Week8_Linear_Regression.ipynb` and choose **Run All**.

Run tests:
```powershell
python -m unittest discover -s tests -v
```

## Project structure
- `data/`: raw synthetic dataset
- `src/`: reusable loading, pipeline, training and evaluation functions
- `notebooks/`: explanatory practical notebook
- `outputs/figures/`: actual-vs-predicted, residual and residual-distribution charts
- `outputs/tables/`: metrics, predictions and coefficients
- `models/`: saved fitted preprocessing + regression pipeline
- `reports/`: final assignment report
- `tests/`: automated tests

## Methodology
1. Select the six explanatory features and `price_lakhs` target.
2. Split into 80% training and 20% testing using a fixed random seed.
3. Fit median imputation and standardization only on training data through a scikit-learn Pipeline.
4. Train ordinary least-squares `LinearRegression`.
5. Predict the held-out test rows.
6. Report MAE, MSE, RMSE and R².

Standardization is included as a reproducible preprocessing step; it is not required for ordinary least squares to fit. Coefficients in the saved model are expressed in standardized feature units.

## Metric interpretation
- MAE: average absolute prediction error, in ₹ lakh.
- MSE: average squared error, in (₹ lakh)².
- RMSE: square root of MSE, in ₹ lakh; penalizes large errors more than MAE.
- R²: fraction of target variance explained relative to predicting the test-set mean. It can be negative on poor predictions.

## Limitations
The data is synthetic and generated from a simplified formula with random noise. Strong test results on it demonstrate the workflow, not real-world generalization. For a real project, use a documented housing dataset, inspect data quality, validate assumptions, check residual patterns, and consider cross-validation.
