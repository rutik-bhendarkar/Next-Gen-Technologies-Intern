# Week 8 Detailed Implementation Guide

## 1. What is regression?
Regression predicts a continuous numeric output. Here, the target is house price in ₹ lakh.

## 2. Why multiple linear regression?
There are multiple input features (area, bedrooms, bathrooms, age, distance and parking). The model estimates a coefficient for each feature.

## 3. Train-test split
An 80:20 split reserves unseen rows for evaluation. `random_state=42` makes the split reproducible.

## 4. Pipeline
`SimpleImputer(strategy='median')` fills missing numeric values. `StandardScaler` transforms features to zero mean and unit variance. `LinearRegression` fits ordinary least squares. The pipeline ensures the imputer and scaler learn only from training data.

## 5. Metrics
- MAE: average absolute error.
- MSE: average squared error; large mistakes count more.
- RMSE: square root of MSE, in target units.
- R²: relative fit versus a mean-only baseline; 1 is perfect, 0 is baseline-like, negative is worse than that baseline.

## 6. Diagnostics
A good actual-vs-predicted plot tends toward the diagonal. Residuals (actual minus predicted) should ideally show no strong pattern and roughly constant spread. A residual histogram helps inspect asymmetry and extreme errors.

## 7. Important viva questions
1. Is price classification or regression? Regression because price is continuous.
2. Why not evaluate on training data? It would give an optimistic estimate of generalization.
3. Is scaling mandatory for linear regression? No, but it helps put features on comparable scales and is useful for coefficient comparison.
4. What does a negative R² mean? The model performed worse than predicting the test-set mean.
5. Does correlation prove causation? No.
6. What is overfitting? Learning training-specific noise and performing poorly on unseen data.
