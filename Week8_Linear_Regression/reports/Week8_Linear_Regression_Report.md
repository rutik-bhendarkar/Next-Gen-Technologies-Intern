# Week 8 Practical Report – House Price Prediction Using Linear Regression

## 1. Objective
Develop a regression model to estimate house prices from property characteristics and evaluate it using MAE, MSE, RMSE and R².

## 2. Problem statement
Estimate a property's sale price (₹ lakh) from its area, bedroom and bathroom counts, age, distance from the city centre and parking spaces.

## 3. Dataset
The project contains 600 synthetic property records and seven columns. The target is `price_lakhs`; six columns are explanatory features. This is generated teaching data, not real property-market data.

## 4. Feature selection
Selected features:
- `area_sqft`
- `bedrooms`
- `bathrooms`
- `property_age_years`
- `distance_to_city_km`
- `parking_spaces`

Target: `price_lakhs`.

These features are selected because they represent plausible property characteristics. This is a domain-informed educational selection, not a claim that they are statistically optimal.

## 5. Methodology
1. Load and inspect the dataset.
2. Separate features and target.
3. Split into 80% train and 20% test with `random_state=42`.
4. Build a pipeline with median imputation, standardization and ordinary least-squares Linear Regression.
5. Fit only on training data.
6. Predict on held-out test data.
7. Calculate MAE, MSE, RMSE and R².

The pipeline prevents preprocessing leakage because imputation and scaling are fitted only on training rows.

## 6. Mathematical model
Multiple linear regression estimates:

`ŷ = β₀ + β₁x₁ + β₂x₂ + ... + βₚxₚ`

The least-squares estimator selects coefficients that minimize the sum of squared residuals on the training data.

## 7. Evaluation metrics
- `MAE = mean(|y - ŷ|)`
- `MSE = mean((y - ŷ)²)`
- `RMSE = sqrt(MSE)`
- `R² = 1 - [sum((y - ŷ)²) / sum((y - mean(y))²)]`

See `outputs/tables/model_metrics.csv` for the exact run-specific values. MAE/RMSE are measured in ₹ lakh; MSE is in squared ₹ lakh.

## 8. Prediction examples
See `outputs/tables/prediction_results.csv`, which includes actual values, predicted values and absolute errors for the held-out test set.

## 9. Visual diagnostics
- Actual versus predicted plot: checks proximity to the ideal diagonal.
- Residuals versus predicted: checks visible patterns or changing spread.
- Residual histogram: inspects the residual distribution.

## 10. Conclusion
A complete multiple-linear-regression workflow was implemented, evaluated on a held-out test set and saved as a reusable pipeline. The exercise demonstrates feature selection, train-test splitting, model fitting, prediction and all four required evaluation metrics.

## 11. Limitations and future work
The dataset is synthetic, so metrics cannot establish real housing-price accuracy. Future work should use a cited real dataset, inspect linearity and multicollinearity, use cross-validation, compare regularized models, and evaluate performance across relevant property segments.
