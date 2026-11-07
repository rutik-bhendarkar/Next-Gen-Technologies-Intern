from pathlib import Path
import sys
import pandas as pd
import matplotlib.pyplot as plt
from src.linear_regression import load_data, train_and_evaluate, save_artifacts, FEATURES, TARGET

ROOT = Path(__file__).resolve().parent
df = load_data(ROOT / "data" / "house_prices_synthetic.csv")
pipeline, X_train, X_test, y_train, y_test, pred, metrics = train_and_evaluate(df)
save_artifacts(pipeline, metrics, ROOT / "models")
pd.DataFrame({"actual_price_lakhs": y_test.values,
              "predicted_price_lakhs": pred,
              "absolute_error": abs(y_test.values-pred)}).to_csv(
                  ROOT / "outputs" / "tables" / "prediction_results.csv", index=False)
pd.DataFrame([metrics]).to_csv(ROOT / "outputs" / "tables" / "model_metrics.csv", index=False)

figdir=ROOT/"outputs"/"figures"
figdir.mkdir(parents=True,exist_ok=True)
plt.figure(figsize=(7,5))
plt.scatter(y_test, pred, alpha=.7)
lims=[min(y_test.min(),pred.min()), max(y_test.max(),pred.max())]
plt.plot(lims,lims,"--")
plt.xlabel("Actual price (₹ lakh)"); plt.ylabel("Predicted price (₹ lakh)")
plt.title("Actual vs Predicted House Prices"); plt.tight_layout()
plt.savefig(figdir/"actual_vs_predicted.png",dpi=150); plt.close()

residuals=y_test.values-pred
plt.figure(figsize=(7,5))
plt.scatter(pred,residuals,alpha=.7); plt.axhline(0,linestyle="--")
plt.xlabel("Predicted price (₹ lakh)"); plt.ylabel("Residual (₹ lakh)")
plt.title("Residuals vs Predicted"); plt.tight_layout()
plt.savefig(figdir/"residuals_vs_predicted.png",dpi=150); plt.close()

plt.figure(figsize=(7,5))
plt.hist(residuals,bins=25,edgecolor="black")
plt.xlabel("Residual (₹ lakh)"); plt.ylabel("Frequency")
plt.title("Residual Distribution"); plt.tight_layout()
plt.savefig(figdir/"residual_distribution.png",dpi=150); plt.close()

# Feature coefficients from standardized pipeline
coef=pipeline.named_steps["model"].coef_
coef_df=pd.DataFrame({"feature":FEATURES,"standardized_coefficient":coef})
coef_df.to_csv(ROOT/"outputs"/"tables"/"standardized_coefficients.csv",index=False)

print("WEEK 8 LINEAR REGRESSION COMPLETE")
print(f"Training rows: {len(X_train)} | Test rows: {len(X_test)}")
for k,v in metrics.items(): print(f"{k}: {v:.4f}")
print("Saved model, metrics, predictions, coefficients and figures.")
