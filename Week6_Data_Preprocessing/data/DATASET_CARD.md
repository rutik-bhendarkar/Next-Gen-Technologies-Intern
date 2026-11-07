# Dataset card — Synthetic Customer Churn

- **Purpose:** Demonstrate missing-value treatment, duplicate removal, range validation, outlier detection, categorical encoding, feature scaling/normalization and train/test splitting.
- **Origin:** Artificially generated for this educational assignment; it is not a downloaded company dataset and does not represent real customers.
- **Rows:** 508 rows in the raw CSV (500 generated customer records plus 8 exact duplicate rows).
- **Target:** `Churn` (`No` / `Yes`).
- **Identifier:** `CustomerID`, excluded from model features.
- **Features:** age, tenure, charges, support calls, contract type, payment method, internet service and paperless billing.
- **Known seeded quality issues:** missing feature values, 8 exact duplicate rows, invalid numeric values outside declared domain ranges, and a few high but plausible values for outlier review.
- **Random seed:** 20261105.
- **Privacy:** Contains no personal information; all rows are synthetic.

The dataset's labels are generated from a probabilistic rule using tenure, contract, support calls, monthly charges and payment method. It should be used to learn preprocessing, not to make business claims or train a production churn model.
