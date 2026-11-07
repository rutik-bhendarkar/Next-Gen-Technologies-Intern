# Dataset provenance and field definitions

This project reuses the synthetic dataset generated for Week 6. It is not a public dataset and does not contain real customers.

| Field | Meaning | Type |
|---|---|---|
| CustomerID | Synthetic record identifier (not an analysis feature) | Identifier |
| Age | Customer age in years | Numeric |
| TenureMonths | Customer tenure in months | Numeric |
| MonthlyCharges | Monthly fee amount in synthetic units | Numeric |
| TotalCharges | Total cumulative charge amount in synthetic units | Numeric |
| SupportCalls | Support calls count | Numeric |
| ContractType | Month-to-month, One year or Two year | Categorical |
| PaymentMethod | Synthetic payment category | Categorical |
| InternetService | Fiber optic, DSL or No internet | Categorical |
| PaperlessBilling | Yes or No | Categorical |
| Churn | Synthetic target label: Yes or No | Target |

The Week 6 domain validation bounds are reused for continuity. Values outside those assumed ranges are set to missing in the EDA working copy; the raw CSV is unchanged.
