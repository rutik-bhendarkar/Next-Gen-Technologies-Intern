# Week 3 Results Explanation — Numerical Computing with NumPy

## 1. Dataset overview

The project uses **120 synthetic retail sales records** covering 12 months, five products, and two regions. The file is `data/retail_sales.csv`. It contains units sold, unit price, discount percentage, marketing spend, returns, and customer rating. All monetary values are in INR. The dataset is generated for learning and is not real business data.

## 2. Numerical operations performed

The notebook performs 14 numerical operations:

1. Element-wise multiplication to calculate gross revenue (`units × unit price`).
2. Element-wise multiplication to calculate discount amount.
3. Subtraction to calculate net revenue.
4. Safe division to calculate revenue per unit.
5. Safe division to calculate return rate per record.
6. Z-score transformation of net revenue.
7. Min-max normalization of units sold.
8. Conditional categorization into High, Medium, and Low sales bands using `np.where`.
9. Concatenation of arrays using `np.concatenate`.
10. Reshaping units into a 12 × 10 matrix.
11. Weighted monthly aggregation using `np.bincount`.
12. Month-to-month differences using `np.diff`.
13. Cumulative totals using `np.cumsum`.
14. Value clipping using `np.clip`.

Broadcasting is demonstrated by applying a month-specific adjustment factor to each row and by standardizing each column of a 2-D feature matrix.

## 3. Statistical calculations

| Calculation | Result |
|---|---:|
| Mean net revenue per record | INR 56,645.78 |
| Median net revenue per record | INR 53,138.69 |
| Population variance | 743,075,283.33 INR² |
| Population standard deviation | INR 27,259.41 |
| 90th percentile | INR 97,192.61 |

Minimum net revenue per record was INR 12,133.08; maximum was INR 136,070.96.

## 4. Matrix operations

The four-column feature matrix contains units sold, unit price, discount percentage, and marketing spend.

- **Transpose:** converts the standardized matrix from 120 × 4 to 4 × 120.
- **Matrix multiplication:** `X.T @ X` creates a 4 × 4 Gram matrix.
- **Covariance matrix:** `np.cov` calculates relationships among the four standardized features.
- Additional operations calculate a regularized determinant, matrix inverse, matrix-vector product, and solution to a linear system.

A small diagonal regularization term (`1e-8`) is added before the determinant and inverse/solve operations to improve numerical stability.

## 5. Key observations from this sample

- Total units sold: **5,305**.
- Total gross revenue: **INR 7,343,004.16**.
- Total discount amount: **INR 545,510.48**.
- Total net revenue: **INR 6,797,493.68**.
- Overall return rate: **4.41%**.
- Highest net-revenue month: **Month 11**.
- Lowest net-revenue month: **Month 1**.

These observations describe only the generated teaching dataset. They do not represent an actual retailer or market.

## 6. Conclusion

The assignment objectives are met: NumPy arrays, indexing, slicing, broadcasting, 14 numerical operations, seven statistical calculations, and multiple matrix operations are demonstrated. The notebook is reproducible and exports a JSON summary, monthly revenue CSV, and correlation matrix CSV.
