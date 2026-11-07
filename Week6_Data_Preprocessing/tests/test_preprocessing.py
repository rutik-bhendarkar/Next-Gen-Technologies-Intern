import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from src.preprocessing import remove_exact_duplicates, validate_domain_values, IQRClipper, run_preprocessing

ROOT = Path(__file__).resolve().parents[1]

class TestWeek6Preprocessing(unittest.TestCase):
    def test_duplicate_removal(self):
        frame = pd.DataFrame({"x": [1, 1, 2], "y": ["a", "a", "b"]})
        cleaned, report = remove_exact_duplicates(frame)
        self.assertEqual(len(cleaned), 2)
        self.assertEqual(report["duplicates_removed"], 1)

    def test_domain_validation(self):
        frame = pd.DataFrame({
            "Age": [-1, 35], "TenureMonths": [2, 150],
            "MonthlyCharges": [55, 999], "TotalCharges": [-5, 25],
            "SupportCalls": [1, 40]
        })
        cleaned, report = validate_domain_values(frame)
        self.assertTrue(pd.isna(cleaned.loc[0, "Age"]))
        self.assertTrue(pd.isna(cleaned.loc[1, "TenureMonths"]))
        self.assertTrue(pd.isna(cleaned.loc[1, "MonthlyCharges"]))
        self.assertEqual(sum(report.values()), 5)

    def test_iqr_clipper_caps_extreme_value(self):
        train = np.array([[1.0], [2.0], [2.0], [3.0], [4.0], [100.0]])
        clipper = IQRClipper().fit(train)
        transformed = clipper.transform(np.array([[1000.0], [-100.0]]))
        self.assertLessEqual(transformed[0, 0], clipper.upper_bounds_[0])
        self.assertGreaterEqual(transformed[1, 0], clipper.lower_bounds_[0])

    def test_full_pipeline(self):
        result = run_preprocessing(ROOT, make_figures=False)
        self.assertEqual(result["duplicate_report"]["duplicates_removed"], 8)
        self.assertEqual(len(result["X_train"]), 400)
        self.assertEqual(len(result["X_test"]), 100)
        self.assertGreater(result["outputs"]["standard"]["feature_count"], 5)
        train = pd.read_csv(ROOT/"data/processed/train_processed_standard.csv")
        test = pd.read_csv(ROOT/"data/processed/test_processed_standard.csv")
        self.assertFalse(train.isna().any().any())
        self.assertFalse(test.isna().any().any())
        self.assertTrue(set(train["Churn"].unique()).issubset({0,1}))
        self.assertTrue(set(test["Churn"].unique()).issubset({0,1}))

    def test_minmax_output_bounded(self):
        run_preprocessing(ROOT, make_figures=False)
        train = pd.read_csv(ROOT/"data/processed/train_processed_minmax.csv")
        feature_cols = [c for c in train.columns if c != "Churn" and c.startswith("numeric__")]
        self.assertGreater(len(feature_cols), 0)
        self.assertTrue((train[feature_cols].to_numpy() >= -1e-9).all())
        self.assertTrue((train[feature_cols].to_numpy() <= 1+1e-9).all())

if __name__ == "__main__":
    unittest.main(verbosity=2)
