import unittest
from pathlib import Path
import pandas as pd
import numpy as np
from src.eda import (
    load_data, audit_and_prepare, descriptive_statistics,
    iqr_outlier_summary, churn_rate_by_category, categorical_chi_square,
    NUMERIC, TARGET
)

ROOT = Path(__file__).resolve().parents[1]

class TestWeek7EDA(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw = load_data(ROOT / "data/raw/customer_churn_raw.csv")
        cls.data, cls.audit = audit_and_prepare(cls.raw)

    def test_raw_dataset_dimensions(self):
        self.assertEqual(self.raw.shape, (508, 11))

    def test_exact_duplicates_removed(self):
        self.assertEqual(self.audit["exact_duplicate_rows"], 8)
        self.assertFalse(self.data.duplicated().any())
        self.assertEqual(len(self.data), 500)

    def test_target_has_expected_labels(self):
        self.assertEqual(set(self.data[TARGET].dropna().unique()), {"No", "Yes"})
        self.assertEqual(int((self.data[TARGET] == "Yes").sum()), 107)

    def test_domain_invalid_values_marked_missing(self):
        self.assertEqual(self.audit["invalid_domain_values_marked_missing"]["Age"], 2)
        self.assertEqual(self.audit["invalid_domain_values_marked_missing"]["SupportCalls"], 1)
        self.assertTrue(self.data["Age"].dropna().between(18, 100).all())
        self.assertTrue(self.data["SupportCalls"].dropna().between(0, 30).all())

    def test_statistics_cover_all_numeric_features(self):
        stats = descriptive_statistics(self.data)
        self.assertEqual(set(stats["feature"]), set(NUMERIC))
        self.assertTrue((stats["count"] > 0).all())

    def test_outlier_summary_is_valid(self):
        out = iqr_outlier_summary(self.data)
        self.assertEqual(set(out["feature"]), set(NUMERIC))
        self.assertTrue((out["potential_outlier_count"] >= 0).all())

    def test_category_churn_rates_are_bounded(self):
        rates = churn_rate_by_category(self.data)
        self.assertFalse(rates.empty)
        self.assertTrue(rates["churn_rate"].between(0, 100).all())

    def test_chi_square_table_generated(self):
        results = categorical_chi_square(self.data)
        self.assertEqual(len(results), 4)
        self.assertTrue(results["p_value"].between(0, 1).all())

if __name__ == "__main__":
    unittest.main(verbosity=2)
