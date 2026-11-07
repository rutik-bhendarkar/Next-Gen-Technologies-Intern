import unittest
from pathlib import Path
import sys
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.numpy_analysis import load_dataset, calculate_analysis

class TestWeek3NumPyAnalysis(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = load_dataset(ROOT / "data" / "retail_sales.csv")
        cls.result = calculate_analysis(cls.data)

    def test_expected_record_count(self):
        self.assertEqual(self.result["record_count"], 120)

    def test_revenue_is_nonnegative(self):
        self.assertTrue(np.all(self.result["net_revenue"] >= 0))

    def test_monthly_revenue_has_12_values(self):
        self.assertEqual(self.result["monthly_revenue"].shape, (12,))

    def test_matrix_shapes(self):
        self.assertEqual(self.result["feature_shape"], (120, 4))
        self.assertEqual(self.result["transpose_shape"], (4, 120))
        self.assertEqual(self.result["gram_matrix"].shape, (4, 4))

    def test_statistical_results_are_finite(self):
        for value in self.result["statistics"].values():
            self.assertTrue(np.isfinite(value))

    def test_monthly_sum_matches_total(self):
        self.assertAlmostEqual(float(np.sum(self.result["monthly_revenue"])),
                               float(np.sum(self.result["net_revenue"])), places=6)

    def test_return_rates_are_bounded(self):
        self.assertTrue(np.all(self.result["return_rate"] >= 0))
        self.assertTrue(np.all(self.result["return_rate"] <= 1))

if __name__ == "__main__":
    unittest.main()
