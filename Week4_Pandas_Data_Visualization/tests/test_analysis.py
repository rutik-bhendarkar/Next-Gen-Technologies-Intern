import unittest
from pathlib import Path
import sys
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis import load_raw_data, clean_data, add_time_features, compute_tables, detect_iqr_outliers


class TestWeek4Analysis(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw = load_raw_data()
        cls.clean, cls.audit = clean_data(cls.raw)
        cls.featured = add_time_features(cls.clean)

    def test_raw_dataset_shape_and_fields(self):
        self.assertEqual(self.raw.shape, (2284, 2))
        self.assertIn("date", self.raw.columns)
        self.assertIn("co2_ppm", self.raw.columns)

    def test_missing_values_are_detected(self):
        self.assertEqual(self.audit["missing_co2_before"], 59)

    def test_cleaning_fills_missing_values(self):
        self.assertEqual(self.audit["missing_co2_after"], 0)
        self.assertFalse(self.clean["co2_ppm"].isna().any())

    def test_duplicate_dates_are_absent(self):
        self.assertEqual(self.audit["duplicate_dates_before"], 0)
        self.assertFalse(self.clean["date"].duplicated().any())

    def test_cleaner_removes_duplicate_dates_and_invalid_dates(self):
        sample = self.raw.head(4).copy()
        duplicate = sample.iloc[[1]].copy()
        invalid = sample.iloc[[2]].copy()
        invalid["date"] = "not-a-date"
        mixed = pd.concat([sample, duplicate, invalid], ignore_index=True)
        cleaned, audit = clean_data(mixed)
        self.assertEqual(audit["duplicate_dates_removed"], 1)
        self.assertEqual(audit["invalid_date_rows_removed"], 1)
        self.assertEqual(len(cleaned), 4)
        self.assertFalse(cleaned["date"].duplicated().any())

    def test_time_features_created(self):
        for column in ["year", "month", "month_name", "quarter", "decade", "week_number", "rolling_52_week_mean"]:
            self.assertIn(column, self.featured.columns)

    def test_summary_tables_created(self):
        tables = compute_tables(self.featured)
        self.assertEqual(set(tables), {"numeric_summary", "annual_summary", "monthly_summary", "decade_summary"})
        self.assertEqual(len(tables["monthly_summary"]), 12)

    def test_iqr_report_has_expected_fields(self):
        result = detect_iqr_outliers(self.featured["co2_ppm"])
        self.assertIn("lower_bound", result)
        self.assertIn("upper_bound", result)
        self.assertIn("outlier_count", result)


if __name__ == "__main__":
    unittest.main(verbosity=2)
