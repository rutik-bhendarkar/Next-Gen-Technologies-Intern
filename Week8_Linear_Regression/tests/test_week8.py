import unittest
from pathlib import Path
import sys
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.linear_regression import load_data, train_and_evaluate, FEATURES, TARGET

class TestWeek8(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df=load_data(ROOT/"data"/"house_prices_synthetic.csv")
        cls.result=train_and_evaluate(cls.df)
    def test_dataset_rows_and_columns(self):
        self.assertEqual(len(self.df),600)
        self.assertEqual(len(FEATURES),6)
        self.assertIn(TARGET,self.df.columns)
    def test_train_test_sizes(self):
        _,Xtr,Xte,_,_,_,_=self.result
        self.assertEqual(len(Xtr),480)
        self.assertEqual(len(Xte),120)
    def test_metrics_exist_and_are_nonnegative(self):
        metrics=self.result[-1]
        for k in ["MAE","MSE","RMSE","R2"]:
            self.assertIn(k,metrics)
        self.assertGreaterEqual(metrics["MAE"],0)
        self.assertGreaterEqual(metrics["MSE"],0)
        self.assertGreaterEqual(metrics["RMSE"],0)
    def test_predictions_finite(self):
        pred=self.result[5]
        self.assertTrue(pd.Series(pred).notna().all())
    def test_r2_reasonable_for_synthetic_data(self):
        self.assertGreater(self.result[-1]["R2"],0.5)

if __name__=="__main__":
    unittest.main()
