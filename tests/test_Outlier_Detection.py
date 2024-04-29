import unittest
import pandas as pd
import numpy as np

class TestOutlierDetection(unittest.TestCase):
    def setUp(self):
        self.data = pd.DataFrame({
            'A': np.random.normal(0, 1, 100),
            'B': np.random.normal(0, 1, 100),
            'C': np.random.normal(0, 1, 100)
        })
        self.outliers_fraction = 0.01

    def test_detect_outliers(self):
        result = detect_outliers(self.data, self.outliers_fraction)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(set(result.columns), {'ECOD','ABOD','HBOS','KNN','LOF','MCD','OCSVM','PCA','LMDD','DBSCAN','Z-Score','IQR'})

    def test_detect_outliers_output(self):
        result = detect_outliers(self.data, self.outliers_fraction)
        for col in result.columns:
            self.assertTrue((result[col] == 0) | (result[col] == 1)).all()

    def test_detect_outliers_fraction(self):
        with self.assertRaises(ValueError):
            detect_outliers(self.data, -0.01)
        with self.assertRaises(ValueError):
            detect_outliers(self.data, 1.01)

if __name__ == '__main__':
    unittest.main()