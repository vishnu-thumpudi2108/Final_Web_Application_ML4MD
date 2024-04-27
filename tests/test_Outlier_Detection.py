import unittest
import pandas as pd
import numpy as np
from your_module import detect_IQR, detect_outliers  # replace 'your_module' with the actual module name

class TestOutlierDetection(unittest.TestCase):
    def setUp(self):
        self.data = pd.DataFrame({
            'A': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'B': [1, 2, 3, 4, 5, 6, 7, 8, 9, 100]  # 100 is an outlier
        })

    def test_detect_IQR(self):
        result = detect_IQR(self.data)
        expected = pd.Series([0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
        pd.testing.assert_series_equal(result, expected)

    def test_detect_outliers(self):
        result = detect_outliers(self.data, outliers_fraction=0.1)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(result.shape[1], 12)  # 12 methods used for outlier detection

if __name__ == '__main__':
    unittest.main()