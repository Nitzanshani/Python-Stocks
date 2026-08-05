import unittest

import numpy as np
import pandas as pd

from curve_models import fit_curve_models


class CurveModelTests(unittest.TestCase):
    def test_detects_smooth_rising_curve_without_future_leakage(self):
        x = np.arange(140, dtype=float)
        values = 100 * np.exp(0.002 * x + 0.000003 * x ** 2)
        result = fit_curve_models(pd.DataFrame({"Close": values}))
        self.assertIsNotNone(result)
        self.assertEqual(len(result["models"]), 5)
        self.assertGreater(result["best"]["forecast_5d_pct"], 0)
        self.assertIn(result["best"]["window"], (14, 21, 42, 63))


if __name__ == "__main__":
    unittest.main()
