import unittest
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from kf_forecasting.models.kf_enbpi import EnbPIConfig, KalmanEnbPI
from kf_forecasting.models.kf_peak_weighted_enbpi import (
    PeakWeightedEnbPIConfig,
    PeakWeightedRegressor,
    high_peak_metrics,
    simulate_and_run_peak_weighted,
)


class PeakWeightedEnbPITests(unittest.TestCase):
    def test_peak_rows_receive_requested_weight(self):
        x = np.arange(90, dtype=float).reshape(30, 3)
        y = np.r_[np.zeros(27), -5.0, 4.0, 6.0]
        base = make_pipeline(
            StandardScaler(),
            MLPRegressor(hidden_layer_sizes=(4,), max_iter=5, random_state=7),
        )
        model = PeakWeightedRegressor(
            base,
            peak_quantile=0.90,
            peak_weight=3.0,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            model.fit(x, y)
        self.assertEqual(int(np.sum(model.sample_weight_ == 3.0)), 3)
        self.assertEqual(int(np.sum(model.sample_weight_ == 1.0)), 27)
        self.assertEqual(model.predict(x[:2]).shape, (2,))

    def test_original_ann_remains_unwrapped(self):
        original = KalmanEnbPI(
            EnbPIConfig(ann_max_iter=5, ann_iteration_candidates=(5,))
        )._new_ann(1, max_iter=5)
        self.assertNotIsInstance(original, PeakWeightedRegressor)

    def test_small_peak_weighted_run_is_finite(self):
        config = PeakWeightedEnbPIConfig(
            window_size=5,
            n_bootstrap=20,
            block_length=3,
            arima_order=(1, 0, 0),
            ann_hidden_layers=(4,),
            ann_max_iter=20,
            ann_iteration_candidates=(20,),
            ann_rolling_validation=False,
            random_state=7,
        )
        result = simulate_and_run_peak_weighted(
            "m1m9",
            train_size=80,
            horizon=10,
            config=config,
            data_seed=8,
        )
        self.assertEqual(result.model_name, "m1m9_peak_weighted")
        self.assertEqual(result.point.shape, (10,))
        self.assertTrue(np.all(np.isfinite(result.point)))
        metrics = high_peak_metrics(result, peak_quantile=0.90)
        self.assertTrue(np.isfinite(metrics["kf_high_tail_rmse"]))
        self.assertTrue(np.isfinite(metrics["kf_high_peak_amplitude_ratio"]))


if __name__ == "__main__":
    unittest.main()
