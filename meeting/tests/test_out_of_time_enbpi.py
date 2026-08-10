import unittest

import numpy as np

from kf_forecasting.models.kf_enbpi import EnbPIConfig, simulate_additive_data
from kf_forecasting.models.kf_out_of_time_enbpi import (
    OutOfTimeEnbPIConfig,
    rolling_origin_crossfit,
    run_out_of_time_enbpi,
    select_residual_window,
)


def small_config() -> OutOfTimeEnbPIConfig:
    base = EnbPIConfig(
        window_size=5,
        alpha=0.10,
        n_bootstrap=30,
        block_length=3,
        batch_size=1,
        beta_grid_size=21,
        arima_order=(1, 0, 0),
        ann_hidden_layers=(6,),
        ann_max_iter=20,
        ann_rolling_validation=False,
        ann_iteration_candidates=(20,),
        random_state=321,
    )
    return OutOfTimeEnbPIConfig(
        base=base,
        initial_train_fraction=0.60,
        n_crossfit_blocks=2,
        residual_window_candidates=(8, 12, None),
        window_validation_fraction=0.40,
        min_window_history=6,
    )


class OutOfTimeEnbPITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = small_config()
        cls.low, cls.high, cls.observed = simulate_additive_data(
            "m1m9", 58, rng=np.random.default_rng(2026)
        )

    def test_crossfit_uses_strictly_earlier_fit_prefix(self):
        predictions, folds = rolling_origin_crossfit(
            self.observed[:50], self.config
        )
        self.assertTrue(
            np.all(
                predictions["fit_end_exclusive"].to_numpy()
                <= predictions["time"].to_numpy()
            )
        )
        self.assertEqual(predictions["time"].max(), 49)
        self.assertEqual(len(folds), 2)
        np.testing.assert_allclose(
            predictions["final_residual"],
            predictions["low_residual"] + predictions["high_residual"],
            atol=1e-10,
        )

    def test_window_selection_is_from_declared_candidates(self):
        predictions, _ = rolling_origin_crossfit(
            self.observed[:50], self.config
        )
        selected, table = select_residual_window(
            predictions["final_residual"].to_numpy(), self.config
        )
        self.assertIn(selected, self.config.residual_window_candidates)
        self.assertEqual(int(table["selected"].sum()), 1)
        self.assertTrue(np.all(np.isfinite(table["mean_interval_score"])))

    def test_end_to_end_forecast_is_finite_and_ordered(self):
        result = run_out_of_time_enbpi(
            self.observed,
            50,
            config=self.config,
            model_name="M1M9_TEST",
            true_low=self.low,
            true_high=self.high,
            data_seed=2026,
        )
        forecast = result.forecast
        self.assertEqual(len(forecast.point), 8)
        self.assertTrue(np.all(np.isfinite(forecast.point)))
        self.assertTrue(np.all(forecast.lower <= forecast.upper))
        self.assertTrue(np.all(forecast.low_lower <= forecast.low_upper))
        self.assertTrue(np.all(forecast.high_lower <= forecast.high_upper))
        self.assertEqual(
            result.initial_pool_size,
            len(forecast.initial_oob_residuals),
        )
        self.assertLess(result.crossfit_predictions["time"].max(), 50)


if __name__ == "__main__":
    unittest.main()
