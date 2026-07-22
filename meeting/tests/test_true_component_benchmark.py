import unittest

import numpy as np

from kf_forecasting.models.kf_enbpi import EnbPIConfig
from kf_forecasting.models.kf_true_component_benchmark import (
    paired_comparison_metrics,
    run_paired_true_component_experiment,
    run_true_component_enbpi,
)


class TrueComponentBenchmarkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = EnbPIConfig(
            window_size=6,
            n_bootstrap=20,
            block_length=4,
            batch_size=1,
            arima_order=(1, 0, 0),
            ann_hidden_layers=(8,),
            ann_max_iter=20,
            ann_rolling_validation=False,
            ann_iteration_candidates=(20,),
            random_state=123,
            oob_bias_correction_mode="combined",
        )
        cls.current, cls.true_component = run_paired_true_component_experiment(
            "m1m9",
            train_size=100,
            horizon=8,
            config=cls.config,
            data_seed=456,
        )

    def test_paired_outputs_are_aligned_and_finite(self):
        self.assertTrue(
            np.array_equal(self.current.test_times, self.true_component.test_times)
        )
        self.assertTrue(np.allclose(self.current.truth, self.true_component.truth))
        self.assertTrue(np.all(np.isfinite(self.true_component.point)))
        self.assertTrue(
            np.all(self.true_component.lower <= self.true_component.upper)
        )

    def test_true_components_replace_kf_references(self):
        self.assertTrue(
            np.array_equal(
                self.true_component.estimated_low, self.true_component.true_low
            )
        )
        self.assertTrue(
            np.array_equal(
                self.true_component.estimated_high, self.true_component.true_high
            )
        )

    def test_comparison_metrics_have_expected_fields(self):
        metrics = paired_comparison_metrics(self.current, self.true_component)
        self.assertAlmostEqual(
            metrics["rmse_gain_kf_minus_true_component"],
            metrics["kf_final_rmse"] - metrics["true_component_final_rmse"],
        )

    def test_rejects_nonsequential_batches(self):
        y = np.linspace(0.0, 1.0, 20)
        with self.assertRaisesRegex(ValueError, "batch_size=1"):
            run_true_component_enbpi(
                y,
                y / 2.0,
                y / 2.0,
                15,
                config=EnbPIConfig(window_size=3, batch_size=2),
            )


if __name__ == "__main__":
    unittest.main()
