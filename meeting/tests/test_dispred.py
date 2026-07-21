import unittest

import numpy as np
import torch

from kf_forecasting.models.dispred import (
    DistPredANNRegressor,
    component_correlation_summary,
    create_joint_windows,
    discrete_crps_from_samples,
    distpred_eq13_loss,
    picp_from_interval,
    qice_from_samples,
    simulate_m1_m3_additive_data,
    simulate_m1_m9_additive_data,
)


class DistPredSharedModuleTests(unittest.TestCase):
    def test_ann_output_shape_and_loss(self):
        model = DistPredANNRegressor(input_dim=4, ensemble_size=5)
        predictions = model(torch.zeros((3, 4), dtype=torch.float32))
        self.assertEqual(tuple(predictions.shape), (3, 5))
        loss = distpred_eq13_loss(predictions, torch.zeros(3))
        self.assertTrue(torch.isfinite(loss))

    def test_metrics_return_finite_values(self):
        samples = np.array([[0.0, 1.0], [0.5, 1.5], [1.0, 2.0]])
        truth = np.array([0.5, 1.5])
        self.assertTrue(np.isfinite(discrete_crps_from_samples(samples, truth)))
        self.assertTrue(np.isfinite(qice_from_samples(samples, truth, n_bins=2)))
        self.assertEqual(picp_from_interval([0, 1], [1, 2], truth), 1.0)

    def test_component_and_window_helpers(self):
        summary = component_correlation_summary([1, 2, 3], [2, 4, 6])
        self.assertAlmostEqual(summary["pearson_corr"], 1.0)
        features, targets = create_joint_windows([1, 2, 3], [4, 5, 6], 2)
        self.assertEqual(features.shape, (1, 4))
        self.assertEqual(targets.shape, (1, 2))

    def test_simulators_return_expected_shapes(self):
        for simulator in (simulate_m1_m3_additive_data, simulate_m1_m9_additive_data):
            low, high, measurements = simulator(20)
            self.assertEqual(low.shape, (20,))
            self.assertEqual(high.shape, (20,))
            self.assertEqual(measurements.shape, (20,))


if __name__ == "__main__":
    unittest.main()
