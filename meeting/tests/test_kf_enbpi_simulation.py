import unittest

import numpy as np

from kf_forecasting.models.kf_enbpi import (
    simulate_additive_data,
    simulate_m1_m9_additive_data,
    simulate_m1_m9_constant_high_variance_data,
    simulate_m1_m9_garch_low_constant_high_variance_data,
)


def m9_conditional_mean(high: np.ndarray) -> np.ndarray:
    lag = high[:-1]
    gate = 1.0 / (
        1.0 + np.exp(np.clip(-10.0 * lag, -700.0, 700.0))
    )
    return 0.8 * lag - 0.8 * lag * gate


class KalmanEnbPISimulationTests(unittest.TestCase):
    def test_constant_high_variance_changes_only_innovation_scale(self):
        n_steps = 200
        seed = 2026
        coupled_low, coupled_high, _ = simulate_m1_m9_additive_data(
            n_steps, rng=np.random.default_rng(seed)
        )
        constant_low, constant_high, _ = (
            simulate_m1_m9_constant_high_variance_data(
                n_steps, rng=np.random.default_rng(seed)
            )
        )

        np.testing.assert_allclose(coupled_low, constant_low)

        coupled_innovation = (
            coupled_high[1:] - m9_conditional_mean(coupled_high)
        )
        constant_innovation = (
            constant_high[1:] - m9_conditional_mean(constant_high)
        )
        expected_scale = 1.0 + 0.5 * np.abs(coupled_low[:-1])
        np.testing.assert_allclose(
            coupled_innovation,
            expected_scale * constant_innovation,
            rtol=1e-12,
            atol=1e-12,
        )

    def test_model_key_dispatches_to_constant_variance_dgp(self):
        seed = 123
        expected = simulate_m1_m9_constant_high_variance_data(
            40, rng=np.random.default_rng(seed)
        )
        actual = simulate_additive_data(
            "m1m9_constant_high_variance",
            40,
            rng=np.random.default_rng(seed),
        )
        for expected_component, actual_component in zip(expected, actual):
            np.testing.assert_allclose(expected_component, actual_component)

    def test_garch_low_has_time_varying_variance_and_fixed_high_path(self):
        n_steps = 500
        seed = 2468
        low, high, observed = (
            simulate_m1_m9_garch_low_constant_high_variance_data(
                n_steps,
                rng=np.random.default_rng(seed),
            )
        )
        alternative_low, alternative_high, _ = (
            simulate_m1_m9_garch_low_constant_high_variance_data(
                n_steps,
                rng=np.random.default_rng(seed),
                low_garch_omega=0.80,
            )
        )

        self.assertTrue(np.all(np.isfinite(observed)))
        self.assertFalse(np.allclose(low, alternative_low))
        # With an identical random-number stream, changing only low GARCH
        # parameters must not change the M9 high path.
        np.testing.assert_allclose(high, alternative_high, atol=0.0, rtol=0.0)

        innovation = np.empty(n_steps)
        innovation[0] = low[0]
        innovation[1:] = low[1:] - 0.6 * low[:-1]
        variance = np.empty(n_steps)
        variance[0] = 0.20 / (1.0 - 0.35 - 0.60)
        for t in range(1, n_steps):
            variance[t] = (
                0.20
                + 0.35 * innovation[t - 1] ** 2
                + 0.60 * variance[t - 1]
            )
        self.assertGreater(float(np.max(variance) / np.min(variance)), 3.0)

    def test_model_key_dispatches_to_garch_low_dgp(self):
        seed = 1357
        expected = simulate_m1_m9_garch_low_constant_high_variance_data(
            60, rng=np.random.default_rng(seed)
        )
        actual = simulate_additive_data(
            "m1m9_garch_low_constant_high_variance",
            60,
            rng=np.random.default_rng(seed),
        )
        for expected_component, actual_component in zip(expected, actual):
            np.testing.assert_allclose(expected_component, actual_component)


if __name__ == "__main__":
    unittest.main()
