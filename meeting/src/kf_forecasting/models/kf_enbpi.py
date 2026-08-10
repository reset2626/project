"""Kalman-feature EnbPI experiments for additive Giordano M1+M3/M1+M9 data.

The Kalman filter is causal and is used only to construct low/high lag features.
Prediction intervals are calibrated on the final observed series with the EnbPI
out-of-bag residual procedure from Xu and Xie (2023).
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Literal
import warnings

import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.exceptions import ConvergenceWarning as SklearnConvergenceWarning
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tools.sm_exceptions import (
    ConvergenceWarning as StatsmodelsConvergenceWarning,
)


Array = np.ndarray
M1M9_GARCH_LOW_OMEGA = 0.50
M1M9_GARCH_LOW_ALPHA = 0.15
M1M9_GARCH_LOW_BETA = 0.80
ModelName = Literal[
    "m1m3",
    "m1m9",
    "m1m9_constant_high_variance",
    "m1m9_garch_low_constant_high_variance",
]


def simulate_m1_m3_additive_data(
    n_steps: int,
    *,
    rng: np.random.Generator,
    noise_std: float = 0.15,
    low_error_std: float = 1.0,
    high_error_base_std: float = 1.0,
    high_error_low_sensitivity: float = 0.5,
) -> tuple[Array, Array, Array]:
    """Existing project DGP: Giordano M1 plus M3 with coupled volatility."""
    low = np.zeros(n_steps, dtype=float)
    high = np.zeros(n_steps, dtype=float)
    low[0] = rng.normal(0.0, 1.0)
    high[: min(2, n_steps)] = rng.normal(0.0, 0.5, min(2, n_steps))

    for t in range(1, n_steps):
        low[t] = 0.6 * low[t - 1] + rng.normal(0.0, low_error_std)
        if t < 2:
            continue
        innovation_std = high_error_base_std * (
            1.0 + high_error_low_sensitivity * abs(low[t - 1])
        )
        h1 = high[t - 1]
        mean = (0.5 + 0.9 * np.exp(-(h1**2))) * h1
        mean += (-0.8 - 1.8 * np.exp(-(h1**2))) * high[t - 2]
        high[t] = mean + rng.normal(0.0, innovation_std)

    observed = low + high + rng.normal(0.0, noise_std, n_steps)
    return low, high, observed


def simulate_m1_m9_additive_data(
    n_steps: int,
    *,
    rng: np.random.Generator,
    noise_std: float = 0.15,
    low_error_std: float = 1.0,
    high_error_base_std: float = 1.0,
    high_error_low_sensitivity: float = 0.5,
) -> tuple[Array, Array, Array]:
    """Additive combination of the exact Giordano M1 and M9 equations."""
    low = np.zeros(n_steps, dtype=float)
    high = np.zeros(n_steps, dtype=float)
    low[0] = rng.normal(0.0, 1.0)
    high[0] = rng.normal(0.0, 0.5)

    for t in range(1, n_steps):
        # Giordano M1: Y_t = 0.6 Y_{t-1} + epsilon_t.
        low[t] = 0.6 * low[t - 1] + rng.normal(0.0, low_error_std)
        innovation_std = high_error_base_std * (
            1.0 + high_error_low_sensitivity * abs(low[t - 1])
        )
        h1 = high[t - 1]
        gate = 1.0 / (1.0 + np.exp(np.clip(-10.0 * h1, -700.0, 700.0)))
        # Giordano M9: Y_t = 0.8 Y_{t-1}
        #                    - 0.8 Y_{t-1}/(1+exp(-10 Y_{t-1})) + epsilon_t.
        mean = 0.8 * h1 - 0.8 * h1 * gate
        high[t] = mean + rng.normal(0.0, innovation_std)

    observed = low + high + rng.normal(0.0, noise_std, n_steps)
    return low, high, observed


def simulate_m1_m9_constant_high_variance_data(
    n_steps: int,
    *,
    rng: np.random.Generator,
    noise_std: float = 0.15,
    low_error_std: float = 1.0,
    high_error_std: float = 1.0,
) -> tuple[Array, Array, Array]:
    """M1+M9 control DGP whose high innovation scale is independent of low.

    This retains the M1 and M9 conditional-mean equations and observation
    noise used by :func:`simulate_m1_m9_additive_data`.  The sole structural
    change is ``sigma_H,t = high_error_std`` instead of
    ``sigma_H,t = high_error_std * (1 + 0.5 * abs(L[t-1]))``.
    """
    return simulate_m1_m9_additive_data(
        n_steps,
        rng=rng,
        noise_std=noise_std,
        low_error_std=low_error_std,
        high_error_base_std=high_error_std,
        high_error_low_sensitivity=0.0,
    )


def simulate_m1_m9_garch_low_constant_high_variance_data(
    n_steps: int,
    *,
    rng: np.random.Generator,
    noise_std: float = 0.15,
    high_error_std: float = 1.0,
    low_garch_omega: float = M1M9_GARCH_LOW_OMEGA,
    low_garch_alpha: float = M1M9_GARCH_LOW_ALPHA,
    low_garch_beta: float = M1M9_GARCH_LOW_BETA,
) -> tuple[Array, Array, Array]:
    """M1+M9 control DGP with volatile GARCH low innovations.

    The low conditional mean remains Giordano M1,

    ``L_t = 0.6 L_{t-1} + u_t``,

    but ``u_t = sqrt(h_t) z_t`` and its conditional variance follows a
    GARCH(1, 1) recursion

    ``h_t = omega + alpha u_{t-1}^2 + beta h_{t-1}``.

    The defaults deliberately create conspicuous, persistent volatility
    clusters: ``alpha + beta = 0.95`` and the unconditional innovation
    variance is four.  The M9 high branch keeps its original conditional-mean
    equation but has constant innovation standard deviation.  Consequently,
    high volatility is structurally independent of the low state and of the
    low GARCH variance.
    """
    if n_steps < 1:
        raise ValueError("n_steps must be positive")
    if low_garch_omega <= 0.0:
        raise ValueError("low_garch_omega must be positive")
    if low_garch_alpha < 0.0 or low_garch_beta < 0.0:
        raise ValueError("GARCH alpha and beta must be non-negative")
    if low_garch_alpha + low_garch_beta >= 1.0:
        raise ValueError("GARCH stationarity requires alpha + beta < 1")
    if high_error_std <= 0.0 or noise_std < 0.0:
        raise ValueError("Innovation scales must be valid")

    low = np.zeros(n_steps, dtype=float)
    high = np.zeros(n_steps, dtype=float)
    low_innovation = np.zeros(n_steps, dtype=float)
    low_variance = np.zeros(n_steps, dtype=float)

    unconditional_variance = low_garch_omega / (
        1.0 - low_garch_alpha - low_garch_beta
    )
    low_variance[0] = unconditional_variance
    low_innovation[0] = np.sqrt(low_variance[0]) * rng.normal()
    low[0] = low_innovation[0]
    high[0] = rng.normal(0.0, 0.5)

    for t in range(1, n_steps):
        low_variance[t] = (
            low_garch_omega
            + low_garch_alpha * low_innovation[t - 1] ** 2
            + low_garch_beta * low_variance[t - 1]
        )
        low_innovation[t] = np.sqrt(low_variance[t]) * rng.normal()
        low[t] = 0.6 * low[t - 1] + low_innovation[t]

        h1 = high[t - 1]
        gate = 1.0 / (1.0 + np.exp(np.clip(-10.0 * h1, -700.0, 700.0)))
        high_mean = 0.8 * h1 - 0.8 * h1 * gate
        high[t] = high_mean + rng.normal(0.0, high_error_std)

    observed = low + high + rng.normal(0.0, noise_std, n_steps)
    return low, high, observed


def simulate_additive_data(
    model: ModelName, n_steps: int, *, rng: np.random.Generator
) -> tuple[Array, Array, Array]:
    if model == "m1m3":
        return simulate_m1_m3_additive_data(n_steps, rng=rng)
    if model == "m1m9":
        return simulate_m1_m9_additive_data(n_steps, rng=rng)
    if model == "m1m9_constant_high_variance":
        return simulate_m1_m9_constant_high_variance_data(n_steps, rng=rng)
    if model == "m1m9_garch_low_constant_high_variance":
        return simulate_m1_m9_garch_low_constant_high_variance_data(
            n_steps, rng=rng
        )
    raise ValueError(f"Unknown model: {model}")


def causal_kalman_decomposition(
    observations: Array,
    *,
    process_variance: float = 2.0,
    measurement_variance: float = 10.0,
    initial_error: float = 1.0,
) -> tuple[Array, Array]:
    """Causal local-level Kalman smoother used by the existing notebooks."""
    y = np.asarray(observations, dtype=float)
    if y.ndim != 1 or len(y) == 0:
        raise ValueError("observations must be a non-empty one-dimensional array")
    low = np.empty_like(y)
    state = float(y[0])
    error = float(initial_error)
    for t, measurement in enumerate(y):
        predicted_error = error + process_variance
        gain = predicted_error / (predicted_error + measurement_variance)
        state = state + gain * (float(measurement) - state)
        error = (1.0 - gain) * predicted_error
        low[t] = state
    return low, y - low


def make_lagged_features(
    observed: Array, low: Array, high: Array, window_size: int
) -> tuple[Array, Array, Array]:
    """Features at t contain information through t-1 only (no look-ahead)."""
    y = np.asarray(observed, dtype=float)
    if not (len(y) == len(low) == len(high)):
        raise ValueError("observed, low, and high must have equal lengths")
    if window_size < 1 or len(y) <= window_size:
        raise ValueError("window_size must be positive and smaller than series length")
    rows, targets, times = [], [], []
    for t in range(window_size, len(y)):
        rows.append(np.r_[low[t - window_size : t], high[t - window_size : t], y[t - window_size : t]])
        targets.append(y[t])
        times.append(t)
    return np.asarray(rows), np.asarray(targets), np.asarray(times, dtype=int)


def moving_block_bootstrap_indices(
    n: int, block_length: int, rng: np.random.Generator
) -> Array:
    """Circular moving-block bootstrap returning exactly n row indices."""
    if not 1 <= block_length <= n:
        raise ValueError("block_length must lie in [1, n]")
    blocks_needed = int(np.ceil(n / block_length))
    starts = rng.integers(0, n, size=blocks_needed)
    offsets = np.arange(block_length)
    return np.concatenate([(start + offsets) % n for start in starts])[:n]


def _arima_fit_converged(fitted: object) -> bool:
    """Return whether an ARIMA fit has finite parameters and reports convergence."""
    params = np.asarray(getattr(fitted, "params", []), dtype=float)
    if params.size == 0 or not np.all(np.isfinite(params)):
        return False
    return bool(getattr(fitted, "mle_retvals", {}).get("converged", True))


def fit_arima_robust(
    series: Array,
    order: tuple[int, int, int],
    *,
    max_iter: int = 500,
) -> tuple[object, int, bool]:
    """Fit ARIMA with a deterministic optimizer retry.

    L-BFGS is attempted first.  If statsmodels does not report convergence,
    Powell is initialized from the first fit.  Warnings are captured here so
    callers can report actual failures rather than printing thousands of
    repeated warnings during a bootstrap Monte Carlo experiment.
    """
    values = np.asarray(series, dtype=float)
    if max_iter < 1:
        raise ValueError("max_iter must be positive")

    candidates: list[object] = []
    last_error: Exception | None = None
    start_params = None
    attempts = ("lbfgs", "powell")
    attempted = 0
    for optimizer in attempts:
        attempted += 1
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", StatsmodelsConvergenceWarning)
                fitted = ARIMA(
                    values,
                    order=order,
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                ).fit(
                    start_params=start_params,
                    method_kwargs={
                        "method": optimizer,
                        "maxiter": max_iter,
                        "disp": 0,
                    },
                )
            candidates.append(fitted)
            if _arima_fit_converged(fitted):
                return fitted, attempted - 1, True
            params = np.asarray(getattr(fitted, "params", []), dtype=float)
            start_params = params if params.size and np.all(np.isfinite(params)) else None
        except Exception as exc:
            last_error = exc
            start_params = None

    if candidates:
        finite = [
            fitted
            for fitted in candidates
            if np.isfinite(float(getattr(fitted, "llf", float("-inf"))))
        ]
        best = max(
            finite or candidates,
            key=lambda fitted: float(getattr(fitted, "llf", float("-inf"))),
        )
        return best, attempted - 1, False
    raise RuntimeError(f"ARIMA{order} failed for both optimizers") from last_error


def find_arima_order(
    series: Array,
    max_p: int = 4,
    max_q: int = 4,
    *,
    max_iter: int = 500,
) -> tuple[int, int, int]:
    """Select a non-seasonal ARMA order by BIC, as in KF_0050dispred."""
    values = np.asarray(series, dtype=float)
    best_bic = np.inf
    best_order = (1, 0, 0)
    for p in range(max_p + 1):
        for q in range(max_q + 1):
            if p == 0 and q == 0:
                continue
            try:
                fitted, _, converged = fit_arima_robust(
                    values, (p, 0, q), max_iter=max_iter
                )
                if not converged:
                    continue
                if np.isfinite(fitted.bic) and fitted.bic < best_bic:
                    best_bic = float(fitted.bic)
                    best_order = (p, 0, q)
            except Exception:
                continue
    return best_order


def shortest_residual_offsets(
    residuals: Array, alpha: float, beta_grid_size: int = 101
) -> tuple[float, float, float]:
    """EnbPI line search for the shortest [beta, 1-alpha+beta] interval."""
    residuals = np.asarray(residuals, dtype=float)
    residuals = residuals[np.isfinite(residuals)]
    if len(residuals) < 2:
        raise ValueError("At least two finite residuals are required")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1)")
    betas = np.linspace(0.0, alpha, beta_grid_size)
    lower = np.quantile(residuals, betas)
    upper = np.quantile(residuals, 1.0 - alpha + betas)
    best = int(np.argmin(upper - lower))
    return float(lower[best]), float(upper[best]), float(betas[best])


@dataclass(frozen=True)
class EnbPIConfig:
    window_size: int = 15
    alpha: float = 0.05
    n_bootstrap: int = 30
    block_length: int | None = None
    batch_size: int = 1
    beta_grid_size: int = 101
    # Recenter point forecasts by mean raw OOB residuals. In "component" mode,
    # low/high are corrected separately before recombination. In "combined"
    # mode, low/high remain raw and only their combined final forecast receives
    # the correction estimated from the final residual pool.
    oob_bias_correction: bool = True
    oob_bias_correction_mode: Literal["component", "combined"] = "combined"
    # None selects (p, 0, q) once by BIC on the causal KF low training series.
    # A base ARIMA is fitted to the chronological KF-low series.  Moving blocks
    # of its one-step residuals are resampled to construct B pseudo-low series;
    # each bootstrap fit supplies parameters, while test forecasts condition
    # those parameters on the genuinely available chronological KF-low history.
    arima_order: tuple[int, int, int] | None = None
    arima_max_p: int = 4
    arima_max_q: int = 4
    arima_max_iter: int = 500
    ann_hidden_layers: tuple[int, ...] = (32, 16)
    ann_max_iter: int = 500
    ann_alpha: float = 1e-4
    ann_learning_rate_init: float = 1e-3
    # Both X and the direct high-frequency target H_t are standardized inside
    # each bootstrap ANN fit. Predictions are inverse-transformed before OOB
    # residuals are computed, so EnbPI remains on the original component scale.
    ann_target_standardization: bool = True
    # sklearn's built-in early_stopping randomly selects its validation set.
    # Keep it disabled and select max_iter with chronological rolling-origin
    # validation using only the rows sampled by the current bootstrap model.
    ann_early_stopping: bool = False
    ann_rolling_validation: bool = True
    ann_rolling_splits: int = 3
    ann_validation_fraction: float = 0.10
    ann_iteration_candidates: tuple[int, ...] | None = None
    ann_tol: float = 1e-3
    random_state: int = 1234
    process_variance: float = 0.5
    measurement_variance: float = 10.0


@dataclass
class EnbPIResult:
    model_name: str
    config: EnbPIConfig
    train_size: int
    test_times: Array
    truth: Array
    point: Array
    lower: Array
    upper: Array
    low_point: Array
    low_lower: Array
    low_upper: Array
    high_point: Array
    high_lower: Array
    high_upper: Array
    beta: Array
    low_beta: Array
    high_beta: Array
    bias_correction: Array
    low_bias_correction: Array
    high_bias_correction: Array
    initial_oob_residuals: Array
    initial_low_oob_residuals: Array
    initial_high_oob_residuals: Array
    oob_counts: Array
    estimated_low: Array
    estimated_high: Array
    observed: Array
    true_low: Array | None
    true_high: Array | None
    data_seed: int | None
    selected_arima_order: tuple[int, int, int]
    selected_ann_max_iters: Array
    ann_rolling_validation_mse: Array
    ann_nonconverged_models: int
    elapsed_seconds: float
    arima_retry_count: int = 0
    arima_nonconverged_fits: int = 0

    def metrics(self) -> dict[str, float]:
        error = self.truth - self.point
        covered = (self.truth >= self.lower) & (self.truth <= self.upper)
        width = self.upper - self.lower
        test_low_reference = self.estimated_low[self.test_times]
        test_high_reference = self.estimated_high[self.test_times]
        low_error = test_low_reference - self.low_point
        high_error = test_high_reference - self.high_point
        low_covered = (test_low_reference >= self.low_lower) & (
            test_low_reference <= self.low_upper
        )
        high_covered = (test_high_reference >= self.high_lower) & (
            test_high_reference <= self.high_upper
        )
        finite_rolling_scores = self.ann_rolling_validation_mse[
            np.isfinite(self.ann_rolling_validation_mse)
        ]
        mean_rolling_validation_mse = (
            float(np.mean(finite_rolling_scores))
            if len(finite_rolling_scores)
            else float("nan")
        )
        result = {
            "mse": float(np.mean(error**2)),
            "rmse": float(np.sqrt(np.mean(error**2))),
            "coverage": float(np.mean(covered)),
            "coverage_percent": float(100.0 * np.mean(covered)),
            "mean_width": float(np.mean(width)),
            "median_width": float(np.median(width)),
            "low_kf_mse": float(np.mean(low_error**2)),
            "low_kf_rmse": float(np.sqrt(np.mean(low_error**2))),
            "low_enbpi_coverage": float(np.mean(low_covered)),
            "low_enbpi_mean_width": float(np.mean(self.low_upper - self.low_lower)),
            "low_mean_beta": float(np.mean(self.low_beta)),
            "high_kf_mse": float(np.mean(high_error**2)),
            "high_kf_rmse": float(np.sqrt(np.mean(high_error**2))),
            "high_enbpi_coverage": float(np.mean(high_covered)),
            "high_enbpi_mean_width": float(np.mean(self.high_upper - self.high_lower)),
            "high_mean_beta": float(np.mean(self.high_beta)),
            "mean_beta": float(np.mean(self.beta)),
            "mean_bias_correction": float(np.mean(self.bias_correction)),
            "low_mean_bias_correction": float(
                np.mean(self.low_bias_correction)
            ),
            "high_mean_bias_correction": float(
                np.mean(self.high_bias_correction)
            ),
            "mean_oob_models_per_train_point": float(np.mean(self.oob_counts)),
            "min_oob_models_per_train_point": float(np.min(self.oob_counts)),
            "ann_selected_max_iter_mean": float(np.mean(self.selected_ann_max_iters)),
            "ann_rolling_validation_mse": mean_rolling_validation_mse,
            "ann_nonconverged_models": float(self.ann_nonconverged_models),
            "arima_retry_count": float(self.arima_retry_count),
            "arima_nonconverged_fits": float(self.arima_nonconverged_fits),
            "elapsed_seconds": float(self.elapsed_seconds),
        }
        if self.true_low is not None:
            true_low_error = self.true_low[self.test_times] - self.low_point
            result["low_true_mse"] = float(np.mean(true_low_error**2))
            result["low_true_rmse"] = float(np.sqrt(np.mean(true_low_error**2)))
            result["low_true_coverage"] = float(
                np.mean(
                    (self.true_low[self.test_times] >= self.low_lower)
                    & (self.true_low[self.test_times] <= self.low_upper)
                )
            )
        if self.true_high is not None:
            true_high_error = self.true_high[self.test_times] - self.high_point
            result["high_true_mse"] = float(np.mean(true_high_error**2))
            result["high_true_rmse"] = float(np.sqrt(np.mean(true_high_error**2)))
            result["high_true_coverage"] = float(
                np.mean(
                    (self.true_high[self.test_times] >= self.high_lower)
                    & (self.true_high[self.test_times] <= self.high_upper)
                )
            )
        return result

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "time": self.test_times,
                "truth": self.truth,
                "point": self.point,
                "lower": self.lower,
                "upper": self.upper,
                "low_reference": self.estimated_low[self.test_times],
                "low_point": self.low_point,
                "low_lower": self.low_lower,
                "low_upper": self.low_upper,
                "low_covered": (self.estimated_low[self.test_times] >= self.low_lower)
                & (self.estimated_low[self.test_times] <= self.low_upper),
                "low_beta": self.low_beta,
                "high_reference": self.estimated_high[self.test_times],
                "high_point": self.high_point,
                "high_lower": self.high_lower,
                "high_upper": self.high_upper,
                "high_covered": (self.estimated_high[self.test_times] >= self.high_lower)
                & (self.estimated_high[self.test_times] <= self.high_upper),
                "high_beta": self.high_beta,
                "bias_correction": self.bias_correction,
                "low_bias_correction": self.low_bias_correction,
                "high_bias_correction": self.high_bias_correction,
                "covered": (self.truth >= self.lower) & (self.truth <= self.upper),
                "width": self.upper - self.lower,
                "beta": self.beta,
            }
        )

    def summary_tables(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Separate final-series and component performance tables."""
        m = self.metrics()
        final = pd.DataFrame(
            [
                {
                    "model": self.model_name,
                    "data_seed": self.data_seed,
                    "ensemble_seed": self.config.random_state,
                    "arima_order": str(self.selected_arima_order),
                    "mse": m["mse"],
                    "rmse": m["rmse"],
                    "coverage_percent": m["coverage_percent"],
                    "mean_width": m["mean_width"],
                    "mean_beta": m["mean_beta"],
                    "mean_bias_correction": m["mean_bias_correction"],
                    "bias_correction_mode": self.config.oob_bias_correction_mode,
                    "ann_selected_max_iter_mean": m["ann_selected_max_iter_mean"],
                    "ann_rolling_validation_mse": m["ann_rolling_validation_mse"],
                    "ann_nonconverged_models": int(m["ann_nonconverged_models"]),
                    "arima_retry_count": int(m["arima_retry_count"]),
                    "arima_nonconverged_fits": int(
                        m["arima_nonconverged_fits"]
                    ),
                    "elapsed_seconds": m["elapsed_seconds"],
                }
            ]
        )
        rows = [
            {
                "component": "low_arima",
                "reference": "kalman_low",
                "mse": m["low_kf_mse"],
                "rmse": m["low_kf_rmse"],
                "coverage_percent": 100.0 * m["low_enbpi_coverage"],
                "enbpi_mean_width": m["low_enbpi_mean_width"],
                "mean_beta": m["low_mean_beta"],
                "mean_bias_correction": m["low_mean_bias_correction"],
            },
            {
                "component": "high_ann_mse",
                "reference": "kalman_high",
                "mse": m["high_kf_mse"],
                "rmse": m["high_kf_rmse"],
                "coverage_percent": 100.0 * m["high_enbpi_coverage"],
                "enbpi_mean_width": m["high_enbpi_mean_width"],
                "mean_beta": m["high_mean_beta"],
                "mean_bias_correction": m["high_mean_bias_correction"],
            },
        ]
        if "low_true_mse" in m:
            rows.append(
                {
                    "component": "low_arima",
                    "reference": "true_low",
                    "mse": m["low_true_mse"],
                    "rmse": m["low_true_rmse"],
                    "coverage_percent": 100.0 * m["low_true_coverage"],
                    "enbpi_mean_width": m["low_enbpi_mean_width"],
                    "mean_beta": m["low_mean_beta"],
                    "mean_bias_correction": m["low_mean_bias_correction"],
                }
            )
        if "high_true_mse" in m:
            rows.append(
                {
                    "component": "high_ann_mse",
                    "reference": "true_high",
                    "mse": m["high_true_mse"],
                    "rmse": m["high_true_rmse"],
                    "coverage_percent": 100.0 * m["high_true_coverage"],
                    "enbpi_mean_width": m["high_enbpi_mean_width"],
                    "mean_beta": m["high_mean_beta"],
                    "mean_bias_correction": m["high_mean_bias_correction"],
                }
            )
        return final, pd.DataFrame(rows)


@dataclass
class BootstrapHybridModel:
    """One bootstrap replicate: low-frequency ARIMA plus high-frequency ANN."""

    arima_result: object
    arima_order: tuple[int, int, int]
    ann_model: object

    def predict_low(self, low_history: Array) -> float:
        """Apply bootstrap parameters to the actual causal history through t-1."""
        available_low = np.asarray(low_history, dtype=float)
        if available_low.ndim != 1 or len(available_low) == 0:
            raise ValueError("low_history must be a non-empty one-dimensional series")
        applied = self.arima_result.apply(available_low, refit=False)
        return float(np.asarray(applied.forecast(1))[0])

    def predict_high(self, ann_features: Array) -> float:
        return float(self.ann_model.predict(np.atleast_2d(ann_features))[0])

    def predict_components(
        self, low_history: Array, ann_features: Array
    ) -> tuple[float, float]:
        return self.predict_low(low_history), self.predict_high(ann_features)


class KalmanEnbPI:
    """EnbPI with B hybrid ARIMA+ANN predictors and nested mean aggregation."""

    def __init__(self, config: EnbPIConfig):
        self.config = config
        self.models: list[BootstrapHybridModel] = []
        self.bootstrap_rows: list[Array] = []
        self.included: Array | None = None
        self.residual_pool: Array | None = None
        self.low_residual_pool: Array | None = None
        self.high_residual_pool: Array | None = None
        self.oob_counts: Array | None = None
        self.ann_nonconverged_models = 0
        self.arima_retry_count = 0
        self.arima_nonconverged_fits = 0
        self.selected_arima_order: tuple[int, int, int] | None = None
        self.selected_ann_max_iters: list[int] = []
        self.ann_rolling_validation_scores: list[float] = []
        self.base_arima_result: object | None = None
        self.arima_bootstrap_residuals: Array | None = None

    def _fit_arima(self, series: Array) -> object:
        if self.selected_arima_order is None:
            raise RuntimeError("ARIMA order must be selected before fitting")
        fitted, retries, converged = fit_arima_robust(
            series,
            self.selected_arima_order,
            max_iter=self.config.arima_max_iter,
        )
        self.arima_retry_count += retries
        self.arima_nonconverged_fits += int(not converged)
        return fitted

    def _new_ann(self, seed: int, *, max_iter: int):
        c = self.config
        # sklearn's MLPRegressor optimizes squared error, i.e. an MSE point loss.
        regressor = make_pipeline(
            StandardScaler(),
            MLPRegressor(
                hidden_layer_sizes=c.ann_hidden_layers,
                activation="relu",
                solver="adam",
                alpha=c.ann_alpha,
                learning_rate_init=c.ann_learning_rate_init,
                max_iter=max_iter,
                # Validation is handled explicitly by chronological rolling
                # folds below; this must stay False to avoid a random split.
                early_stopping=False,
                n_iter_no_change=30,
                tol=c.ann_tol,
                random_state=seed,
            ),
        )
        if not c.ann_target_standardization:
            return regressor
        return TransformedTargetRegressor(
            regressor=regressor,
            transformer=StandardScaler(),
        )

    def _ann_iteration_grid(self) -> tuple[int, ...]:
        c = self.config
        if c.ann_iteration_candidates is None:
            candidates = (max(25, c.ann_max_iter // 4), max(50, c.ann_max_iter // 2), c.ann_max_iter)
        else:
            candidates = c.ann_iteration_candidates
        cleaned = tuple(sorted({int(value) for value in candidates if int(value) > 0}))
        if not cleaned:
            raise ValueError("ann_iteration_candidates must contain a positive integer")
        return cleaned

    def _select_ann_max_iter(
        self,
        x_train: Array,
        high_targets: Array,
        bootstrap_rows: Array,
        *,
        seed: int,
    ) -> tuple[int, float]:
        """Select ANN iterations with rolling validation inside one bootstrap sample.

        Every validation target is itself a member of this model's bootstrap
        sample. Therefore the hyperparameter search does not let an OOB target
        influence a model that is later used to predict that same target.
        """
        candidates = self._ann_iteration_grid()
        c = self.config
        if not c.ann_rolling_validation or len(candidates) == 1:
            return candidates[-1], float("nan")
        if c.ann_rolling_splits < 2:
            raise ValueError("ann_rolling_splits must be at least 2")
        if not 0.0 < c.ann_validation_fraction < 0.5:
            raise ValueError("ann_validation_fraction must lie in (0, 0.5)")

        # Work on the original time indices, not on the random concatenation
        # order of the resampled blocks. Duplicate training rows retain their
        # bootstrap multiplicity, while validation is scored once per time.
        sampled_times = np.unique(np.asarray(bootstrap_rows, dtype=int))
        test_size = max(1, int(round(len(sampled_times) * c.ann_validation_fraction)))
        max_splits = (len(sampled_times) - 2) // test_size
        n_splits = min(c.ann_rolling_splits, max_splits)
        if n_splits < 2:
            return candidates[-1], float("nan")

        splitter = TimeSeriesSplit(n_splits=n_splits, test_size=test_size)
        candidate_scores: list[float] = []
        for candidate in candidates:
            fold_scores: list[float] = []
            for fold, (train_pos, val_pos) in enumerate(splitter.split(sampled_times)):
                train_times = sampled_times[train_pos]
                val_times = sampled_times[val_pos]
                fold_rows = bootstrap_rows[np.isin(bootstrap_rows, train_times)]
                if len(fold_rows) < 2 or len(val_times) == 0:
                    continue
                fold_seed = int(
                    np.random.SeedSequence([seed, fold]).generate_state(1)[0]
                )
                model = self._new_ann(fold_seed, max_iter=candidate)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", SklearnConvergenceWarning)
                    model.fit(x_train[fold_rows], high_targets[fold_rows])
                prediction = np.asarray(model.predict(x_train[val_times]), dtype=float)
                fold_scores.append(
                    float(np.mean((high_targets[val_times] - prediction) ** 2))
                )
            candidate_scores.append(
                float(np.mean(fold_scores)) if fold_scores else float("inf")
            )

        best = int(np.argmin(candidate_scores))
        if not np.isfinite(candidate_scores[best]):
            return candidates[-1], float("nan")
        return candidates[best], candidate_scores[best]

    @staticmethod
    def _arima_training_predictions(arima_result, low_series: Array, start: int) -> Array:
        """Causal one-step predictions on the original low series using fixed parameters."""
        applied = arima_result.apply(np.asarray(low_series, dtype=float), refit=False)
        predictions = np.asarray(
            applied.predict(start=start, end=len(low_series) - 1, dynamic=False), dtype=float
        )
        if len(predictions) != len(low_series) - start:
            raise RuntimeError("Unexpected ARIMA training prediction length")
        return predictions

    def fit(
        self,
        x_train: Array,
        y_train: Array,
        low_train: Array,
        high_train: Array,
        target_times: Array,
    ) -> "KalmanEnbPI":
        x_train = np.asarray(x_train, dtype=float)
        y_train = np.asarray(y_train, dtype=float)
        low_train = np.asarray(low_train, dtype=float)
        high_train = np.asarray(high_train, dtype=float)
        target_times = np.asarray(target_times, dtype=int)
        n = len(y_train)
        if x_train.ndim != 2 or len(x_train) != n or len(target_times) != n:
            raise ValueError("Training arrays must be aligned")
        block_length = self.config.block_length or max(2, int(round(np.sqrt(n))))
        rng = np.random.default_rng(self.config.random_state)
        self.selected_arima_order = self.config.arima_order or find_arima_order(
            low_train,
            max_p=self.config.arima_max_p,
            max_q=self.config.arima_max_q,
            max_iter=self.config.arima_max_iter,
        )
        # Preserve the genuine chronological low-frequency path in every
        # bootstrap replicate.  Only the base ARIMA one-step residuals are
        # resampled; directly concatenating low-level blocks would create fake
        # jumps and can shift the fitted ARIMA level (especially for prices).
        self.arima_retry_count = 0
        self.arima_nonconverged_fits = 0
        self.base_arima_result = self._fit_arima(low_train)
        base_fitted = np.asarray(self.base_arima_result.fittedvalues, dtype=float)
        if len(base_fitted) != len(low_train):
            raise RuntimeError("Unexpected base ARIMA fitted-value length")
        bootstrap_residuals = low_train[target_times] - base_fitted[target_times]
        if not np.all(np.isfinite(bootstrap_residuals)):
            raise RuntimeError("Base ARIMA produced non-finite bootstrap residuals")
        # Centering makes the bootstrap describe innovation uncertainty without
        # repeatedly adding the base fit's finite-sample mean error.
        bootstrap_residuals = bootstrap_residuals - np.mean(bootstrap_residuals)
        self.arima_bootstrap_residuals = bootstrap_residuals.copy()

        self.models, self.bootstrap_rows = [], []
        self.selected_ann_max_iters = []
        self.ann_rolling_validation_scores = []
        included = np.zeros((self.config.n_bootstrap, n), dtype=bool)
        base_low_training_predictions = np.empty((self.config.n_bootstrap, n), dtype=float)
        base_high_training_predictions = np.empty((self.config.n_bootstrap, n), dtype=float)

        for b in range(self.config.n_bootstrap):
            rows = moving_block_bootstrap_indices(n, block_length, rng)
            # The same block indices define ANN training membership/OOB status
            # and select locally consecutive ARIMA residuals.  Resampled
            # residuals are placed back on the original time grid, so the ARIMA
            # sees the real trend/order rather than concatenated level blocks.
            pseudo_low = base_fitted.copy()
            first_target = int(target_times[0])
            pseudo_low[:first_target] = low_train[:first_target]
            pseudo_low[target_times] = (
                base_fitted[target_times] + bootstrap_residuals[rows]
            )
            arima_result = self._fit_arima(pseudo_low)
            ann_seed = int(rng.integers(0, 2**31 - 1))
            high_targets = high_train[target_times]
            selected_max_iter, rolling_mse = self._select_ann_max_iter(
                x_train,
                high_targets,
                rows,
                seed=ann_seed,
            )
            ann_model = self._new_ann(ann_seed, max_iter=selected_max_iter)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", SklearnConvergenceWarning)
                ann_model.fit(x_train[rows], high_targets[rows])
            self.ann_nonconverged_models += sum(
                issubclass(item.category, SklearnConvergenceWarning)
                for item in caught
            )
            hybrid = BootstrapHybridModel(
                arima_result=arima_result,
                arima_order=self.selected_arima_order,
                ann_model=ann_model,
            )
            self.models.append(hybrid)
            self.bootstrap_rows.append(rows)
            self.selected_ann_max_iters.append(selected_max_iter)
            self.ann_rolling_validation_scores.append(rolling_mse)
            included[b, np.unique(rows)] = True

            low_prediction = self._arima_training_predictions(
                arima_result, low_train, int(target_times[0])
            )
            high_prediction = ann_model.predict(x_train)
            base_low_training_predictions[b] = low_prediction
            base_high_training_predictions[b] = high_prediction

        oob_predictions = np.empty(n, dtype=float)
        oob_low_predictions = np.empty(n, dtype=float)
        oob_high_predictions = np.empty(n, dtype=float)
        oob_counts = np.empty(n, dtype=int)
        for i in range(n):
            eligible = ~included[:, i]
            oob_counts[i] = int(eligible.sum())
            if oob_counts[i] == 0:
                raise RuntimeError(
                    "A training point has no OOB model; increase n_bootstrap or reduce block_length"
                )
            oob_low_predictions[i] = np.mean(base_low_training_predictions[eligible, i])
            oob_high_predictions[i] = np.mean(base_high_training_predictions[eligible, i])
            oob_predictions[i] = oob_low_predictions[i] + oob_high_predictions[i]

        self.included = included
        self.oob_counts = oob_counts
        self.residual_pool = y_train - oob_predictions
        self.low_residual_pool = low_train[target_times] - oob_low_predictions
        self.high_residual_pool = high_train[target_times] - oob_high_predictions
        return self

    def predict_base_components(
        self, low_history: Array, ann_features: Array
    ) -> tuple[Array, Array]:
        """Return the B low ARIMA and B high ANN point forecasts."""
        if not self.models:
            raise RuntimeError("Call fit before prediction")
        component_predictions = np.asarray(
            [model.predict_components(low_history, ann_features) for model in self.models],
            dtype=float,
        )
        return component_predictions[:, 0], component_predictions[:, 1]

    def _nested_aggregate(self, base_predictions: Array) -> float:
        """Aggregate B predictions inside each OOB set, then aggregate over i."""
        if self.included is None:
            raise RuntimeError("Call fit before prediction")
        loo_ensemble = np.empty(self.included.shape[1], dtype=float)
        for i in range(self.included.shape[1]):
            eligible = ~self.included[:, i]
            loo_ensemble[i] = np.mean(base_predictions[eligible])
        return float(np.mean(loo_ensemble))

    def predict_nested_loo(
        self, low_history: Array, ann_features: Array
    ) -> tuple[float, float, float, Array, Array]:
        """Return final/low/high nested centers plus B component forecasts."""
        low_base, high_base = self.predict_base_components(low_history, ann_features)
        final_center = self._nested_aggregate(low_base + high_base)
        low_center = self._nested_aggregate(low_base)
        high_center = self._nested_aggregate(high_base)
        return final_center, low_center, high_center, low_base, high_base


def run_kf_enbpi(
    observed: Array,
    train_size: int,
    *,
    config: EnbPIConfig,
    model_name: str = "custom",
    true_low: Array | None = None,
    true_high: Array | None = None,
    data_seed: int | None = None,
) -> EnbPIResult:
    """Fit EnbPI on train observations, then predict sequentially with feedback."""
    started = perf_counter()
    y = np.asarray(observed, dtype=float)
    if not config.window_size < train_size < len(y):
        raise ValueError("Need window_size < train_size < len(observed)")

    # The full causal decomposition is retained for diagnostics only. Test features
    # are constructed below from the information available at prediction time.
    estimated_low, estimated_high = causal_kalman_decomposition(
        y,
        process_variance=config.process_variance,
        measurement_variance=config.measurement_variance,
    )
    train_low, train_high = causal_kalman_decomposition(
        y[:train_size],
        process_variance=config.process_variance,
        measurement_variance=config.measurement_variance,
    )
    x_train, y_train, train_target_times = make_lagged_features(
        y[:train_size], train_low, train_high, config.window_size
    )
    y_test = y[train_size:]
    test_times = np.arange(train_size, len(y), dtype=int)

    enbpi = KalmanEnbPI(config).fit(
        x_train, y_train, train_low, train_high, train_target_times
    )
    residual_pool = np.array(enbpi.residual_pool, copy=True)
    low_residual_pool = np.array(enbpi.low_residual_pool, copy=True)
    high_residual_pool = np.array(enbpi.high_residual_pool, copy=True)
    pool_size = len(residual_pool)
    point = np.empty(len(y_test))
    lower = np.empty(len(y_test))
    upper = np.empty(len(y_test))
    low_point = np.empty(len(y_test))
    low_lower = np.empty(len(y_test))
    low_upper = np.empty(len(y_test))
    high_point = np.empty(len(y_test))
    high_lower = np.empty(len(y_test))
    high_upper = np.empty(len(y_test))
    betas = np.empty(len(y_test))
    low_betas = np.empty(len(y_test))
    high_betas = np.empty(len(y_test))
    bias_corrections = np.empty(len(y_test))
    low_bias_corrections = np.empty(len(y_test))
    high_bias_corrections = np.empty(len(y_test))
    raw_point = np.empty(len(y_test))
    raw_low_point = np.empty(len(y_test))
    raw_high_point = np.empty(len(y_test))
    history = list(y[:train_size])
    for batch_start in range(0, len(y_test), config.batch_size):
        batch_stop = min(batch_start + config.batch_size, len(y_test))
        pseudo_history = list(history)

        # Estimate systematic under/over-prediction from raw OOB residuals.
        # "combined" applies one correction only after low/high recombination;
        # component plots and component intervals then remain uncorrected.
        correction_mode = config.oob_bias_correction_mode
        if correction_mode not in ("component", "combined"):
            raise ValueError(
                "oob_bias_correction_mode must be 'component' or 'combined'"
            )
        if not config.oob_bias_correction:
            low_bias = high_bias = final_bias = 0.0
        elif correction_mode == "component":
            low_bias = float(np.mean(low_residual_pool))
            high_bias = float(np.mean(high_residual_pool))
            final_bias = low_bias + high_bias
        else:
            low_bias = 0.0
            high_bias = 0.0
            final_bias = float(np.mean(residual_pool))

        # Once the mean residual has moved the point forecast, remove that mean
        # before computing EnbPI offsets to avoid applying the same bias twice.
        centered_residual_pool = residual_pool - final_bias
        centered_low_residual_pool = low_residual_pool - low_bias
        centered_high_residual_pool = high_residual_pool - high_bias
        lo_offset, hi_offset, beta = shortest_residual_offsets(
            centered_residual_pool, config.alpha, config.beta_grid_size
        )
        low_lo_offset, low_hi_offset, low_beta = shortest_residual_offsets(
            centered_low_residual_pool, config.alpha, config.beta_grid_size
        )
        high_lo_offset, high_hi_offset, high_beta = shortest_residual_offsets(
            centered_high_residual_pool, config.alpha, config.beta_grid_size
        )

        # Within a no-feedback batch, lagged responses are unknown. Use recursive
        # ensemble means as pseudo observations, never the unrevealed truths.
        for j in range(batch_start, batch_stop):
            pseudo = np.asarray(pseudo_history, dtype=float)
            pseudo_low, pseudo_high = causal_kalman_decomposition(
                pseudo,
                process_variance=config.process_variance,
                measurement_variance=config.measurement_variance,
            )
            x_t = np.r_[
                pseudo_low[-config.window_size :],
                pseudo_high[-config.window_size :],
                pseudo[-config.window_size :],
            ]
            center, low_center, high_center, low_base, high_base = enbpi.predict_nested_loo(
                pseudo_low, x_t
            )
            raw_point[j] = center
            raw_low_point[j] = low_center
            raw_high_point[j] = high_center
            low_point[j] = low_center + low_bias
            high_point[j] = high_center + high_bias
            # In combined mode the final correction is intentionally not
            # allocated back to either component.
            point[j] = center + final_bias
            lower[j] = point[j] + lo_offset
            upper[j] = point[j] + hi_offset
            low_lower[j] = low_point[j] + low_lo_offset
            low_upper[j] = low_point[j] + low_hi_offset
            high_lower[j] = high_point[j] + high_lo_offset
            high_upper[j] = high_point[j] + high_hi_offset
            betas[j] = beta
            low_betas[j] = low_beta
            high_betas[j] = high_beta
            bias_corrections[j] = final_bias
            low_bias_corrections[j] = low_bias
            high_bias_corrections[j] = high_bias
            pseudo_history.append(point[j])

        # Responses are revealed together only after all forecasts in the batch.
        batch_truth = y_test[batch_start:batch_stop]
        # Keep raw model residuals in the pool.  Storing already-corrected errors
        # would make the estimated correction cancel itself on later steps.
        pending_residuals = list(
            batch_truth - raw_point[batch_start:batch_stop]
        )
        k = len(pending_residuals)
        residual_pool = np.r_[residual_pool[k:], pending_residuals][-pool_size:]
        component_times = test_times[batch_start:batch_stop]
        pending_low_residuals = (
            estimated_low[component_times]
            - raw_low_point[batch_start:batch_stop]
        )
        pending_high_residuals = (
            estimated_high[component_times]
            - raw_high_point[batch_start:batch_stop]
        )
        low_residual_pool = np.r_[low_residual_pool[k:], pending_low_residuals][-pool_size:]
        high_residual_pool = np.r_[high_residual_pool[k:], pending_high_residuals][-pool_size:]
        history.extend(batch_truth.tolist())

    return EnbPIResult(
        model_name=model_name,
        config=config,
        train_size=train_size,
        test_times=test_times,
        truth=y_test,
        point=point,
        lower=lower,
        upper=upper,
        low_point=low_point,
        low_lower=low_lower,
        low_upper=low_upper,
        high_point=high_point,
        high_lower=high_lower,
        high_upper=high_upper,
        beta=betas,
        low_beta=low_betas,
        high_beta=high_betas,
        bias_correction=bias_corrections,
        low_bias_correction=low_bias_corrections,
        high_bias_correction=high_bias_corrections,
        initial_oob_residuals=np.array(enbpi.residual_pool, copy=True),
        initial_low_oob_residuals=np.array(enbpi.low_residual_pool, copy=True),
        initial_high_oob_residuals=np.array(enbpi.high_residual_pool, copy=True),
        oob_counts=np.array(enbpi.oob_counts, copy=True),
        estimated_low=estimated_low,
        estimated_high=estimated_high,
        observed=np.array(y, copy=True),
        true_low=None if true_low is None else np.asarray(true_low),
        true_high=None if true_high is None else np.asarray(true_high),
        data_seed=data_seed,
        selected_arima_order=enbpi.selected_arima_order,
        selected_ann_max_iters=np.asarray(enbpi.selected_ann_max_iters, dtype=int),
        ann_rolling_validation_mse=np.asarray(
            enbpi.ann_rolling_validation_scores, dtype=float
        ),
        ann_nonconverged_models=enbpi.ann_nonconverged_models,
        elapsed_seconds=perf_counter() - started,
        arima_retry_count=enbpi.arima_retry_count,
        arima_nonconverged_fits=enbpi.arima_nonconverged_fits,
    )


def simulate_and_run(
    model: ModelName,
    *,
    train_size: int = 650,
    horizon: int = 50,
    config: EnbPIConfig | None = None,
    data_seed: int = 2026,
) -> EnbPIResult:
    config = config or EnbPIConfig()
    rng = np.random.default_rng(data_seed)
    low, high, observed = simulate_additive_data(model, train_size + horizon, rng=rng)
    return run_kf_enbpi(
        observed,
        train_size,
        config=config,
        model_name=model,
        true_low=low,
        true_high=high,
        data_seed=data_seed,
    )


def monte_carlo_summary(
    model: ModelName,
    *,
    n_runs: int = 20,
    train_size: int = 650,
    horizon: int = 50,
    config: EnbPIConfig | None = None,
    seed: int = 2026,
) -> tuple[pd.DataFrame, pd.DataFrame, list[EnbPIResult]]:
    """Return run metrics, mean/std summary, and trajectories for diagnostic plots."""
    config = config or EnbPIConfig()
    seed_sequence = np.random.SeedSequence(seed)
    rows = []
    results: list[EnbPIResult] = []
    for run, child in enumerate(seed_sequence.spawn(n_runs), start=1):
        data_seed = int(child.generate_state(1)[0])
        result = simulate_and_run(
            model,
            train_size=train_size,
            horizon=horizon,
            config=config,
            data_seed=data_seed,
        )
        results.append(result)
        rows.append({"run": run, "data_seed": data_seed, **result.metrics()})
    runs = pd.DataFrame(rows)
    fields = [
        "mse", "rmse", "coverage", "mean_width",
        "low_kf_mse", "low_kf_rmse", "high_kf_mse", "high_kf_rmse",
        "low_true_mse", "low_true_rmse", "high_true_mse", "high_true_rmse",
        "low_enbpi_coverage", "low_enbpi_mean_width", "low_mean_beta",
        "high_enbpi_coverage", "high_enbpi_mean_width", "high_mean_beta",
        "low_true_coverage", "high_true_coverage",
        "mean_beta", "mean_bias_correction",
        "low_mean_bias_correction", "high_mean_bias_correction",
        "ann_selected_max_iter_mean", "ann_rolling_validation_mse",
        "ann_nonconverged_models", "elapsed_seconds",
    ]
    summary = runs[fields].agg(["mean", "std"]).T.reset_index(names="metric")
    return runs, summary, results


def select_representative_run(
    runs: pd.DataFrame, results: list[EnbPIResult]
) -> tuple[int, EnbPIResult]:
    """Select the run whose final RMSE is closest to the Monte Carlo median."""
    if len(runs) != len(results) or len(results) == 0:
        raise ValueError("runs and results must be non-empty and aligned")
    position = int(np.argmin(np.abs(runs["rmse"].to_numpy() - runs["rmse"].median())))
    return int(runs.iloc[position]["run"]), results[position]


def monte_carlo_oob_residual_diagnostics(
    runs: pd.DataFrame, results: list[EnbPIResult]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Quantify OOB residual slopes and covariance cancellation for every run.

    The final OOB residual satisfies e_Y = e_L + e_H exactly.  A nearly flat
    final residual-versus-fitted trend can nevertheless arise from cancellation
    among the four fitted/residual covariance terms.  This routine reports that
    cancellation separately from cancellation (or amplification) between the
    component errors themselves.
    """
    if len(runs) != len(results) or len(results) == 0:
        raise ValueError("runs and results must be non-empty and aligned")

    def slope_and_correlation(x: Array, residual: Array) -> tuple[float, float]:
        x = np.asarray(x, dtype=float)
        residual = np.asarray(residual, dtype=float)
        finite = np.isfinite(x) & np.isfinite(residual)
        x = x[finite]
        residual = residual[finite]
        if len(x) < 2 or np.std(x) == 0.0 or np.std(residual) == 0.0:
            return float("nan"), float("nan")
        slope = float(np.polyfit(x, residual, deg=1)[0])
        correlation = float(np.corrcoef(x, residual)[0, 1])
        return slope, correlation

    diagnostic_rows: list[dict[str, float | int | bool]] = []
    for position, result in enumerate(results):
        final_residual = np.asarray(result.initial_oob_residuals, dtype=float)
        low_residual = np.asarray(result.initial_low_oob_residuals, dtype=float)
        high_residual = np.asarray(result.initial_high_oob_residuals, dtype=float)
        n_oob = len(final_residual)
        if n_oob == 0 or len(low_residual) != n_oob or len(high_residual) != n_oob:
            raise ValueError("Final, low, and high OOB residual pools must be non-empty and aligned")

        target_start = result.train_size - n_oob
        if target_start < 0:
            raise ValueError("OOB residual pool is longer than the training series")
        target_times = np.arange(target_start, result.train_size, dtype=int)
        final_target = np.asarray(result.observed, dtype=float)[target_times]
        low_target = np.asarray(result.estimated_low, dtype=float)[target_times]
        high_target = np.asarray(result.estimated_high, dtype=float)[target_times]
        final_fitted = final_target - final_residual
        low_fitted = low_target - low_residual
        high_fitted = high_target - high_residual

        final_slope, final_corr = slope_and_correlation(final_fitted, final_residual)
        low_branch_slope, low_branch_corr = slope_and_correlation(
            low_fitted, low_residual
        )
        high_branch_slope, high_branch_corr = slope_and_correlation(
            high_fitted, high_residual
        )
        final_by_low_slope, final_by_low_corr = slope_and_correlation(
            low_fitted, final_residual
        )
        final_by_high_slope, final_by_high_corr = slope_and_correlation(
            high_fitted, final_residual
        )

        cov_low_fit_low_error = float(
            np.cov(low_fitted, low_residual, ddof=0)[0, 1]
        )
        cov_low_fit_high_error = float(
            np.cov(low_fitted, high_residual, ddof=0)[0, 1]
        )
        cov_high_fit_low_error = float(
            np.cov(high_fitted, low_residual, ddof=0)[0, 1]
        )
        cov_high_fit_high_error = float(
            np.cov(high_fitted, high_residual, ddof=0)[0, 1]
        )
        covariance_terms = np.asarray(
            [
                cov_low_fit_low_error,
                cov_low_fit_high_error,
                cov_high_fit_low_error,
                cov_high_fit_high_error,
            ],
            dtype=float,
        )
        fitted_residual_cov_total = float(np.sum(covariance_terms))
        covariance_abs_sum = float(np.sum(np.abs(covariance_terms)))
        covariance_cancellation_fraction = (
            1.0 - abs(fitted_residual_cov_total) / covariance_abs_sum
            if covariance_abs_sum > 0.0
            else float("nan")
        )

        low_error_mse = float(np.mean(low_residual**2))
        high_error_mse = float(np.mean(high_residual**2))
        error_cross_term = float(2.0 * np.mean(low_residual * high_residual))
        combined_oob_mse = float(np.mean(final_residual**2))
        component_error_corr = (
            float(np.corrcoef(low_residual, high_residual)[0, 1])
            if np.std(low_residual) > 0.0 and np.std(high_residual) > 0.0
            else float("nan")
        )

        # Diagnose whether high-frequency error magnitude changes with the
        # previous low state. The coupled M1M9 DGP should exhibit this relation;
        # the constant-high-variance control should not.
        abs_kf_low_lag = np.abs(
            np.asarray(result.estimated_low, dtype=float)[target_times - 1]
        )
        _, high_abs_error_by_abs_kf_low_corr = slope_and_correlation(
            abs_kf_low_lag, np.abs(high_residual)
        )
        _, high_squared_error_by_abs_kf_low_corr = slope_and_correlation(
            abs_kf_low_lag, high_residual**2
        )
        low_state_q25, low_state_q75 = np.quantile(abs_kf_low_lag, [0.25, 0.75])
        calm_state = abs_kf_low_lag <= low_state_q25
        extreme_state = abs_kf_low_lag >= low_state_q75
        calm_high_mse = float(np.mean(high_residual[calm_state] ** 2))
        extreme_high_mse = float(np.mean(high_residual[extreme_state] ** 2))
        high_oob_mse_extreme_to_calm_low_state_ratio = (
            extreme_high_mse / calm_high_mse
            if calm_high_mse > 0.0
            else float("nan")
        )

        true_innovation_squared_by_abs_true_low_corr = float("nan")
        true_innovation_mse_extreme_to_calm_low_state_ratio = float("nan")
        if result.true_low is not None and result.true_high is not None:
            true_low = np.asarray(result.true_low, dtype=float)
            true_high = np.asarray(result.true_high, dtype=float)
            h1 = true_high[target_times - 1]
            if result.model_name.lower() in (
                "m1m9",
                "m1m9_constant_high_variance",
                "m1m9_garch_low_constant_high_variance",
            ):
                gate = 1.0 / (
                    1.0 + np.exp(np.clip(-10.0 * h1, -700.0, 700.0))
                )
                high_conditional_mean = 0.8 * h1 - 0.8 * h1 * gate
            elif result.model_name.lower() == "m1m3":
                h2 = true_high[target_times - 2]
                high_conditional_mean = (
                    (0.5 + 0.9 * np.exp(-(h1**2))) * h1
                    + (-0.8 - 1.8 * np.exp(-(h1**2))) * h2
                )
            else:
                high_conditional_mean = None

            if high_conditional_mean is not None:
                true_high_innovation = (
                    true_high[target_times] - high_conditional_mean
                )
                abs_true_low_lag = np.abs(true_low[target_times - 1])
                (
                    _,
                    true_innovation_squared_by_abs_true_low_corr,
                ) = slope_and_correlation(
                    abs_true_low_lag, true_high_innovation**2
                )
                true_q25, true_q75 = np.quantile(abs_true_low_lag, [0.25, 0.75])
                true_calm = abs_true_low_lag <= true_q25
                true_extreme = abs_true_low_lag >= true_q75
                true_calm_mse = float(
                    np.mean(true_high_innovation[true_calm] ** 2)
                )
                true_extreme_mse = float(
                    np.mean(true_high_innovation[true_extreme] ** 2)
                )
                true_innovation_mse_extreme_to_calm_low_state_ratio = (
                    true_extreme_mse / true_calm_mse
                    if true_calm_mse > 0.0
                    else float("nan")
                )

        diagnostic_rows.append(
            {
                "run": int(runs.iloc[position]["run"]),
                "data_seed": int(runs.iloc[position]["data_seed"]),
                "final_mean_residual": float(np.mean(final_residual)),
                "low_mean_residual": float(np.mean(low_residual)),
                "high_mean_residual": float(np.mean(high_residual)),
                "final_fitted_residual_slope": final_slope,
                "final_fitted_residual_corr": final_corr,
                "low_branch_residual_slope": low_branch_slope,
                "low_branch_residual_corr": low_branch_corr,
                "high_branch_residual_slope": high_branch_slope,
                "high_branch_residual_corr": high_branch_corr,
                "final_residual_by_low_slope": final_by_low_slope,
                "final_residual_by_low_corr": final_by_low_corr,
                "final_residual_by_high_slope": final_by_high_slope,
                "final_residual_by_high_corr": final_by_high_corr,
                "opposite_branch_slope_signs": bool(
                    np.isfinite(low_branch_slope)
                    and np.isfinite(high_branch_slope)
                    and low_branch_slope * high_branch_slope < 0.0
                ),
                "opposite_final_component_slope_signs": bool(
                    np.isfinite(final_by_low_slope)
                    and np.isfinite(final_by_high_slope)
                    and final_by_low_slope * final_by_high_slope < 0.0
                ),
                "component_error_corr": component_error_corr,
                "low_error_mse": low_error_mse,
                "high_error_mse": high_error_mse,
                "component_error_cross_term": error_cross_term,
                "component_error_cross_term_share": (
                    error_cross_term / combined_oob_mse
                    if combined_oob_mse > 0.0
                    else float("nan")
                ),
                "combined_oob_mse": combined_oob_mse,
                "high_abs_error_by_abs_kf_low_corr": (
                    high_abs_error_by_abs_kf_low_corr
                ),
                "high_squared_error_by_abs_kf_low_corr": (
                    high_squared_error_by_abs_kf_low_corr
                ),
                "high_oob_mse_extreme_to_calm_low_state_ratio": (
                    high_oob_mse_extreme_to_calm_low_state_ratio
                ),
                "true_innovation_squared_by_abs_true_low_corr": (
                    true_innovation_squared_by_abs_true_low_corr
                ),
                "true_innovation_mse_extreme_to_calm_low_state_ratio": (
                    true_innovation_mse_extreme_to_calm_low_state_ratio
                ),
                "cov_low_fit_low_error": cov_low_fit_low_error,
                "cov_low_fit_high_error": cov_low_fit_high_error,
                "cov_high_fit_low_error": cov_high_fit_low_error,
                "cov_high_fit_high_error": cov_high_fit_high_error,
                "fitted_residual_cov_total": fitted_residual_cov_total,
                "covariance_cancellation_fraction": covariance_cancellation_fraction,
                "residual_identity_max_abs_error": float(
                    np.max(np.abs(final_residual - low_residual - high_residual))
                ),
            }
        )

    diagnostics = pd.DataFrame(diagnostic_rows)
    summary_fields = [
        "final_fitted_residual_slope",
        "low_branch_residual_slope",
        "high_branch_residual_slope",
        "final_residual_by_low_slope",
        "final_residual_by_high_slope",
        "component_error_corr",
        "component_error_cross_term",
        "component_error_cross_term_share",
        "combined_oob_mse",
        "high_abs_error_by_abs_kf_low_corr",
        "high_squared_error_by_abs_kf_low_corr",
        "high_oob_mse_extreme_to_calm_low_state_ratio",
        "true_innovation_squared_by_abs_true_low_corr",
        "true_innovation_mse_extreme_to_calm_low_state_ratio",
        "fitted_residual_cov_total",
        "covariance_cancellation_fraction",
    ]
    summary = (
        diagnostics[summary_fields]
        .agg(["mean", "std"])
        .T.reset_index(names="metric")
    )
    sign_rates = pd.DataFrame(
        [
            {
                "metric": "opposite_branch_slope_sign_rate",
                "mean": float(diagnostics["opposite_branch_slope_signs"].mean()),
                "std": float("nan"),
            },
            {
                "metric": "opposite_final_component_slope_sign_rate",
                "mean": float(
                    diagnostics["opposite_final_component_slope_signs"].mean()
                ),
                "std": float("nan"),
            },
        ]
    )
    summary = pd.concat([summary, sign_rates], ignore_index=True)
    return diagnostics, summary


def plot_monte_carlo_forecast_diagnostics(
    runs: pd.DataFrame,
    results: list[EnbPIResult],
    *,
    model_name: str,
    alpha: float = 0.05,
):
    """Plot a representative trajectory and horizon-wise Monte Carlo diagnostics."""
    import matplotlib.pyplot as plt

    run_number, representative = select_representative_run(runs, results)
    detailed_fig, detailed_axes = plot_result(representative)
    detailed_fig.suptitle(
        f"Representative Monte Carlo run {run_number}: {model_name.upper()} "
        f"(RMSE closest to median)",
        fontsize=14,
        y=1.002,
    )
    detailed_fig.tight_layout()

    horizon = len(results[0].truth)
    if any(len(result.truth) != horizon for result in results):
        raise ValueError("All Monte Carlo results must use the same forecast horizon")
    errors = np.vstack([result.truth - result.point for result in results])
    covered = np.vstack(
        [(result.truth >= result.lower) & (result.truth <= result.upper) for result in results]
    )
    widths = np.vstack([result.upper - result.lower for result in results])
    steps = np.arange(1, horizon + 1)
    step_rmse = np.sqrt(np.mean(errors**2, axis=0))
    step_coverage = 100.0 * np.mean(covered, axis=0)
    step_width = np.mean(widths, axis=0)

    aggregate_fig, axes = plt.subplots(3, 1, figsize=(14, 11), sharex=True)
    axes[0].plot(steps, step_rmse, marker="o", linewidth=1.3)
    axes[0].axhline(np.sqrt(np.mean(errors**2)), color="tab:red", linestyle="--",
                    label=f"Overall RMSE={np.sqrt(np.mean(errors**2)):.3f}")
    axes[0].set_ylabel("RMSE")
    axes[0].set_title("Point-prediction accuracy at each forecast step")
    axes[0].legend()

    axes[1].bar(steps, step_coverage, color="tab:green", alpha=0.75)
    axes[1].axhline(100.0 * (1.0 - alpha), color="tab:red", linestyle="--",
                    label=f"Target={100.0*(1.0-alpha):.1f}%")
    axes[1].axhline(100.0 * covered.mean(), color="black", linestyle=":",
                    label=f"Overall={100.0*covered.mean():.2f}%")
    axes[1].set_ylim(0.0, 105.0)
    axes[1].set_ylabel("Coverage (%)")
    axes[1].set_title("Empirical EnbPI coverage at each forecast step")
    axes[1].legend()

    axes[2].plot(steps, step_width, marker="o", color="tab:purple", linewidth=1.3)
    axes[2].axhline(widths.mean(), color="tab:red", linestyle="--",
                    label=f"Overall mean width={widths.mean():.3f}")
    axes[2].set_xlabel("Forecast step")
    axes[2].set_ylabel("Mean interval width")
    axes[2].set_title("Mean EnbPI interval width at each forecast step")
    axes[2].legend()
    for axis in axes:
        axis.grid(alpha=0.25)
    aggregate_fig.suptitle(
        f"Monte Carlo forecast diagnostics: {model_name.upper()} (R={len(results)})",
        fontsize=14,
    )
    aggregate_fig.tight_layout(rect=(0, 0, 1, 0.97))
    return (detailed_fig, detailed_axes), (aggregate_fig, axes)


def plot_monte_carlo_summary(
    runs: pd.DataFrame,
    *,
    model_name: str,
    alpha: float = 0.05,
):
    """Visualize final and component performance over Monte Carlo repetitions."""
    import matplotlib.pyplot as plt

    required = {
        "run", "rmse", "coverage", "mean_width",
        "low_kf_rmse", "high_kf_rmse", "low_true_rmse", "high_true_rmse",
    }
    missing = required.difference(runs.columns)
    if missing:
        raise ValueError(f"runs is missing Monte Carlo metrics: {sorted(missing)}")

    x = runs["run"].to_numpy()
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].plot(x, runs["rmse"], marker="o", linewidth=1.2)
    axes[0, 0].axhline(runs["rmse"].mean(), color="tab:red", linestyle="--",
                       label=f"Mean={runs['rmse'].mean():.3f}")
    axes[0, 0].set_title("Final forecast RMSE by run")
    axes[0, 0].set_xlabel("Monte Carlo run")
    axes[0, 0].set_ylabel("RMSE")
    axes[0, 0].legend()

    coverage_percent = 100.0 * runs["coverage"].to_numpy()
    target_percent = 100.0 * (1.0 - alpha)
    axes[0, 1].bar(x, coverage_percent, color="tab:green", alpha=0.75)
    axes[0, 1].axhline(target_percent, color="tab:red", linestyle="--",
                       label=f"Target={target_percent:.1f}%")
    axes[0, 1].axhline(coverage_percent.mean(), color="black", linestyle=":",
                       label=f"Mean={coverage_percent.mean():.2f}%")
    axes[0, 1].set_ylim(0.0, 105.0)
    axes[0, 1].set_title("EnbPI coverage by run")
    axes[0, 1].set_xlabel("Monte Carlo run")
    axes[0, 1].set_ylabel("Coverage (%)")
    axes[0, 1].legend()

    axes[1, 0].plot(x, runs["mean_width"], marker="o", color="tab:purple", linewidth=1.2)
    axes[1, 0].axhline(runs["mean_width"].mean(), color="tab:red", linestyle="--",
                       label=f"Mean={runs['mean_width'].mean():.3f}")
    axes[1, 0].set_title("Mean EnbPI width by run")
    axes[1, 0].set_xlabel("Monte Carlo run")
    axes[1, 0].set_ylabel("Mean interval width")
    axes[1, 0].legend()

    component_columns = ["low_kf_rmse", "high_kf_rmse", "low_true_rmse", "high_true_rmse"]
    component_labels = ["Low vs KF", "High vs KF", "Low vs true", "High vs true"]
    values = [runs[column].dropna().to_numpy() for column in component_columns]
    box = axes[1, 1].boxplot(values, tick_labels=component_labels, patch_artist=True)
    colors = ["tab:blue", "tab:orange", "lightskyblue", "moccasin"]
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    axes[1, 1].set_title("Component RMSE distributions")
    axes[1, 1].set_ylabel("RMSE")
    axes[1, 1].tick_params(axis="x", rotation=15)

    for axis in axes.flat:
        axis.grid(alpha=0.25)
    fig.suptitle(
        f"Monte Carlo summary: {model_name.upper()} (R={len(runs)})",
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig, axes


def plot_oob_residual_diagnostics(result: EnbPIResult):
    """Plot initial training OOB residuals against their OOB fitted values.

    The component panels use the KF-decomposed training targets because those
    are the targets actually fitted by the low ARIMA and high ANN branches.
    These are raw OOB residuals, before any online test feedback is appended.
    """
    import matplotlib.pyplot as plt

    final_residual = np.asarray(result.initial_oob_residuals, dtype=float)
    low_residual = np.asarray(result.initial_low_oob_residuals, dtype=float)
    high_residual = np.asarray(result.initial_high_oob_residuals, dtype=float)
    n_oob = len(final_residual)
    if n_oob == 0:
        raise ValueError("No initial OOB residuals are available")
    if len(low_residual) != n_oob or len(high_residual) != n_oob:
        raise ValueError("Final, low, and high OOB residual pools must have equal length")

    # make_lagged_features creates one target for every training time after the
    # initial lag window, so the OOB residual pools align with this trailing
    # portion of the training series.
    target_start = result.train_size - n_oob
    if target_start < 0:
        raise ValueError("OOB residual pool is longer than the training series")
    target_times = np.arange(target_start, result.train_size, dtype=int)

    final_target = np.asarray(result.observed, dtype=float)[target_times]
    low_target = np.asarray(result.estimated_low, dtype=float)[target_times]
    high_target = np.asarray(result.estimated_high, dtype=float)[target_times]
    final_fitted = final_target - final_residual
    low_fitted = low_target - low_residual
    high_fitted = high_target - high_residual

    panels = (
        (
            final_fitted,
            final_residual,
            r"Combined OOB fitted $\hat{Y}_{-i}$",
            r"Combined residual $Y_i-\hat{Y}_{-i}$",
            "Final hybrid",
            "tab:green",
        ),
        (
            low_fitted,
            low_residual,
            r"Linear/low OOB fitted $\hat{L}_{-i}$",
            r"Low residual $L_i^{KF}-\hat{L}_{-i}$",
            "Linear branch: ARIMA",
            "tab:blue",
        ),
        (
            high_fitted,
            high_residual,
            r"Nonlinear/high OOB fitted $\hat{H}_{-i}$",
            r"High residual $H_i^{KF}-\hat{H}_{-i}$",
            "Nonlinear branch: ANN",
            "tab:orange",
        ),
    )

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    for axis, (fitted, residual, xlabel, ylabel, title, color) in zip(axes, panels):
        finite = np.isfinite(fitted) & np.isfinite(residual)
        x = fitted[finite]
        e = residual[finite]
        if len(x) == 0:
            raise ValueError(f"No finite values are available for {title}")

        axis.scatter(x, e, s=24, alpha=0.58, color=color, edgecolors="none")
        axis.axhline(0.0, color="black", linestyle="--", linewidth=1.1)

        correlation = float("nan")
        if len(x) >= 2 and np.std(x) > 0 and np.std(e) > 0:
            correlation = float(np.corrcoef(x, e)[0, 1])
            slope, intercept = np.polyfit(x, e, deg=1)
            order = np.argsort(x)
            axis.plot(
                x[order],
                intercept + slope * x[order],
                color="tab:red",
                linewidth=1.5,
                label="Linear diagnostic trend",
            )
            axis.legend(loc="best")

        correlation_text = (
            f"{correlation:.3f}" if np.isfinite(correlation) else "undefined"
        )
        axis.text(
            0.03,
            0.97,
            f"mean residual = {np.mean(e):.4f}\n"
            f"corr(fitted, residual) = {correlation_text}",
            transform=axis.transAxes,
            va="top",
            bbox={"facecolor": "white", "alpha": 0.78, "edgecolor": "0.75"},
        )
        axis.set_title(title)
        axis.set_xlabel(xlabel)
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25)

    fig.suptitle(
        f"Initial training OOB residual diagnostics: {result.model_name.upper()}",
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig, axes


def plot_combined_oob_residual_by_component_fitted(result: EnbPIResult):
    """Relate the final OOB residual to each pre-combination fitted component.

    These plots diagnose whether the error of the final hybrid forecast changes
    systematically with the predicted low or high component. They complement,
    but do not replace, the component-specific residual plots because a pattern
    here can be caused by either branch or by dependence between the branches.
    """
    import matplotlib.pyplot as plt

    final_residual = np.asarray(result.initial_oob_residuals, dtype=float)
    low_residual = np.asarray(result.initial_low_oob_residuals, dtype=float)
    high_residual = np.asarray(result.initial_high_oob_residuals, dtype=float)
    n_oob = len(final_residual)
    if n_oob == 0:
        raise ValueError("No initial OOB residuals are available")
    if len(low_residual) != n_oob or len(high_residual) != n_oob:
        raise ValueError("Final, low, and high OOB residual pools must have equal length")

    target_start = result.train_size - n_oob
    if target_start < 0:
        raise ValueError("OOB residual pool is longer than the training series")
    target_times = np.arange(target_start, result.train_size, dtype=int)
    low_target = np.asarray(result.estimated_low, dtype=float)[target_times]
    high_target = np.asarray(result.estimated_high, dtype=float)[target_times]
    low_fitted = low_target - low_residual
    high_fitted = high_target - high_residual

    panels = (
        (
            low_fitted,
            r"Linear/low OOB fitted $\hat{L}_{-i}$",
            "Final residual conditioned on ARIMA fitted value",
            "tab:blue",
        ),
        (
            high_fitted,
            r"Nonlinear/high OOB fitted $\hat{H}_{-i}$",
            "Final residual conditioned on ANN fitted value",
            "tab:orange",
        ),
    )

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for axis, (fitted, xlabel, title, color) in zip(axes, panels):
        finite = np.isfinite(fitted) & np.isfinite(final_residual)
        x = fitted[finite]
        e = final_residual[finite]
        if len(x) == 0:
            raise ValueError(f"No finite values are available for {title}")

        axis.scatter(x, e, s=24, alpha=0.58, color=color, edgecolors="none")
        axis.axhline(0.0, color="black", linestyle="--", linewidth=1.1)

        correlation = float("nan")
        if len(x) >= 2 and np.std(x) > 0 and np.std(e) > 0:
            correlation = float(np.corrcoef(x, e)[0, 1])
            slope, intercept = np.polyfit(x, e, deg=1)
            order = np.argsort(x)
            axis.plot(
                x[order],
                intercept + slope * x[order],
                color="tab:red",
                linewidth=1.5,
                label="Linear diagnostic trend",
            )
            axis.legend(loc="best")

        correlation_text = (
            f"{correlation:.3f}" if np.isfinite(correlation) else "undefined"
        )
        axis.text(
            0.03,
            0.97,
            f"mean final residual = {np.mean(e):.4f}\n"
            f"corr(component fitted, final residual) = {correlation_text}",
            transform=axis.transAxes,
            va="top",
            bbox={"facecolor": "white", "alpha": 0.78, "edgecolor": "0.75"},
        )
        axis.set_title(title)
        axis.set_xlabel(xlabel)
        axis.set_ylabel(r"Combined residual $Y_i-\hat{Y}_{-i}$")
        axis.grid(alpha=0.25)

    fig.suptitle(
        "Combined training OOB residual versus pre-combination fitted components: "
        f"{result.model_name.upper()}",
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return fig, axes


def plot_representative_monte_carlo_oob_residuals(
    runs: pd.DataFrame,
    results: list[EnbPIResult],
    *,
    model_name: str,
):
    """Plot both OOB residual diagnostics for the median-RMSE MC run."""
    run_number, representative = select_representative_run(runs, results)

    component_fig, component_axes = plot_oob_residual_diagnostics(representative)
    component_fig.suptitle(
        f"Representative Monte Carlo run {run_number}: {model_name.upper()} "
        "initial training OOB residual diagnostics\n"
        "(final RMSE closest to Monte Carlo median)",
        fontsize=14,
    )
    component_fig.tight_layout(rect=(0, 0, 1, 0.90))

    combined_fig, combined_axes = (
        plot_combined_oob_residual_by_component_fitted(representative)
    )
    combined_fig.suptitle(
        f"Representative Monte Carlo run {run_number}: {model_name.upper()} "
        "combined OOB residual by fitted component\n"
        "(final RMSE closest to Monte Carlo median)",
        fontsize=14,
    )
    combined_fig.tight_layout(rect=(0, 0, 1, 0.88))

    return (component_fig, component_axes), (combined_fig, combined_axes)


def plot_result(result: EnbPIResult):
    import matplotlib.pyplot as plt

    frame = result.frame()
    full_time = np.arange(len(result.observed))
    fig, axes = plt.subplots(5, 1, figsize=(15, 20))

    axes[0].plot(full_time, result.observed, alpha=0.7, label="Mixed data")
    if result.true_low is not None:
        axes[0].plot(full_time, result.true_low, label="True low component")
    axes[0].plot(full_time, result.estimated_low, label="KF estimated low")
    axes[0].axvline(result.train_size, color="gray", linestyle="--", label="Forecast start")
    axes[0].set_ylabel("Low / signal")
    axes[0].set_title(f"KF decomposition and EnbPI forecasts: {result.model_name.upper()}")
    axes[0].legend(loc="best")

    if result.true_high is not None:
        axes[1].plot(full_time, result.true_high, label="True high component")
    axes[1].plot(full_time, result.estimated_high, label="KF estimated high")
    axes[1].axvline(result.train_size, color="gray", linestyle="--", label="Forecast start")
    axes[1].set_ylabel("High / residual")
    axes[1].legend(loc="best")

    axes[2].plot(frame["time"], frame["low_reference"], label="KF low reference")
    axes[2].plot(frame["time"], frame["low_point"], label="Nested-LOO ARIMA low forecast")
    axes[2].fill_between(
        frame["time"], frame["low_lower"], frame["low_upper"], alpha=0.22,
        label=f"Low EnbPI {100 * (1-result.config.alpha):.0f}% interval",
    )
    axes[2].set_ylabel("Low")
    axes[2].legend(loc="best")

    axes[3].plot(frame["time"], frame["high_reference"], label="KF high reference")
    axes[3].plot(
        frame["time"],
        frame["high_point"],
        label="Nested-LOO MSE-ANN high forecast",
    )
    axes[3].fill_between(
        frame["time"], frame["high_lower"], frame["high_upper"], alpha=0.22,
        color="tab:orange",
        label=f"High EnbPI {100 * (1-result.config.alpha):.0f}% interval",
    )
    axes[3].set_ylabel("High")
    axes[3].legend(loc="best")

    if result.true_low is not None and result.true_high is not None:
        clean = result.true_low[result.test_times] + result.true_high[result.test_times]
        axes[4].plot(frame["time"], clean, label="True clean signal")
    axes[4].plot(frame["time"], frame["truth"], label="True mixed data")
    axes[4].plot(frame["time"], frame["point"], color="tab:green", label="Hybrid point forecast")
    axes[4].fill_between(
        frame["time"], frame["lower"], frame["upper"], color="gray", alpha=0.28,
        label=f"EnbPI {100 * (1-result.config.alpha):.0f}% interval",
    )
    axes[4].set_ylabel("Final forecast")
    axes[4].set_xlabel("Time")
    axes[4].legend(loc="best")

    for axis in axes:
        axis.grid(alpha=0.25)
    fig.tight_layout()
    return fig, axes
