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
from sklearn.neural_network import MLPRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.arima.model import ARIMA


Array = np.ndarray
ModelName = Literal["m1m3", "m1m9"]


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


def simulate_additive_data(
    model: ModelName, n_steps: int, *, rng: np.random.Generator
) -> tuple[Array, Array, Array]:
    if model == "m1m3":
        return simulate_m1_m3_additive_data(n_steps, rng=rng)
    if model == "m1m9":
        return simulate_m1_m9_additive_data(n_steps, rng=rng)
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


def find_arima_order(
    series: Array, max_p: int = 4, max_q: int = 4
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
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    fitted = ARIMA(
                        values,
                        order=(p, 0, q),
                        enforce_stationarity=False,
                        enforce_invertibility=False,
                    ).fit()
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
    # None selects (p, 0, q) once by BIC on the causal KF low training series.
    # The selected order is then re-fitted at every forecast time point.
    arima_order: tuple[int, int, int] | None = None
    arima_max_p: int = 4
    arima_max_q: int = 4
    ann_hidden_layers: tuple[int, ...] = (32, 16)
    ann_max_iter: int = 500
    ann_alpha: float = 1e-4
    ann_learning_rate_init: float = 1e-3
    ann_early_stopping: bool = True
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
    ann_nonconverged_models: int
    elapsed_seconds: float

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
            "mean_oob_models_per_train_point": float(np.mean(self.oob_counts)),
            "min_oob_models_per_train_point": float(np.min(self.oob_counts)),
            "ann_nonconverged_models": float(self.ann_nonconverged_models),
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
                    "ann_nonconverged_models": int(m["ann_nonconverged_models"]),
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
            },
            {
                "component": "high_ann_mse",
                "reference": "kalman_high",
                "mse": m["high_kf_mse"],
                "rmse": m["high_kf_rmse"],
                "coverage_percent": 100.0 * m["high_enbpi_coverage"],
                "enbpi_mean_width": m["high_enbpi_mean_width"],
                "mean_beta": m["high_mean_beta"],
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
                }
            )
        return final, pd.DataFrame(rows)


@dataclass
class BootstrapHybridModel:
    """One bootstrap replicate: low-frequency ARIMA plus high-frequency ANN."""

    arima_result: object
    arima_order: tuple[int, int, int]
    bootstrap_low_history: Array
    original_train_length: int
    ann_model: object

    def predict_low(self, low_history: Array) -> float:
        """Re-fit this bootstrap ARIMA at each step on its expanding history."""
        available_low = np.asarray(low_history, dtype=float)
        if len(available_low) < self.original_train_length:
            raise ValueError("low_history is shorter than the original training series")
        # Preserve bootstrap-model diversity in the training portion, then append
        # only low-frequency states revealed (or recursively formed) after train.
        rolling_history = np.r_[
            self.bootstrap_low_history,
            available_low[self.original_train_length :],
        ]
        candidate_orders = [self.arima_order]
        for fallback in ((1, 0, 0), (0, 0, 1), (0, 0, 0)):
            if fallback not in candidate_orders:
                candidate_orders.append(fallback)
        last_error: Exception | None = None
        for order in candidate_orders:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    fitted = ARIMA(
                        rolling_history,
                        order=order,
                        enforce_stationarity=False,
                        enforce_invertibility=False,
                    ).fit()
                return float(np.asarray(fitted.forecast(1))[0])
            except Exception as exc:
                last_error = exc
        if last_error is not None:
            raise last_error
        raise RuntimeError("No ARIMA order was available for rolling prediction")

    def predict_high(self, ann_features: Array) -> float:
        return float(self.ann_model.predict(np.atleast_2d(ann_features))[0])

    def predict_components(self, low_history: Array, ann_features: Array) -> tuple[float, float]:
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
        self.selected_arima_order: tuple[int, int, int] | None = None

    def _new_ann(self, seed: int):
        c = self.config
        # sklearn's MLPRegressor optimizes squared error, i.e. an MSE point loss.
        return make_pipeline(
            StandardScaler(),
            MLPRegressor(
                hidden_layer_sizes=c.ann_hidden_layers,
                activation="relu",
                solver="adam",
                alpha=c.ann_alpha,
                learning_rate_init=c.ann_learning_rate_init,
                max_iter=c.ann_max_iter,
                early_stopping=c.ann_early_stopping,
                validation_fraction=0.1,
                n_iter_no_change=30,
                tol=c.ann_tol,
                random_state=seed,
            ),
        )

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
        )
        self.models, self.bootstrap_rows = [], []
        included = np.zeros((self.config.n_bootstrap, n), dtype=bool)
        base_low_training_predictions = np.empty((self.config.n_bootstrap, n), dtype=float)
        base_high_training_predictions = np.empty((self.config.n_bootstrap, n), dtype=float)

        for b in range(self.config.n_bootstrap):
            rows = moving_block_bootstrap_indices(n, block_length, rng)
            # Consecutive blocks preserve local order; concatenating sampled blocks
            # gives the bootstrap low-frequency series used to fit this ARIMA.
            boot_low = low_train[target_times[rows]]
            arima_result = ARIMA(
                boot_low,
                order=self.selected_arima_order,
                enforce_stationarity=False,
                enforce_invertibility=False,
            ).fit()
            ann_model = self._new_ann(int(rng.integers(0, 2**31 - 1)))
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", ConvergenceWarning)
                ann_model.fit(x_train[rows], high_train[target_times[rows]])
            self.ann_nonconverged_models += sum(
                issubclass(item.category, ConvergenceWarning) for item in caught
            )
            hybrid = BootstrapHybridModel(
                arima_result=arima_result,
                arima_order=self.selected_arima_order,
                bootstrap_low_history=np.asarray(boot_low, dtype=float),
                original_train_length=len(low_train),
                ann_model=ann_model,
            )
            self.models.append(hybrid)
            self.bootstrap_rows.append(rows)
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
    history = list(y[:train_size])
    for batch_start in range(0, len(y_test), config.batch_size):
        batch_stop = min(batch_start + config.batch_size, len(y_test))
        pseudo_history = list(history)
        lo_offset, hi_offset, beta = shortest_residual_offsets(
            residual_pool, config.alpha, config.beta_grid_size
        )
        low_lo_offset, low_hi_offset, low_beta = shortest_residual_offsets(
            low_residual_pool, config.alpha, config.beta_grid_size
        )
        high_lo_offset, high_hi_offset, high_beta = shortest_residual_offsets(
            high_residual_pool, config.alpha, config.beta_grid_size
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
            point[j] = center
            lower[j] = center + lo_offset
            upper[j] = center + hi_offset
            low_point[j] = low_center
            low_lower[j] = low_center + low_lo_offset
            low_upper[j] = low_center + low_hi_offset
            high_point[j] = high_center
            high_lower[j] = high_center + high_lo_offset
            high_upper[j] = high_center + high_hi_offset
            betas[j] = beta
            low_betas[j] = low_beta
            high_betas[j] = high_beta
            pseudo_history.append(center)

        # Responses are revealed together only after all forecasts in the batch.
        batch_truth = y_test[batch_start:batch_stop]
        batch_points = point[batch_start:batch_stop]
        # The online residual uses exactly the same nested LOO center used when
        # the interval was issued; no all-model shortcut is used here.
        pending_residuals = list(batch_truth - batch_points)
        k = len(pending_residuals)
        residual_pool = np.r_[residual_pool[k:], pending_residuals][-pool_size:]
        component_times = test_times[batch_start:batch_stop]
        pending_low_residuals = estimated_low[component_times] - low_point[batch_start:batch_stop]
        pending_high_residuals = estimated_high[component_times] - high_point[batch_start:batch_stop]
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
        ann_nonconverged_models=enbpi.ann_nonconverged_models,
        elapsed_seconds=perf_counter() - started,
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
        "mean_beta", "ann_nonconverged_models", "elapsed_seconds",
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
    axes[3].plot(frame["time"], frame["high_point"], label="Nested-LOO MSE-ANN high forecast")
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
