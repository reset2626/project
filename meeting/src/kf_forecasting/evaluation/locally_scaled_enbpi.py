"""Locally scaled extension of out-of-time EnbPI.

The extension is data-agnostic: a robust k-nearest-neighbour scale estimator
maps causal state features to expected absolute forecast error.  EnbPI is then
applied to residuals divided by that local scale.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

from .kf_enbpi import shortest_residual_offsets
from .kf_out_of_time_enbpi import (
    OutOfTimeEnbPIConfig,
    OutOfTimeEnbPIResult,
    select_residual_window,
)


Array = np.ndarray


@dataclass
class LocallyScaledResult:
    predictions: pd.DataFrame
    summary: pd.DataFrame
    window_selection: pd.DataFrame
    selected_window: int | None


def causal_state_features(y: Array, times: Array, points: Array) -> Array:
    """Generic level, slope and trailing-volatility features available at t."""
    y = np.asarray(y, float)
    rows = []
    for t, point in zip(np.asarray(times, int), np.asarray(points, float)):
        history = y[:t]
        d12 = np.diff(history[-13:])
        d36 = np.diff(history[-37:])
        rows.append([
            point,
            point - history[-1],
            float(np.std(d12, ddof=1)) if len(d12) > 1 else 0.0,
            float(np.std(d36, ddof=1)) if len(d36) > 1 else 0.0,
        ])
    return np.asarray(rows, float)


def _knn_scale(
    train_x: Array,
    abs_residual: Array,
    query_x: Array,
    *,
    neighbours: int,
    leave_self_out: bool = False,
) -> Array:
    scaler = RobustScaler().fit(train_x)
    x = scaler.transform(train_x)
    q = scaler.transform(query_x)
    distances = ((q[:, None, :] - x[None, :, :]) ** 2).sum(axis=2)
    if leave_self_out and len(q) == len(x):
        np.fill_diagonal(distances, np.inf)
    k = min(neighbours, len(train_x) - int(leave_self_out))
    indices = np.argpartition(distances, kth=k - 1, axis=1)[:, :k]
    # Median absolute residual is robust to the sharp solar-like spikes, while
    # the floor prevents nearly zero scales in quiet regimes.
    scale = np.median(np.asarray(abs_residual)[indices], axis=1)
    floor = max(float(np.quantile(abs_residual, 0.10)), 1e-6)
    return np.maximum(scale, floor)


def _metrics(truth: Array, point: Array, lower: Array, upper: Array, alpha: float):
    miss_l, miss_u = truth < lower, truth > upper
    score = upper - lower + 2 / alpha * (
        (lower - truth) * miss_l + (truth - upper) * miss_u
    )
    return {
        "coverage": np.mean((truth >= lower) & (truth <= upper)),
        "mean_width": np.mean(upper - lower),
        "median_width": np.median(upper - lower),
        "interval_score": np.mean(score),
        "negative_lower_rate": np.mean(lower < 0),
        "rmse": np.sqrt(np.mean((truth - point) ** 2)),
        "mae": np.mean(np.abs(truth - point)),
    }


def locally_scaled_from_fitted(
    fitted: OutOfTimeEnbPIResult,
    dates: pd.DatetimeIndex,
    *,
    neighbours: int = 60,
    support_lower: float | None = None,
) -> LocallyScaledResult:
    """Recalibrate one fitted OOT-EnbPI result with local residual scales."""
    forecast = fitted.forecast
    y = forecast.observed
    cross = fitted.crossfit_predictions
    train_times = cross["time"].to_numpy(int)
    train_points = cross["final_point"].to_numpy(float)
    train_residuals = cross["final_residual"].to_numpy(float)
    train_x = causal_state_features(y, train_times, train_points)
    calibration_scale = _knn_scale(
        train_x, np.abs(train_residuals), train_x,
        neighbours=neighbours, leave_self_out=True,
    )
    standardized = train_residuals / calibration_scale
    selected, selection = select_residual_window(
        standardized,
        OutOfTimeEnbPIConfig(
            base=forecast.config,
            residual_window_candidates=(24, 60, 120, 180, 300, None),
        ),
    )
    pool = standardized if selected is None else standardized[-selected:]
    max_size = len(pool)

    test_times = forecast.test_times
    raw_point = forecast.point - forecast.bias_correction
    test_x = causal_state_features(y, test_times, raw_point)
    truth = forecast.truth
    point, lower, upper, scales = (np.empty_like(truth) for _ in range(4))
    dynamic_x = np.array(train_x, copy=True)
    dynamic_abs = np.abs(train_residuals).copy()
    for j, (actual, raw, x_t) in enumerate(zip(truth, raw_point, test_x)):
        sigma = _knn_scale(
            dynamic_x, dynamic_abs, x_t[None, :], neighbours=neighbours
        )[0]
        bias_z = float(np.mean(pool)) if forecast.config.oob_bias_correction else 0.0
        lo, hi, _ = shortest_residual_offsets(
            pool - bias_z, forecast.config.alpha, forecast.config.beta_grid_size
        )
        point[j] = raw + sigma * bias_z
        lower[j] = point[j] + sigma * lo
        upper[j] = point[j] + sigma * hi
        scales[j] = sigma
        residual = actual - raw
        pool = np.r_[pool, residual / sigma][-max_size:]
        dynamic_x = np.vstack([dynamic_x, x_t])
        dynamic_abs = np.r_[dynamic_abs, abs(residual)]

    variants = [("Local-scale M6", lower.copy())]
    if support_lower is not None:
        variants.append(("Local-scale M6 + support", np.maximum(lower, support_lower)))
    frames, rows = [], []
    for method, method_lower in variants:
        frames.append(pd.DataFrame({
            "date": dates, "method": method, "truth": truth, "point": point,
            "lower": method_lower, "upper": upper, "local_scale": scales,
        }))
        rows.append({"method": method, "selected_window": selected, **_metrics(
            truth, point, method_lower, upper, forecast.config.alpha
        )})
    return LocallyScaledResult(
        pd.concat(frames, ignore_index=True), pd.DataFrame(rows), selection, selected
    )
