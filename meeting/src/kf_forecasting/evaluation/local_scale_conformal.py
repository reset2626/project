"""Local-scale conformal calibration that preserves a method's raw interval."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler


Array = np.ndarray


@dataclass
class LocalScaleConformalResult:
    predictions: pd.DataFrame
    calibration: pd.DataFrame


def causal_state_features(y: Array, times: Array, points: Array) -> Array:
    """Level, forecast change, and short/medium trailing difference scales."""
    y = np.asarray(y, dtype=float)
    rows = []
    for t, point in zip(np.asarray(times, int), np.asarray(points, float)):
        history = y[:t]
        d12, d36 = np.diff(history[-13:]), np.diff(history[-37:])
        rows.append([
            point,
            point - history[-1],
            np.std(d12, ddof=1) if len(d12) > 1 else 0.0,
            np.std(d36, ddof=1) if len(d36) > 1 else 0.0,
        ])
    return np.asarray(rows, dtype=float)


def _local_scale(train_x: Array, abs_error: Array, query_x: Array, k: int) -> Array:
    scaler = RobustScaler().fit(train_x)
    x, query = scaler.transform(train_x), scaler.transform(query_x)
    distances = ((query[:, None, :] - x[None, :, :]) ** 2).sum(axis=2)
    neighbours = min(k, len(train_x))
    indices = np.argpartition(distances, neighbours - 1, axis=1)[:, :neighbours]
    scale = np.median(np.asarray(abs_error)[indices], axis=1)
    floor = max(float(np.quantile(abs_error, 0.10)), 1e-6)
    return np.maximum(scale, floor)


def _higher_quantile(values: Array, alpha: float) -> float:
    """Finite-sample split-conformal quantile using the higher order statistic."""
    values = np.asarray(values, dtype=float)
    probability = min(1.0, np.ceil((len(values) + 1) * (1 - alpha)) / len(values))
    return float(np.quantile(values, probability, method="higher"))


def calibrate_original_intervals(
    observed: Array,
    calibration: pd.DataFrame,
    testing: pd.DataFrame,
    *,
    alpha: float = 0.05,
    neighbours: int = 60,
    sequential_update: bool = True,
) -> LocalScaleConformalResult:
    """Calibrate raw [lower, upper] while retaining their shape and asymmetry.

    Required columns in both frames are time, truth, point, lower and upper.
    Calibration rows must precede testing rows chronologically.
    """
    required = {"time", "truth", "point", "lower", "upper"}
    if not required.issubset(calibration) or not required.issubset(testing):
        raise ValueError(f"Both frames require columns: {sorted(required)}")
    if calibration["time"].max() >= testing["time"].min():
        raise ValueError("Calibration observations must precede testing observations")
    y = np.asarray(observed, dtype=float)
    cal = calibration.sort_values("time").reset_index(drop=True).copy()
    test = testing.sort_values("time").reset_index(drop=True).copy()
    cal_x = causal_state_features(y, cal.time, cal.point)
    cal_error = cal.truth.to_numpy() - cal.point.to_numpy()
    # All holdout labels are available before formal testing begins.  Leave-one-
    # out scales prevent an observation's own error from determining its scale.
    cal_scales = np.empty(len(cal))
    for i in range(len(cal)):
        keep = np.arange(len(cal)) != i
        cal_scales[i] = _local_scale(
            cal_x[keep], np.abs(cal_error[keep]), cal_x[i : i + 1], neighbours
        )[0]
    cal_score = np.maximum(
        cal.lower.to_numpy() - cal.truth.to_numpy(),
        cal.truth.to_numpy() - cal.upper.to_numpy(),
    )
    standardized_scores = cal_score / cal_scales
    score_pool = standardized_scores.copy()
    max_pool = len(score_pool)
    dynamic_x, dynamic_abs = cal_x.copy(), np.abs(cal_error).copy()

    out_rows = []
    for row in test.itertuples(index=False):
        x_t = causal_state_features(y, [int(row.time)], [float(row.point)])
        sigma = _local_scale(dynamic_x, dynamic_abs, x_t, neighbours)[0]
        q = _higher_quantile(score_pool, alpha)
        lower = float(row.lower - sigma * q)
        upper = float(row.upper + sigma * q)
        # A negative score quantile may shrink an over-conservative interval.
        # Do not allow the two endpoints to cross.
        if lower > upper:
            midpoint = 0.5 * (float(row.lower) + float(row.upper))
            lower = upper = midpoint
        out_rows.append({
            "time": int(row.time), "truth": float(row.truth),
            "point": float(row.point), "raw_lower": float(row.lower),
            "raw_upper": float(row.upper), "lower": lower, "upper": upper,
            "local_scale": sigma, "score_quantile": q,
        })
        if sequential_update:
            error = float(row.truth - row.point)
            score = max(row.lower - row.truth, row.truth - row.upper)
            dynamic_x = np.vstack([dynamic_x, x_t])
            dynamic_abs = np.r_[dynamic_abs, abs(error)]
            score_pool = np.r_[score_pool, score / sigma][-max_pool:]

    cal["local_scale"] = cal_scales
    cal["conformity_score"] = cal_score
    cal["standardized_score"] = standardized_scores
    return LocalScaleConformalResult(pd.DataFrame(out_rows), cal)
