"""Fixed residual-window EnbPI sensitivity analysis for sunspot data.

The expensive OOT cross-fitting and final hybrid fit are performed once.  The
same raw point forecasts and honest training residuals are then replayed with
each requested rolling residual-window length.  Consequently, differences
between results are caused only by the residual-window length.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np
import pandas as pd

from .kf_enbpi import EnbPIConfig, shortest_residual_offsets
from .kf_out_of_time_enbpi import (
    OutOfTimeEnbPIConfig,
    run_out_of_time_enbpi,
)


@dataclass
class SunspotFixedWindowResult:
    predictions: pd.DataFrame
    summary: pd.DataFrame
    activity_summary: pd.DataFrame
    crossfit_residuals: pd.DataFrame
    fit_seconds: float


def _interval_metrics(
    truth: np.ndarray,
    point: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    alpha: float,
) -> dict[str, float]:
    miss_low = truth < lower
    miss_high = truth > upper
    score = (
        upper - lower
        + (2.0 / alpha) * (lower - truth) * miss_low
        + (2.0 / alpha) * (truth - upper) * miss_high
    )
    return {
        "n": float(len(truth)),
        "rmse": float(np.sqrt(np.mean((truth - point) ** 2))),
        "mae": float(np.mean(np.abs(truth - point))),
        "coverage": float(np.mean((truth >= lower) & (truth <= upper))),
        "mean_width": float(np.mean(upper - lower)),
        "median_width": float(np.median(upper - lower)),
        "interval_score": float(np.mean(score)),
        "lower_miss_rate": float(np.mean(miss_low)),
        "upper_miss_rate": float(np.mean(miss_high)),
        "negative_lower_rate": float(np.mean(lower < 0.0)),
    }


def run_sunspot_fixed_windows(
    observed: np.ndarray,
    train_size: int,
    dates: pd.DatetimeIndex,
    *,
    windows: tuple[int, ...] = (24, 60, 120, 180, 300),
    alpha: float = 0.05,
    random_state: int = 2026,
) -> SunspotFixedWindowResult:
    """Run one raw-scale M6 fit and replay it using fixed residual windows."""
    if not windows or any(window < 2 for window in windows):
        raise ValueError("windows must contain integers >= 2")
    y = np.asarray(observed, dtype=float)
    if len(dates) != len(y) - train_size:
        raise ValueError("dates must correspond exactly to the test observations")

    started = perf_counter()
    base = EnbPIConfig(alpha=alpha, random_state=random_state)
    # One candidate disables adaptive choice while retaining the exact OOT
    # cross-fitting and hybrid fitting code used by M6.
    fitted = run_out_of_time_enbpi(
        y,
        train_size,
        config=OutOfTimeEnbPIConfig(
            base=base,
            residual_window_candidates=(max(windows),),
        ),
        model_name="Sunspot fixed-window EnbPI",
        data_seed=random_state,
    )
    fit_seconds = perf_counter() - started
    forecast = fitted.forecast
    raw_point = forecast.point - forecast.bias_correction
    calibration = fitted.crossfit_predictions["final_residual"].to_numpy(float)
    truth = y[train_size:]

    frames: list[pd.DataFrame] = []
    summary_rows: list[dict[str, float | int]] = []
    activity_rows: list[dict[str, float | int | str]] = []
    for window in windows:
        window_started = perf_counter()
        pool = np.array(calibration[-window:], copy=True)
        point = np.empty_like(truth)
        lower = np.empty_like(truth)
        upper = np.empty_like(truth)
        for j, (actual, raw) in enumerate(zip(truth, raw_point)):
            bias = float(np.mean(pool)) if base.oob_bias_correction else 0.0
            lo, hi, _ = shortest_residual_offsets(
                pool - bias, base.alpha, base.beta_grid_size
            )
            point[j] = raw + bias
            lower[j] = point[j] + lo
            upper[j] = point[j] + hi
            pool = np.r_[pool, actual - raw][-window:]
        replay_seconds = perf_counter() - window_started
        metrics = _interval_metrics(truth, point, lower, upper, alpha)
        summary_rows.append({
            "window": window,
            **metrics,
            "shared_fit_seconds": fit_seconds,
            "window_replay_seconds": replay_seconds,
        })
        frame = pd.DataFrame({
            "date": dates,
            "window": window,
            "truth": truth,
            "point": point,
            "lower": lower,
            "upper": upper,
        })
        frames.append(frame)

        levels = pd.cut(
            truth,
            bins=[-np.inf, 10.0, 100.0, np.inf],
            labels=["low (y<=10)", "middle (10<y<=100)", "high (y>100)"],
        )
        for level in levels.categories:
            mask = np.asarray(levels == level)
            activity_rows.append({
                "window": window,
                "activity": str(level),
                **_interval_metrics(
                    truth[mask], point[mask], lower[mask], upper[mask], alpha
                ),
            })

    return SunspotFixedWindowResult(
        predictions=pd.concat(frames, ignore_index=True),
        summary=pd.DataFrame(summary_rows),
        activity_summary=pd.DataFrame(activity_rows),
        crossfit_residuals=fitted.crossfit_predictions.copy(),
        fit_seconds=fit_seconds,
    )
