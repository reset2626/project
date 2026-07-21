"""Oracle conditional-mean benchmarks for the simulated M1+M3/M1+M9 cases.

The oracle is intentionally unavailable for real data such as 0050.  It uses
the true simulated components and the exact data-generating equations, but it
does not observe the new innovations at the forecast time.  Its test RMSE is
therefore an empirical lower-bound benchmark, not a deployable predictor.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from kf_enbpi import EnbPIResult, select_representative_run


Array = np.ndarray
OracleModelName = Literal["m1m3", "m1m9"]


@dataclass(frozen=True)
class OracleForecast:
    times: Array
    low: Array
    high: Array
    point: Array
    theoretical_observed_rmse: float
    theoretical_clean_rmse: float


def oracle_forecast(
    result: EnbPIResult,
    model_name: OracleModelName,
    *,
    low_error_std: float = 1.0,
    high_error_base_std: float = 1.0,
    high_error_low_sensitivity: float = 0.5,
    observation_noise_std: float = 0.15,
) -> OracleForecast:
    """Return one-step conditional means from the exact simulation equations."""
    if result.true_low is None or result.true_high is None:
        raise ValueError("Oracle forecasts require simulated true_low and true_high")
    if model_name not in ("m1m3", "m1m9"):
        raise ValueError("model_name must be 'm1m3' or 'm1m9'")

    times = np.asarray(result.test_times, dtype=int)
    true_low = np.asarray(result.true_low, dtype=float)
    true_high = np.asarray(result.true_high, dtype=float)
    if model_name == "m1m3" and np.any(times < 2):
        raise ValueError("M1M3 oracle requires two high-frequency lags")
    if np.any(times < 1):
        raise ValueError("Oracle forecasts require at least one lag")

    low = 0.6 * true_low[times - 1]
    h1 = true_high[times - 1]
    if model_name == "m1m9":
        gate = 1.0 / (1.0 + np.exp(np.clip(-10.0 * h1, -700.0, 700.0)))
        high = 0.8 * h1 - 0.8 * h1 * gate
    else:
        h2 = true_high[times - 2]
        exp_term = np.exp(-(h1**2))
        high = (0.5 + 0.9 * exp_term) * h1
        high += (-0.8 - 1.8 * exp_term) * h2

    high_std = high_error_base_std * (
        1.0 + high_error_low_sensitivity * np.abs(true_low[times - 1])
    )
    clean_variance = low_error_std**2 + high_std**2
    observed_variance = clean_variance + observation_noise_std**2
    return OracleForecast(
        times=times,
        low=np.asarray(low, dtype=float),
        high=np.asarray(high, dtype=float),
        point=np.asarray(low + high, dtype=float),
        theoretical_observed_rmse=float(np.sqrt(np.mean(observed_variance))),
        theoretical_clean_rmse=float(np.sqrt(np.mean(clean_variance))),
    )


def oracle_comparison_metrics(
    result: EnbPIResult, model_name: OracleModelName
) -> dict[str, float]:
    """Compare the deployable hybrid forecast with the unattainable oracle."""
    oracle = oracle_forecast(result, model_name)
    clean_truth = (
        np.asarray(result.true_low)[result.test_times]
        + np.asarray(result.true_high)[result.test_times]
    )
    hybrid_observed_error = np.asarray(result.truth) - np.asarray(result.point)
    oracle_observed_error = np.asarray(result.truth) - oracle.point
    hybrid_clean_error = clean_truth - np.asarray(result.point)
    oracle_clean_error = clean_truth - oracle.point
    hybrid_rmse = float(np.sqrt(np.mean(hybrid_observed_error**2)))
    oracle_rmse = float(np.sqrt(np.mean(oracle_observed_error**2)))
    return {
        "hybrid_observed_rmse": hybrid_rmse,
        "oracle_observed_rmse": oracle_rmse,
        "hybrid_minus_oracle_rmse": hybrid_rmse - oracle_rmse,
        "hybrid_to_oracle_rmse_ratio": hybrid_rmse / oracle_rmse,
        "hybrid_clean_rmse": float(np.sqrt(np.mean(hybrid_clean_error**2))),
        "oracle_clean_rmse": float(np.sqrt(np.mean(oracle_clean_error**2))),
        "theoretical_oracle_observed_rmse": oracle.theoretical_observed_rmse,
        "theoretical_oracle_clean_rmse": oracle.theoretical_clean_rmse,
    }


def oracle_comparison_table(
    result: EnbPIResult, model_name: OracleModelName
) -> pd.DataFrame:
    """One-row comparison table for a single simulated experiment."""
    return pd.DataFrame(
        [{"model": model_name.upper(), **oracle_comparison_metrics(result, model_name)}]
    )


def oracle_monte_carlo_summary(
    runs: pd.DataFrame,
    results: list[EnbPIResult],
    model_name: OracleModelName,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return run-level and mean/std Oracle comparison tables."""
    if len(runs) != len(results) or not results:
        raise ValueError("runs and results must be non-empty and aligned")
    rows: list[dict[str, float | int]] = []
    for position, result in enumerate(results):
        run_number = int(runs.iloc[position]["run"])
        rows.append(
            {
                "run": run_number,
                "data_seed": int(runs.iloc[position]["data_seed"]),
                **oracle_comparison_metrics(result, model_name),
            }
        )
    per_run = pd.DataFrame(rows)
    metric_columns = [
        column for column in per_run.columns if column not in ("run", "data_seed")
    ]
    summary = (
        per_run[metric_columns]
        .agg(["mean", "std"])
        .T.reset_index(names="metric")
    )
    return per_run, summary


def plot_oracle_comparison(result: EnbPIResult, model_name: OracleModelName):
    """Plot final and component hybrid forecasts against the true-equation oracle."""
    import matplotlib.pyplot as plt

    oracle = oracle_forecast(result, model_name)
    times = result.test_times
    fig, axes = plt.subplots(3, 1, figsize=(14, 11), sharex=True)

    axes[0].plot(times, result.truth, label="Observed mixed data", linewidth=1.5)
    axes[0].plot(times, result.point, label="KF-ARIMA+ANN point forecast", linewidth=1.7)
    axes[0].plot(times, oracle.point, label="Oracle conditional mean", linewidth=1.7)
    axes[0].fill_between(
        times, result.lower, result.upper, alpha=0.20, color="grey", label="EnbPI interval"
    )
    axes[0].set_ylabel("Final signal")
    axes[0].set_title(f"Final point forecast versus oracle: {model_name.upper()}")
    axes[0].legend(loc="best")
    axes[0].grid(alpha=0.25)

    axes[1].plot(times, np.asarray(result.true_low)[times], label="True low")
    axes[1].plot(times, result.low_point, label="KF-ARIMA low forecast")
    axes[1].plot(times, oracle.low, label="Oracle low conditional mean")
    axes[1].set_ylabel("Low")
    axes[1].legend(loc="best")
    axes[1].grid(alpha=0.25)

    axes[2].plot(times, np.asarray(result.true_high)[times], label="True high")
    axes[2].plot(times, result.high_point, label="KF-ANN high forecast")
    axes[2].plot(times, oracle.high, label="Oracle high conditional mean")
    axes[2].set_ylabel("High")
    axes[2].set_xlabel("Time")
    axes[2].legend(loc="best")
    axes[2].grid(alpha=0.25)
    fig.tight_layout()
    return fig, axes


def plot_representative_oracle_run(
    runs: pd.DataFrame,
    results: list[EnbPIResult],
    model_name: OracleModelName,
):
    """Use the same median-hybrid-RMSE representative rule as existing plots."""
    run_number, representative = select_representative_run(runs, results)
    fig, axes = plot_oracle_comparison(representative, model_name)
    fig.suptitle(
        f"Representative Monte Carlo run {run_number}: {model_name.upper()} oracle benchmark",
        y=1.01,
    )
    fig.tight_layout()
    return run_number, representative, fig, axes
