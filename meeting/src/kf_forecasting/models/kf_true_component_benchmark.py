"""Paired ablation that replaces KF components with simulated true components.

This benchmark keeps ARIMA, ANN, moving-block bootstrap, nested OOB aggregation,
bias correction, and EnbPI calibration unchanged.  It changes only the source
of the component histories.  True components through t-1 are revealed to the
benchmark predictor, so this is a simulation-only ablation rather than a
deployable forecasting method.
"""

from __future__ import annotations

from time import perf_counter

import numpy as np
import pandas as pd

from kf_forecasting.models.kf_enbpi import (
    Array,
    EnbPIConfig,
    EnbPIResult,
    KalmanEnbPI,
    ModelName,
    make_lagged_features,
    run_kf_enbpi,
    shortest_residual_offsets,
    simulate_additive_data,
)
from kf_forecasting.models.kf_oracle_benchmark import oracle_comparison_metrics


def _biases_and_offsets(
    residual_pool: Array,
    low_residual_pool: Array,
    high_residual_pool: Array,
    config: EnbPIConfig,
) -> tuple[float, float, float, tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]:
    """Use exactly the same bias/quantile rules as the main KF-EnbPI runner."""
    mode = config.oob_bias_correction_mode
    if mode not in ("component", "combined"):
        raise ValueError("oob_bias_correction_mode must be 'component' or 'combined'")
    if not config.oob_bias_correction:
        low_bias = high_bias = final_bias = 0.0
    elif mode == "component":
        low_bias = float(np.mean(low_residual_pool))
        high_bias = float(np.mean(high_residual_pool))
        final_bias = low_bias + high_bias
    else:
        low_bias = high_bias = 0.0
        final_bias = float(np.mean(residual_pool))

    final_offsets = shortest_residual_offsets(
        residual_pool - final_bias, config.alpha, config.beta_grid_size
    )
    low_offsets = shortest_residual_offsets(
        low_residual_pool - low_bias, config.alpha, config.beta_grid_size
    )
    high_offsets = shortest_residual_offsets(
        high_residual_pool - high_bias, config.alpha, config.beta_grid_size
    )
    return low_bias, high_bias, final_bias, final_offsets, low_offsets, high_offsets


def run_true_component_enbpi(
    observed: Array,
    true_low: Array,
    true_high: Array,
    train_size: int,
    *,
    config: EnbPIConfig,
    model_name: str = "true_components",
    data_seed: int | None = None,
) -> EnbPIResult:
    """Run the same hybrid EnbPI algorithm using true component histories.

    At forecast time t the predictor receives true low/high only through t-1.
    ``batch_size=1`` is required because latent components are revealed only
    after each one-step forecast in this simulation-only ablation.
    """
    started = perf_counter()
    y = np.asarray(observed, dtype=float)
    low = np.asarray(true_low, dtype=float)
    high = np.asarray(true_high, dtype=float)
    if not (y.ndim == low.ndim == high.ndim == 1):
        raise ValueError("observed, true_low, and true_high must be one-dimensional")
    if not (len(y) == len(low) == len(high)):
        raise ValueError("observed, true_low, and true_high must have equal lengths")
    if not config.window_size < train_size < len(y):
        raise ValueError("Need window_size < train_size < len(observed)")
    if config.batch_size != 1:
        raise ValueError("True-component benchmark currently requires batch_size=1")

    x_train, y_train, target_times = make_lagged_features(
        y[:train_size], low[:train_size], high[:train_size], config.window_size
    )
    engine = KalmanEnbPI(config).fit(
        x_train,
        y_train,
        low[:train_size],
        high[:train_size],
        target_times,
    )

    residual_pool = np.array(engine.residual_pool, copy=True)
    low_residual_pool = np.array(engine.low_residual_pool, copy=True)
    high_residual_pool = np.array(engine.high_residual_pool, copy=True)
    pool_size = len(residual_pool)
    test_times = np.arange(train_size, len(y), dtype=int)
    n_test = len(test_times)

    point = np.empty(n_test)
    lower = np.empty(n_test)
    upper = np.empty(n_test)
    low_point = np.empty(n_test)
    low_lower = np.empty(n_test)
    low_upper = np.empty(n_test)
    high_point = np.empty(n_test)
    high_lower = np.empty(n_test)
    high_upper = np.empty(n_test)
    betas = np.empty(n_test)
    low_betas = np.empty(n_test)
    high_betas = np.empty(n_test)
    bias_corrections = np.empty(n_test)
    low_bias_corrections = np.empty(n_test)
    high_bias_corrections = np.empty(n_test)
    raw_point = np.empty(n_test)
    raw_low_point = np.empty(n_test)
    raw_high_point = np.empty(n_test)

    for j, t in enumerate(test_times):
        (
            low_bias,
            high_bias,
            final_bias,
            (lo_offset, hi_offset, beta),
            (low_lo_offset, low_hi_offset, low_beta),
            (high_lo_offset, high_hi_offset, high_beta),
        ) = _biases_and_offsets(
            residual_pool, low_residual_pool, high_residual_pool, config
        )

        # Strictly causal benchmark features: true latent histories stop at t-1.
        x_t = np.r_[
            low[t - config.window_size : t],
            high[t - config.window_size : t],
            y[t - config.window_size : t],
        ]
        center, low_center, high_center, _, _ = engine.predict_nested_loo(
            low[:t], x_t
        )
        raw_point[j] = center
        raw_low_point[j] = low_center
        raw_high_point[j] = high_center
        low_point[j] = low_center + low_bias
        high_point[j] = high_center + high_bias
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

        # Reveal y_t, L_t, H_t only after the forecast at t is complete.
        residual_pool = np.r_[residual_pool[1:], y[t] - raw_point[j]][-pool_size:]
        low_residual_pool = np.r_[
            low_residual_pool[1:], low[t] - raw_low_point[j]
        ][-pool_size:]
        high_residual_pool = np.r_[
            high_residual_pool[1:], high[t] - raw_high_point[j]
        ][-pool_size:]

    return EnbPIResult(
        model_name=model_name,
        config=config,
        train_size=train_size,
        test_times=test_times,
        truth=y[test_times],
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
        initial_oob_residuals=np.array(engine.residual_pool, copy=True),
        initial_low_oob_residuals=np.array(engine.low_residual_pool, copy=True),
        initial_high_oob_residuals=np.array(engine.high_residual_pool, copy=True),
        oob_counts=np.array(engine.oob_counts, copy=True),
        # EnbPIResult calls these reference arrays "estimated"; in this ablation
        # they intentionally contain the true components used by the predictor.
        estimated_low=np.array(low, copy=True),
        estimated_high=np.array(high, copy=True),
        observed=np.array(y, copy=True),
        true_low=np.array(low, copy=True),
        true_high=np.array(high, copy=True),
        data_seed=data_seed,
        selected_arima_order=engine.selected_arima_order,
        selected_ann_max_iters=np.asarray(engine.selected_ann_max_iters, dtype=int),
        ann_rolling_validation_mse=np.asarray(
            engine.ann_rolling_validation_scores, dtype=float
        ),
        ann_nonconverged_models=engine.ann_nonconverged_models,
        elapsed_seconds=perf_counter() - started,
    )


def run_paired_true_component_experiment(
    model: ModelName,
    *,
    train_size: int = 650,
    horizon: int = 50,
    config: EnbPIConfig | None = None,
    data_seed: int = 2026,
) -> tuple[EnbPIResult, EnbPIResult]:
    """Run current-KF and true-component hybrids on exactly the same data."""
    config = config or EnbPIConfig()
    rng = np.random.default_rng(data_seed)
    low, high, observed = simulate_additive_data(
        model, train_size + horizon, rng=rng
    )
    current = run_kf_enbpi(
        observed,
        train_size,
        config=config,
        model_name=f"{model}_kf",
        true_low=low,
        true_high=high,
        data_seed=data_seed,
    )
    true_component = run_true_component_enbpi(
        observed,
        low,
        high,
        train_size,
        config=config,
        model_name=f"{model}_true_components",
        data_seed=data_seed,
    )
    return current, true_component


def paired_comparison_metrics(
    current: EnbPIResult, true_component: EnbPIResult
) -> dict[str, float]:
    """Return current/true-component/oracle metrics on the same realization."""
    if not np.array_equal(current.test_times, true_component.test_times):
        raise ValueError("Paired results must use the same test times")
    if not np.allclose(current.truth, true_component.truth):
        raise ValueError("Paired results must use the same test observations")
    current_metrics = current.metrics()
    truth_metrics = true_component.metrics()
    current_rmse = current_metrics["rmse"]
    truth_rmse = truth_metrics["rmse"]
    model_name = current.model_name.split("_", 1)[0].lower()
    if model_name not in ("m1m3", "m1m9"):
        raise ValueError("Cannot infer M1M3/M1M9 model name from current result")
    oracle_rmse = oracle_comparison_metrics(current, model_name)[
        "oracle_observed_rmse"
    ]
    return {
        "kf_final_rmse": current_rmse,
        "true_component_final_rmse": truth_rmse,
        "equation_oracle_rmse": oracle_rmse,
        "rmse_gain_kf_minus_true_component": current_rmse - truth_rmse,
        "kf_to_true_component_rmse_ratio": current_rmse / truth_rmse,
        "kf_minus_equation_oracle_rmse": current_rmse - oracle_rmse,
        "true_component_minus_equation_oracle_rmse": truth_rmse - oracle_rmse,
        "kf_final_coverage": current_metrics["coverage"],
        "true_component_final_coverage": truth_metrics["coverage"],
        "kf_final_mean_width": current_metrics["mean_width"],
        "true_component_final_mean_width": truth_metrics["mean_width"],
        "kf_low_true_rmse": current_metrics["low_true_rmse"],
        "true_component_low_rmse": truth_metrics["low_true_rmse"],
        "kf_high_true_rmse": current_metrics["high_true_rmse"],
        "true_component_high_rmse": truth_metrics["high_true_rmse"],
    }


def paired_comparison_table(
    current: EnbPIResult, true_component: EnbPIResult
) -> pd.DataFrame:
    """One-row paired comparison for a reproducible single run."""
    return pd.DataFrame(
        [
            {
                "model": current.model_name.rsplit("_kf", 1)[0].upper(),
                "data_seed": current.data_seed,
                **paired_comparison_metrics(current, true_component),
            }
        ]
    )


def paired_true_component_monte_carlo(
    model: ModelName,
    *,
    n_runs: int = 20,
    train_size: int = 650,
    horizon: int = 50,
    config: EnbPIConfig | None = None,
    seed: int = 2026,
) -> tuple[pd.DataFrame, pd.DataFrame, list[EnbPIResult], list[EnbPIResult]]:
    """Paired Monte Carlo with identical DGP and ensemble settings per run."""
    config = config or EnbPIConfig()
    seed_sequence = np.random.SeedSequence(seed)
    rows: list[dict[str, float | int]] = []
    current_results: list[EnbPIResult] = []
    true_component_results: list[EnbPIResult] = []
    for run, child in enumerate(seed_sequence.spawn(n_runs), start=1):
        data_seed = int(child.generate_state(1)[0])
        current, true_component = run_paired_true_component_experiment(
            model,
            train_size=train_size,
            horizon=horizon,
            config=config,
            data_seed=data_seed,
        )
        current_results.append(current)
        true_component_results.append(true_component)
        rows.append(
            {
                "run": run,
                "data_seed": data_seed,
                **paired_comparison_metrics(current, true_component),
            }
        )
    runs = pd.DataFrame(rows)
    metric_columns = [
        column for column in runs.columns if column not in ("run", "data_seed")
    ]
    summary = runs[metric_columns].agg(["mean", "std"]).T.reset_index(names="metric")
    return runs, summary, current_results, true_component_results


def select_representative_pair(
    runs: pd.DataFrame,
    current_results: list[EnbPIResult],
    true_component_results: list[EnbPIResult],
) -> tuple[int, EnbPIResult, EnbPIResult]:
    """Select the pair whose current-KF RMSE is nearest its median."""
    if not (len(runs) == len(current_results) == len(true_component_results)):
        raise ValueError("runs and result lists must be aligned")
    position = int(
        np.argmin(
            np.abs(
                runs["kf_final_rmse"].to_numpy()
                - runs["kf_final_rmse"].median()
            )
        )
    )
    return (
        int(runs.iloc[position]["run"]),
        current_results[position],
        true_component_results[position],
    )


def plot_paired_comparison(
    current: EnbPIResult,
    true_component: EnbPIResult,
):
    """Plot final and component effects of replacing KF with true histories."""
    import matplotlib.pyplot as plt

    t = current.test_times
    fig, axes = plt.subplots(3, 1, figsize=(14, 11), sharex=True)
    axes[0].plot(t, current.truth, label="Observed mixed data", linewidth=1.4)
    axes[0].plot(t, current.point, label="Current KF hybrid", linewidth=1.7)
    axes[0].plot(
        t, true_component.point, label="True-component hybrid", linewidth=1.7
    )
    axes[0].fill_between(
        t,
        current.lower,
        current.upper,
        color="grey",
        alpha=0.18,
        label="Current KF EnbPI interval",
    )
    axes[0].set_ylabel("Final signal")
    axes[0].set_title("Same ARIMA+ANN+EnbPI algorithm; component source ablation")

    axes[1].plot(t, np.asarray(current.true_low)[t], label="True low")
    axes[1].plot(t, current.estimated_low[t], label="KF estimated low")
    axes[1].plot(t, current.low_point, label="Current KF-ARIMA forecast")
    axes[1].plot(t, true_component.low_point, label="True-component ARIMA forecast")
    axes[1].set_ylabel("Low")

    axes[2].plot(t, np.asarray(current.true_high)[t], label="True high")
    axes[2].plot(t, current.estimated_high[t], label="KF estimated high")
    axes[2].plot(t, current.high_point, label="Current KF-ANN forecast")
    axes[2].plot(t, true_component.high_point, label="True-component ANN forecast")
    axes[2].set_ylabel("High")
    axes[2].set_xlabel("Time")

    for axis in axes:
        axis.legend(loc="best")
        axis.grid(alpha=0.25)
    fig.tight_layout()
    return fig, axes


def plot_representative_pair(
    runs: pd.DataFrame,
    current_results: list[EnbPIResult],
    true_component_results: list[EnbPIResult],
    *,
    model_name: str,
):
    run, current, true_component = select_representative_pair(
        runs, current_results, true_component_results
    )
    fig, axes = plot_paired_comparison(current, true_component)
    fig.suptitle(
        f"Representative paired Monte Carlo run {run}: {model_name.upper()}",
        y=1.01,
    )
    fig.tight_layout()
    return run, current, true_component, fig, axes
