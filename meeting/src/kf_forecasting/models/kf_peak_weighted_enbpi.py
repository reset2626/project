"""Peak-weighted MSE ANN variant of the existing KF-EnbPI experiment.

The original :mod:`kf_enbpi` implementation is intentionally left unchanged.
Only the high-frequency ANN training loss differs: rare large-magnitude targets
receive a larger sample weight inside every bootstrap fit and rolling fold.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import TimeSeriesSplit

from kf_forecasting.models.kf_enbpi import (
    Array,
    EnbPIConfig,
    EnbPIResult,
    KalmanEnbPI,
    ModelName,
    causal_kalman_decomposition,
    make_lagged_features,
    shortest_residual_offsets,
    simulate_additive_data,
)


@dataclass(frozen=True)
class PeakWeightedEnbPIConfig(EnbPIConfig):
    """EnbPI configuration with a peak-weighted high-frequency ANN loss.

    ``ann_peak_quantile=0.90`` marks the largest 10% absolute deviations from
    the bootstrap target median as peaks. ``ann_peak_weight=3`` makes each such
    row contribute three times as much as a regular row to the MSE objective.
    """

    ann_peak_quantile: float = 0.90
    ann_peak_weight: float = 3.0


class PeakWeightedRegressor:
    """Wrap an sklearn ANN and derive peak weights from each training subset."""

    def __init__(
        self,
        base_model,
        *,
        peak_quantile: float,
        peak_weight: float,
    ) -> None:
        if not 0.0 < peak_quantile < 1.0:
            raise ValueError("peak_quantile must lie in (0, 1)")
        if peak_weight < 1.0:
            raise ValueError("peak_weight must be at least 1")
        self.base_model = base_model
        self.peak_quantile = float(peak_quantile)
        self.peak_weight = float(peak_weight)
        self.target_median_: float | None = None
        self.peak_threshold_: float | None = None
        self.sample_weight_: Array | None = None

    def fit(self, x: Array, y: Array) -> "PeakWeightedRegressor":
        targets = np.asarray(y, dtype=float)
        if targets.ndim != 1 or len(targets) < 2:
            raise ValueError("Peak-weighted ANN requires at least two targets")
        center = float(np.median(targets))
        deviation = np.abs(targets - center)
        threshold = float(np.quantile(deviation, self.peak_quantile))
        peak_mask = deviation > threshold
        weights = np.ones(len(targets), dtype=float)
        weights[peak_mask] = self.peak_weight

        # Both the Pipeline and the optional TransformedTargetRegressor forward
        # this parameter to the final MLPRegressor step under sklearn 1.7.
        self.base_model.fit(
            np.asarray(x, dtype=float),
            targets,
            mlpregressor__sample_weight=weights,
        )
        self.target_median_ = center
        self.peak_threshold_ = threshold
        self.sample_weight_ = weights
        return self

    def predict(self, x: Array) -> Array:
        return np.asarray(self.base_model.predict(x), dtype=float)


class PeakWeightedKalmanEnbPI(KalmanEnbPI):
    """Reuse the complete original EnbPI pipeline with a weighted ANN loss."""

    config: PeakWeightedEnbPIConfig

    def _new_ann(self, seed: int, *, max_iter: int) -> PeakWeightedRegressor:
        base_model = super()._new_ann(seed, max_iter=max_iter)
        return PeakWeightedRegressor(
            base_model,
            peak_quantile=self.config.ann_peak_quantile,
            peak_weight=self.config.ann_peak_weight,
        )

    def _select_ann_max_iter(
        self,
        x_train: Array,
        high_targets: Array,
        bootstrap_rows: Array,
        *,
        seed: int,
    ) -> tuple[int, float]:
        """Use chronological peak-weighted validation for iteration selection."""
        candidates = self._ann_iteration_grid()
        c = self.config
        if not c.ann_rolling_validation or len(candidates) == 1:
            return candidates[-1], float("nan")
        if c.ann_rolling_splits < 2:
            raise ValueError("ann_rolling_splits must be at least 2")
        if not 0.0 < c.ann_validation_fraction < 0.5:
            raise ValueError("ann_validation_fraction must lie in (0, 0.5)")

        sampled_times = np.unique(np.asarray(bootstrap_rows, dtype=int))
        test_size = max(
            1, int(round(len(sampled_times) * c.ann_validation_fraction))
        )
        max_splits = (len(sampled_times) - 2) // test_size
        n_splits = min(c.ann_rolling_splits, max_splits)
        if n_splits < 2:
            return candidates[-1], float("nan")

        splitter = TimeSeriesSplit(n_splits=n_splits, test_size=test_size)
        candidate_scores: list[float] = []
        for candidate in candidates:
            fold_scores: list[float] = []
            for fold, (train_pos, val_pos) in enumerate(
                splitter.split(sampled_times)
            ):
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
                    warnings.simplefilter("ignore", ConvergenceWarning)
                    model.fit(x_train[fold_rows], high_targets[fold_rows])
                prediction = model.predict(x_train[val_times])
                deviation = np.abs(
                    high_targets[val_times] - float(model.target_median_)
                )
                validation_weights = np.ones(len(val_times), dtype=float)
                validation_weights[
                    deviation > float(model.peak_threshold_)
                ] = c.ann_peak_weight
                squared_error = (high_targets[val_times] - prediction) ** 2
                fold_scores.append(
                    float(np.average(squared_error, weights=validation_weights))
                )
            candidate_scores.append(
                float(np.mean(fold_scores)) if fold_scores else float("inf")
            )

        best = int(np.argmin(candidate_scores))
        if not np.isfinite(candidate_scores[best]):
            return candidates[-1], float("nan")
        return candidates[best], candidate_scores[best]


def run_peak_weighted_kf_enbpi(
    observed: Array,
    train_size: int,
    *,
    config: PeakWeightedEnbPIConfig,
    model_name: str = "custom_peak_weighted",
    true_low: Array | None = None,
    true_high: Array | None = None,
    data_seed: int | None = None,
) -> EnbPIResult:
    """Run the original causal KF-EnbPI flow with peak-weighted ANN fits."""
    started = perf_counter()
    y = np.asarray(observed, dtype=float)
    if not config.window_size < train_size < len(y):
        raise ValueError("Need window_size < train_size < len(observed)")

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

    enbpi = PeakWeightedKalmanEnbPI(config).fit(
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
            center, low_center, high_center, _, _ = enbpi.predict_nested_loo(
                pseudo_low, x_t
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
            pseudo_history.append(point[j])

        batch_truth = y_test[batch_start:batch_stop]
        pending_residuals = list(batch_truth - raw_point[batch_start:batch_stop])
        k = len(pending_residuals)
        residual_pool = np.r_[residual_pool[k:], pending_residuals][-pool_size:]
        component_times = test_times[batch_start:batch_stop]
        pending_low_residuals = (
            estimated_low[component_times] - raw_low_point[batch_start:batch_stop]
        )
        pending_high_residuals = (
            estimated_high[component_times] - raw_high_point[batch_start:batch_stop]
        )
        low_residual_pool = np.r_[low_residual_pool[k:], pending_low_residuals][
            -pool_size:
        ]
        high_residual_pool = np.r_[high_residual_pool[k:], pending_high_residuals][
            -pool_size:
        ]
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
    )


def simulate_and_run_peak_weighted(
    model: ModelName,
    *,
    train_size: int = 650,
    horizon: int = 50,
    config: PeakWeightedEnbPIConfig | None = None,
    data_seed: int = 2026,
) -> EnbPIResult:
    """Simulate M1M3/M1M9 data and run the peak-weighted experiment."""
    config = config or PeakWeightedEnbPIConfig()
    rng = np.random.default_rng(data_seed)
    low, high, observed = simulate_additive_data(
        model, train_size + horizon, rng=rng
    )
    return run_peak_weighted_kf_enbpi(
        observed,
        train_size,
        config=config,
        model_name=f"{model}_peak_weighted",
        true_low=low,
        true_high=high,
        data_seed=data_seed,
    )


def high_peak_metrics(
    result: EnbPIResult,
    *,
    peak_quantile: float = 0.90,
) -> dict[str, float]:
    """Evaluate ANN accuracy separately on rare high-component magnitudes."""
    if not 0.0 < peak_quantile < 1.0:
        raise ValueError("peak_quantile must lie in (0, 1)")

    def _metrics(reference: Array, prefix: str) -> dict[str, float]:
        truth = np.asarray(reference, dtype=float)
        prediction = np.asarray(result.high_point, dtype=float)
        center = float(np.median(truth))
        deviation = np.abs(truth - center)
        threshold = float(np.quantile(deviation, peak_quantile))
        peak = deviation >= threshold
        regular = ~peak
        error = truth - prediction
        predicted_deviation = np.abs(prediction - center)
        denominator = float(np.mean(deviation[peak]))
        sign_truth = np.sign(truth[peak] - center)
        sign_prediction = np.sign(prediction[peak] - center)
        return {
            f"{prefix}_peak_threshold": threshold,
            f"{prefix}_tail_rmse": float(np.sqrt(np.mean(error[peak] ** 2))),
            f"{prefix}_regular_rmse": float(
                np.sqrt(np.mean(error[regular] ** 2))
            ),
            f"{prefix}_peak_recall": float(
                np.mean(predicted_deviation[peak] >= threshold)
            ),
            f"{prefix}_peak_sign_accuracy": float(
                np.mean(sign_truth == sign_prediction)
            ),
            f"{prefix}_peak_amplitude_ratio": (
                float(np.mean(predicted_deviation[peak]) / denominator)
                if denominator > 0.0
                else float("nan")
            ),
        }

    metrics = _metrics(
        np.asarray(result.estimated_high)[result.test_times], "kf_high"
    )
    if result.true_high is not None:
        metrics.update(
            _metrics(np.asarray(result.true_high)[result.test_times], "true_high")
        )
    return metrics


def peak_weighted_monte_carlo_summary(
    model: ModelName,
    *,
    n_runs: int = 20,
    train_size: int = 650,
    horizon: int = 50,
    config: PeakWeightedEnbPIConfig | None = None,
    seed: int = 2026,
) -> tuple[pd.DataFrame, pd.DataFrame, list[EnbPIResult]]:
    """Monte Carlo summary including ordinary and peak-specific metrics."""
    config = config or PeakWeightedEnbPIConfig()
    if n_runs < 1:
        raise ValueError("n_runs must be positive")
    seed_sequence = np.random.SeedSequence(seed)
    data_seeds = [int(child.generate_state(1)[0]) for child in seed_sequence.spawn(n_runs)]
    results: list[EnbPIResult] = []
    rows: list[dict[str, float | int]] = []
    for run, data_seed in enumerate(data_seeds, start=1):
        result = simulate_and_run_peak_weighted(
            model,
            train_size=train_size,
            horizon=horizon,
            config=config,
            data_seed=data_seed,
        )
        results.append(result)
        rows.append(
            {
                "run": run,
                "data_seed": data_seed,
                **result.metrics(),
                **high_peak_metrics(
                    result, peak_quantile=config.ann_peak_quantile
                ),
            }
        )
    runs = pd.DataFrame(rows)
    metric_columns = [
        column for column in runs.columns if column not in ("run", "data_seed")
    ]
    summary = (
        runs[metric_columns]
        .agg(["mean", "std"])
        .T.reset_index(names="metric")
    )
    return runs, summary, results


def paired_peak_comparison(
    baseline_runs: pd.DataFrame,
    baseline_results: list[EnbPIResult],
    weighted_runs: pd.DataFrame,
    weighted_results: list[EnbPIResult],
    *,
    peak_quantile: float = 0.90,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare baseline and weighted models on identical Monte Carlo seeds."""
    if not (
        len(baseline_runs)
        == len(weighted_runs)
        == len(baseline_results)
        == len(weighted_results)
    ):
        raise ValueError("Paired inputs must have equal lengths")
    rows: list[dict[str, float | int]] = []
    for position in range(len(baseline_results)):
        baseline = baseline_results[position]
        weighted = weighted_results[position]
        baseline_seed = int(baseline_runs.iloc[position]["data_seed"])
        weighted_seed = int(weighted_runs.iloc[position]["data_seed"])
        if baseline_seed != weighted_seed:
            raise ValueError("Baseline and weighted data seeds are not aligned")
        baseline_peak = high_peak_metrics(
            baseline, peak_quantile=peak_quantile
        )
        weighted_peak = high_peak_metrics(
            weighted, peak_quantile=peak_quantile
        )
        baseline_metrics = baseline.metrics()
        weighted_metrics = weighted.metrics()
        rows.append(
            {
                "run": position + 1,
                "data_seed": baseline_seed,
                "baseline_rmse": baseline_metrics["rmse"],
                "weighted_rmse": weighted_metrics["rmse"],
                "rmse_gain_baseline_minus_weighted": (
                    baseline_metrics["rmse"] - weighted_metrics["rmse"]
                ),
                "baseline_kf_high_tail_rmse": baseline_peak[
                    "kf_high_tail_rmse"
                ],
                "weighted_kf_high_tail_rmse": weighted_peak[
                    "kf_high_tail_rmse"
                ],
                "tail_rmse_gain_baseline_minus_weighted": (
                    baseline_peak["kf_high_tail_rmse"]
                    - weighted_peak["kf_high_tail_rmse"]
                ),
                "baseline_coverage": baseline_metrics["coverage"],
                "weighted_coverage": weighted_metrics["coverage"],
                "baseline_width": baseline_metrics["mean_width"],
                "weighted_width": weighted_metrics["mean_width"],
                "baseline_peak_recall": baseline_peak["kf_high_peak_recall"],
                "weighted_peak_recall": weighted_peak["kf_high_peak_recall"],
                "baseline_peak_amplitude_ratio": baseline_peak[
                    "kf_high_peak_amplitude_ratio"
                ],
                "weighted_peak_amplitude_ratio": weighted_peak[
                    "kf_high_peak_amplitude_ratio"
                ],
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
