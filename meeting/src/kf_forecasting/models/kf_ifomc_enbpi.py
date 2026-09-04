"""Rolling IFOMC residual intervals for an already fitted M6 predictor.

The ARIMA/ANN point predictor is not refitted.  Ordered residual states are
defined from training-only out-of-time residuals.  During testing, the current
transition row is converted to a mixture over destination-state residuals;
the requested tail quantiles are added to the unchanged M6 point forecast.
Only after the response is observed is the transition matrix updated.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .kf_out_of_time_enbpi import OutOfTimeEnbPIResult, _local_scale_metrics

Array = np.ndarray


@dataclass(frozen=True)
class IFOMCIntervalConfig:
    """Settings for the rolling first-order Markov residual interval."""

    n_states: int = 10
    alpha: float | None = None
    laplace: float = 0.0
    min_source_transitions: int = 10
    support_lower: float | None = None


@dataclass
class IFOMCIntervalResult:
    predictions: pd.DataFrame
    summary: pd.DataFrame
    state_table: pd.DataFrame
    initial_transition_matrix: pd.DataFrame
    final_transition_matrix: pd.DataFrame


def _bias_adjusted_training_residuals(
    raw_residuals: Array, window: int | None, use_bias: bool
) -> Array:
    """Recreate causal point errors after the M6 rolling mean-bias correction."""
    raw = np.asarray(raw_residuals, dtype=float)
    adjusted = np.empty_like(raw)
    for i, value in enumerate(raw):
        start = 0 if window is None else max(0, i - window)
        bias = float(np.mean(raw[start:i])) if use_bias and i > 0 else 0.0
        adjusted[i] = value - bias
    return adjusted


def _state_edges(values: Array, requested_states: int) -> Array:
    if requested_states < 2:
        raise ValueError("n_states must be at least 2")
    values = np.asarray(values, dtype=float)
    if len(values) < requested_states * 2:
        raise ValueError("Too few OOT residuals for the requested number of states")
    cuts = np.quantile(values, np.arange(1, requested_states) / requested_states)
    return np.r_[-np.inf, np.unique(cuts), np.inf]


def _assign_state(values: Array, edges: Array) -> Array:
    return np.searchsorted(edges[1:-1], np.asarray(values, dtype=float), side="right")


def _transition_counts(states: Array, n_states: int) -> Array:
    counts = np.zeros((n_states, n_states), dtype=float)
    for source, destination in zip(states[:-1], states[1:]):
        counts[int(source), int(destination)] += 1.0
    return counts


def _transition_probabilities(counts: Array, laplace: float) -> Array:
    if laplace < 0:
        raise ValueError("laplace must be non-negative")
    smoothed = np.asarray(counts, dtype=float) + float(laplace)
    totals = smoothed.sum(axis=1, keepdims=True)
    return np.divide(smoothed, totals, out=np.zeros_like(smoothed), where=totals > 0)


def _weighted_quantile(values: Array, weights: Array, probabilities: Array) -> Array:
    order = np.argsort(values)
    x = np.asarray(values, dtype=float)[order]
    w = np.asarray(weights, dtype=float)[order]
    keep = w > 0
    x, w = x[keep], w[keep]
    if not len(x) or not np.isfinite(w.sum()) or w.sum() <= 0:
        raise ValueError("Cannot calculate a weighted quantile without positive weights")
    cumulative = np.cumsum(w) / np.sum(w)
    return np.interp(np.asarray(probabilities, dtype=float), cumulative, x)


def _conditional_offsets(
    residuals: Array,
    states: Array,
    counts: Array,
    source_state: int,
    *,
    alpha: float,
    laplace: float,
    min_source_transitions: int,
) -> tuple[float, float, Array, bool]:
    """Return residual quantiles from the current transition row.

    Each destination-state probability is spread uniformly over the observed
    residual values in that state.  If the source row is too sparse, the
    training/rolling marginal residual distribution is used as a documented
    fallback rather than an unstable conditional distribution.
    """
    probabilities = _transition_probabilities(counts, laplace)[source_state]
    fallback = counts[source_state].sum() < min_source_transitions
    if fallback or probabilities.sum() <= 0:
        weights = np.full(len(residuals), 1.0 / len(residuals))
    else:
        weights = np.zeros(len(residuals), dtype=float)
        for destination, probability in enumerate(probabilities):
            members = states == destination
            if members.any() and probability > 0:
                weights[members] = probability / members.sum()
        if weights.sum() <= 0:
            weights.fill(1.0 / len(weights))
            fallback = True
        else:
            weights /= weights.sum()
    lower, upper = _weighted_quantile(
        residuals, weights, np.array([alpha / 2.0, 1.0 - alpha / 2.0])
    )
    return float(lower), float(upper), probabilities, fallback


def rolling_ifomc_from_fitted(
    fitted: OutOfTimeEnbPIResult,
    dates: pd.Index,
    *,
    config: IFOMCIntervalConfig = IFOMCIntervalConfig(),
) -> IFOMCIntervalResult:
    """Replace M6's residual-window interval by a causal rolling IFOMC interval."""
    forecast = fitted.forecast
    if len(dates) != len(forecast.truth):
        raise ValueError("dates must have one entry per test observation")
    alpha = forecast.config.alpha if config.alpha is None else config.alpha
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1)")
    if config.min_source_transitions < 1:
        raise ValueError("min_source_transitions must be positive")

    raw_training = fitted.crossfit_predictions["final_residual"].to_numpy(float)
    residuals = _bias_adjusted_training_residuals(
        raw_training,
        fitted.selected_residual_window,
        forecast.config.oob_bias_correction,
    )
    edges = _state_edges(residuals, config.n_states)
    n_states = len(edges) - 1
    states = _assign_state(residuals, edges)
    counts = _transition_counts(states, n_states)
    initial_counts = counts.copy()

    truth = np.asarray(forecast.truth, dtype=float)
    point = np.asarray(forecast.point, dtype=float).copy()
    lower, upper = np.empty_like(point), np.empty_like(point)
    source_states = np.empty(len(point), dtype=int)
    lower_offsets, upper_offsets = np.empty_like(point), np.empty_like(point)
    fallback_used = np.empty(len(point), dtype=bool)
    source_transition_counts = np.empty(len(point), dtype=int)

    previous_state = int(states[-1])
    for j, actual in enumerate(truth):
        lo, hi, _, fallback = _conditional_offsets(
            residuals,
            states,
            counts,
            previous_state,
            alpha=alpha,
            laplace=config.laplace,
            min_source_transitions=config.min_source_transitions,
        )
        source_states[j] = previous_state
        lower_offsets[j], upper_offsets[j] = lo, hi
        fallback_used[j] = fallback
        source_transition_counts[j] = int(counts[previous_state].sum())
        lower[j], upper[j] = point[j] + lo, point[j] + hi
        if config.support_lower is not None:
            lower[j] = max(lower[j], float(config.support_lower))

        # The current response is revealed only after its interval is issued.
        new_residual = float(actual - point[j])
        new_state = int(_assign_state(np.array([new_residual]), edges)[0])
        counts[previous_state, new_state] += 1.0
        residuals = np.r_[residuals, new_residual]
        states = np.r_[states, new_state]
        previous_state = new_state

    labels = [f"S{i + 1}" for i in range(n_states)]
    initial_probabilities = _transition_probabilities(initial_counts, config.laplace)
    final_probabilities = _transition_probabilities(counts, config.laplace)
    predictions = pd.DataFrame(
        {
            "date": dates,
            "method": "M6 OOT EnbPI + rolling IFOMC",
            "truth": truth,
            "point": point,
            "lower": lower,
            "upper": upper,
            "source_state": source_states + 1,
            "source_transition_count": source_transition_counts,
            "lower_residual_quantile": lower_offsets,
            "upper_residual_quantile": upper_offsets,
            "fallback_used": fallback_used,
        }
    )
    summary = pd.DataFrame(
        [
            {
                "method": "M6 OOT EnbPI + rolling IFOMC",
                "requested_states": config.n_states,
                "actual_states": n_states,
                "laplace": config.laplace,
                "min_source_transitions": config.min_source_transitions,
                "fallback_rate": float(np.mean(fallback_used)),
                **_local_scale_metrics(truth, point, lower, upper, alpha),
            }
        ]
    )
    state_table = pd.DataFrame(
        {
            "state": labels,
            "lower_edge": edges[:-1],
            "upper_edge": edges[1:],
            "initial_count": np.bincount(_assign_state(
                _bias_adjusted_training_residuals(
                    raw_training,
                    fitted.selected_residual_window,
                    forecast.config.oob_bias_correction,
                ),
                edges,
            ), minlength=n_states),
            "final_count": np.bincount(states, minlength=n_states),
        }
    )
    return IFOMCIntervalResult(
        predictions=predictions,
        summary=summary,
        state_table=state_table,
        initial_transition_matrix=pd.DataFrame(
            initial_probabilities, index=labels, columns=labels
        ),
        final_transition_matrix=pd.DataFrame(
            final_probabilities, index=labels, columns=labels
        ),
    )
