"""Shared building blocks for the DistPred experiment notebooks."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


class DistPredANNRegressor(nn.Module):
    """Small ensemble-output MLP used by the component DistPred models."""

    def __init__(self, input_dim, ensemble_size, hidden_dim_1=10, hidden_dim_2=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim_1),
            nn.ReLU(),
            nn.Linear(hidden_dim_1, hidden_dim_2),
            nn.ReLU(),
            nn.Linear(hidden_dim_2, ensemble_size),
        )

    def forward(self, x):
        return self.net(x)


def component_correlation_summary(linear_part, nonlinear_part):
    linear_part = np.asarray(linear_part, dtype=float)
    nonlinear_part = np.asarray(nonlinear_part, dtype=float)

    if linear_part.shape != nonlinear_part.shape:
        raise ValueError("linear_part and nonlinear_part must have the same shape.")

    if linear_part.size < 2:
        return {
            "pearson_corr": np.nan,
            "abs_pearson_corr": np.nan,
            "corr_sq": np.nan,
            "covariance": np.nan,
        }

    pearson_corr = float(np.corrcoef(linear_part, nonlinear_part)[0, 1])
    covariance = float(np.cov(linear_part, nonlinear_part, ddof=1)[0, 1])
    return {
        "pearson_corr": pearson_corr,
        "abs_pearson_corr": abs(pearson_corr),
        "corr_sq": pearson_corr**2,
        "covariance": covariance,
    }


def component_std_summary(low_part, high_part):
    low_part = np.asarray(low_part, dtype=float)
    high_part = np.asarray(high_part, dtype=float)

    if low_part.shape != high_part.shape:
        raise ValueError("low_part and high_part must have the same shape.")

    return {
        "low_std": float(np.std(low_part, ddof=1)) if low_part.size > 1 else np.nan,
        "high_std": float(np.std(high_part, ddof=1)) if high_part.size > 1 else np.nan,
    }


def rolling_std_correlation_summary(low_part, high_part, window_size=15):
    low_part = np.asarray(low_part, dtype=float)
    high_part = np.asarray(high_part, dtype=float)

    if low_part.shape != high_part.shape:
        raise ValueError("low_part and high_part must have the same shape.")

    if low_part.size < window_size or window_size < 2:
        return {
            "rolling_std_corr": np.nan,
            "abs_rolling_std_corr": np.nan,
            "rolling_std_corr_sq": np.nan,
            "rolling_low_std_mean": np.nan,
            "rolling_high_std_mean": np.nan,
        }

    low_std = np.array(
        [
            np.std(low_part[idx - window_size : idx], ddof=1)
            for idx in range(window_size, low_part.size + 1)
        ]
    )
    high_std = np.array(
        [
            np.std(high_part[idx - window_size : idx], ddof=1)
            for idx in range(window_size, high_part.size + 1)
        ]
    )

    if np.std(low_std, ddof=1) < 1e-12 or np.std(high_std, ddof=1) < 1e-12:
        corr = np.nan
    else:
        corr = float(np.corrcoef(low_std, high_std)[0, 1])

    return {
        "rolling_std_corr": corr,
        "abs_rolling_std_corr": abs(corr) if np.isfinite(corr) else np.nan,
        "rolling_std_corr_sq": corr**2 if np.isfinite(corr) else np.nan,
        "rolling_low_std_mean": float(np.mean(low_std)),
        "rolling_high_std_mean": float(np.mean(high_std)),
    }


def create_joint_windows(low_series, high_series, window_size):
    features = []
    targets = []
    for idx in range(window_size, len(low_series)):
        low_window = np.asarray(low_series[idx - window_size : idx], dtype=float)
        high_window = np.asarray(high_series[idx - window_size : idx], dtype=float)
        joint_window = np.column_stack([low_window, high_window]).reshape(-1)
        features.append(joint_window)
        targets.append([float(low_series[idx]), float(high_series[idx])])
    return np.asarray(features, dtype=float), np.asarray(targets, dtype=float)


def discrete_crps_from_samples(samples, y_true):
    samples = np.asarray(samples, dtype=float)
    y_true = np.asarray(y_true, dtype=float)
    if samples.ndim != 2:
        raise ValueError("samples must have shape (ensemble_size, horizon).")
    if y_true.ndim != 1 or y_true.shape[0] != samples.shape[1]:
        raise ValueError("y_true must have shape (horizon,).")

    crps_values = []
    for step_idx in range(samples.shape[1]):
        ensemble = samples[:, step_idx]
        obs = y_true[step_idx]
        calibration = np.mean(np.abs(ensemble - obs))
        spread = np.mean(np.abs(ensemble[:, None] - ensemble[None, :]))
        crps_values.append(calibration - 0.5 * spread)
    return float(np.mean(crps_values))


def distpred_eq13_loss(pred_ensemble, y_true):
    if pred_ensemble.ndim != 2:
        raise ValueError("pred_ensemble must have shape (batch_size, ensemble_size).")

    ensemble_size = pred_ensemble.shape[1]
    if ensemble_size < 2:
        raise ValueError("Eq.13 requires ensemble_size >= 2.")

    y_true = y_true.unsqueeze(1)
    calibration_term = torch.abs(pred_ensemble - y_true).mean(dim=1)
    sorted_pred, _ = torch.sort(pred_ensemble, dim=1)
    order = torch.arange(
        ensemble_size, device=pred_ensemble.device, dtype=pred_ensemble.dtype
    )
    l_moment_term = sorted_pred.mean(dim=1) - (
        2.0 / (ensemble_size * (ensemble_size - 1))
    ) * (sorted_pred * order.unsqueeze(0)).sum(dim=1)
    return (calibration_term + l_moment_term).mean()


def picp_from_interval(interval_lower, interval_upper, y_true):
    interval_lower = np.asarray(interval_lower, dtype=float)
    interval_upper = np.asarray(interval_upper, dtype=float)
    y_true = np.asarray(y_true, dtype=float)
    covered = (y_true >= interval_lower) & (y_true <= interval_upper)
    return float(np.mean(covered))


def qice_from_samples(samples, y_true, n_bins=10):
    samples = np.asarray(samples, dtype=float)
    y_true = np.asarray(y_true, dtype=float)
    quantile_levels = np.linspace(0.0, 1.0, n_bins + 1)
    quantile_grid = np.quantile(samples, quantile_levels, axis=0)
    counts = np.zeros(n_bins, dtype=float)

    for step_idx, obs in enumerate(y_true):
        boundaries = quantile_grid[:, step_idx]
        bin_idx = np.searchsorted(boundaries, obs, side="right") - 1
        bin_idx = int(np.clip(bin_idx, 0, n_bins - 1))
        counts[bin_idx] += 1.0

    observed_freq = counts / max(len(y_true), 1)
    ideal_freq = np.full(n_bins, 1.0 / n_bins, dtype=float)
    return float(np.mean(np.abs(observed_freq - ideal_freq)) * 100.0)


def simulate_m1_m3_additive_data(
    n_steps,
    noise_std=0.15,
    low_error_std=1.0,
    high_error_base_std=1.0,
    high_error_low_sensitivity=0.5,
):
    low_true = np.zeros(n_steps, dtype=float)
    high_true = np.zeros(n_steps, dtype=float)

    low_true[0] = np.random.normal(0, 1.0)
    high_true[0] = np.random.normal(0, 0.5)
    high_true[1] = np.random.normal(0, 0.5)

    for idx in range(1, n_steps):
        low_error = np.random.normal(0, low_error_std)
        high_error_std = high_error_base_std * (
            1.0 + high_error_low_sensitivity * abs(low_true[idx - 1])
        )
        high_error = np.random.normal(0, high_error_std)
        low_true[idx] = 0.6 * low_true[idx - 1] + low_error

        if idx < 2:
            continue

        coeff_1 = 0.5 + 0.9 * np.exp(-(high_true[idx - 1] ** 2))
        coeff_2 = -0.8 - 1.8 * np.exp(-(high_true[idx - 1] ** 2))
        high_mean = coeff_1 * high_true[idx - 1] + coeff_2 * high_true[idx - 2]
        high_true[idx] = high_mean + high_error

    noise = np.random.normal(0, noise_std, size=n_steps)
    measurements = low_true + high_true + noise
    return low_true, high_true, measurements


def simulate_m1_m9_additive_data(
    n_steps,
    noise_std=0.15,
    low_error_std=1.0,
    high_error_base_std=1.0,
    high_error_low_sensitivity=0.5,
):
    low_true = np.zeros(n_steps, dtype=float)
    high_true = np.zeros(n_steps, dtype=float)

    low_true[0] = np.random.normal(0, 1.0)
    high_true[0] = np.random.normal(0, 0.5)

    for idx in range(1, n_steps):
        low_error = np.random.normal(0, low_error_std)
        high_error_std = high_error_base_std * (
            1.0 + high_error_low_sensitivity * abs(low_true[idx - 1])
        )
        high_error = np.random.normal(0, high_error_std)
        low_true[idx] = 0.4 * low_true[idx - 1] + low_error

        logistic_gate = 1.0 / (1.0 + np.exp(-10.0 * high_true[idx - 1]))
        high_mean = (
            1.2 * 0.8 * high_true[idx - 1]
            - 0.6 * high_true[idx - 1] * logistic_gate
        )
        high_true[idx] = high_mean + high_error

    noise = np.random.normal(0, noise_std, size=n_steps)
    measurements = low_true + high_true + noise
    return low_true, high_true, measurements


__all__ = [
    "DistPredANNRegressor",
    "component_correlation_summary",
    "component_std_summary",
    "create_joint_windows",
    "discrete_crps_from_samples",
    "distpred_eq13_loss",
    "picp_from_interval",
    "qice_from_samples",
    "rolling_std_correlation_summary",
    "simulate_m1_m3_additive_data",
    "simulate_m1_m9_additive_data",
]
