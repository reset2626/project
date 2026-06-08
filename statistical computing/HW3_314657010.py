import matplotlib
import numpy as np
import os
import time
import warnings

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tools.sm_exceptions import ConvergenceWarning


def kalman_filter(observations, process_variance=0.5, measurement_variance=10.0, initial_error=1.0):
    """Match the 1D Kalman smoothing procedure used in paper38."""
    initial_estimate = float(observations[0])
    estimated_state = initial_estimate
    estimate_error = float(initial_error)
    estimated_states = []
    for measurement in observations:
        predicted_state = estimated_state
        predicted_error = estimate_error + process_variance

        kalman_gain = predicted_error / (predicted_error + measurement_variance)
        estimated_state = predicted_state + kalman_gain * (measurement - predicted_state)
        estimate_error = (1.0 - kalman_gain) * predicted_error

        estimated_states.append(estimated_state)

    return np.array(estimated_states, dtype=float)


def decompose_with_simple_kf(series, process_variance=0.5, measurement_variance=10.0):
    estimated_low = kalman_filter(
        series,
        process_variance=process_variance,
        measurement_variance=measurement_variance,
        initial_error=1.0
    )
    estimated_high = np.array(series, dtype=float) - estimated_low
    return estimated_low, estimated_high


def create_windows(series, window_size):
    x_data = []
    y_data = []

    for idx in range(window_size, len(series)):
        x_data.append(series[idx - window_size:idx])
        y_data.append(series[idx])

    return np.array(x_data), np.array(y_data)


def recursive_ann_forecast(
    model,
    x_scaler,
    y_scaler,
    history,
    horizon,
    window_size,
    clip_bounds=None,
    innovation_sequence=None
):
    rolling = list(history[-window_size:])
    forecasts = []

    for step_idx in range(horizon):
        features = np.array(rolling[-window_size:], dtype=float).reshape(1, -1)
        features_scaled = x_scaler.transform(features)
        pred_scaled = model.predict(features_scaled).reshape(-1, 1)
        pred = y_scaler.inverse_transform(pred_scaled)[0, 0]
        if innovation_sequence is not None:
            pred = pred + float(innovation_sequence[step_idx])
        if clip_bounds is not None:
            # Prevent recursive forecasts from drifting far outside the training support.
            pred = float(np.clip(pred, clip_bounds[0], clip_bounds[1]))
        forecasts.append(pred)
        rolling.append(pred)

    return np.array(forecasts)


def find_arima_order(dataset, p_max=4, q_max=4):
    best_aic = np.inf
    best_order = (1, 0, 0)

    for p in range(p_max + 1):
        for q in range(q_max + 1):
            try:
                model = ARIMA(
                    dataset,
                    order=(p, 0, q),
                    enforce_stationarity=False,
                    enforce_invertibility=False
                )
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", ConvergenceWarning)
                    warnings.simplefilter("ignore", UserWarning)
                    result = model.fit()
                if result.aic < best_aic:
                    best_aic = result.aic
                    best_order = (p, 0, q)
            except Exception:
                continue

    return best_order


def fit_arima_low_component(low_train, horizon, arima_order):
    rolling_history = list(np.asarray(low_train, dtype=float))
    forecasts = []

    for _ in range(horizon):
        arima_model = ARIMA(
            rolling_history,
            order=arima_order,
            trend="c",
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            warnings.simplefilter("ignore", UserWarning)
            arima_result = arima_model.fit()
        pred = arima_result.get_forecast(1).predicted_mean[0]
        forecasts.append(pred)
        rolling_history.append(pred)

    return np.array(forecasts)


def fit_ann_model_bundle(high_train, window_size):
    x_train, y_train = create_windows(high_train, window_size)
    if len(x_train) == 0:
        raise ValueError("Not enough samples for ANN windows. Increase training size or reduce window_size.")

    x_scaler = StandardScaler()
    y_scaler = StandardScaler()
    x_train_scaled = x_scaler.fit_transform(x_train)
    y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1, 1)).ravel()

    ann_model = MLPRegressor(
        hidden_layer_sizes=(10, 10, 10),
        activation="relu",
        solver="adam",
        alpha=1e-5,
        learning_rate_init=0.001,
        max_iter=3000,
        random_state=42
    )
    ann_model.fit(x_train_scaled, y_train_scaled)

    # Use a robust empirical range from the training high component as a stability guard
    # for recursive multi-step ANN forecasts.
    lower_q, upper_q = np.quantile(high_train, [0.01, 0.99])
    iqr = np.subtract(*np.quantile(high_train, [0.75, 0.25]))
    margin = max(0.5 * iqr, 0.25)
    clip_bounds = (float(lower_q - margin), float(upper_q + margin))

    return {
        "model": ann_model,
        "x_scaler": x_scaler,
        "y_scaler": y_scaler,
        "clip_bounds": clip_bounds,
        "x_train_scaled": x_train_scaled,
        "y_train": y_train,
        "window_size": window_size,
    }


def fit_ann_high_component(high_train, horizon, window_size):
    ann_bundle = fit_ann_model_bundle(high_train, window_size)

    high_forecast = recursive_ann_forecast(
        ann_bundle["model"],
        ann_bundle["x_scaler"],
        ann_bundle["y_scaler"],
        high_train,
        horizon,
        window_size,
        clip_bounds=ann_bundle["clip_bounds"]
    )

    return high_forecast


def fit_hybrid_in_sample_residuals(train_series, arima_order, window_size):
    estimated_low, estimated_high = decompose_with_simple_kf(train_series)

    low_model = ARIMA(
        estimated_low,
        order=arima_order,
        trend="c",
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        warnings.simplefilter("ignore", UserWarning)
        low_result = low_model.fit()
    low_fitted = np.asarray(low_result.fittedvalues, dtype=float)

    ann_bundle = fit_ann_model_bundle(estimated_high, window_size)
    high_fitted_scaled = ann_bundle["model"].predict(ann_bundle["x_train_scaled"]).reshape(-1, 1)
    high_fitted = ann_bundle["y_scaler"].inverse_transform(high_fitted_scaled).ravel()

    combined_fitted = low_fitted[window_size:] + high_fitted
    actual_aligned = np.asarray(train_series, dtype=float)[window_size:]
    combined_residuals = actual_aligned - combined_fitted
    centered_combined_residuals = combined_residuals - np.mean(combined_residuals)
    high_residuals = ann_bundle["y_train"] - high_fitted
    centered_high_residuals = high_residuals - np.mean(high_residuals)

    return {
        "combined_residuals": np.asarray(combined_residuals, dtype=float),
        "centered_combined_residuals": np.asarray(centered_combined_residuals, dtype=float),
        "centered_high_residuals": np.asarray(centered_high_residuals, dtype=float),
        "ann_bundle": ann_bundle,
        "estimated_high": estimated_high,
    }


def hybrid_point_forecast(train_series, horizon, arima_order, window_size=30):
    estimated_low, estimated_high = decompose_with_simple_kf(train_series)
    low_forecast = fit_arima_low_component(estimated_low, horizon, arima_order=arima_order)
    high_forecast = fit_ann_high_component(estimated_high, horizon, window_size=window_size)

    return {
        "estimated_low": estimated_low,
        "estimated_high": estimated_high,
        "low_forecast": low_forecast,
        "high_forecast": high_forecast,
        "final_forecast": low_forecast + high_forecast,
    }


def insample_calibration_residuals(series, train_size, arima_order, window_size=30):
    train_series = np.asarray(series[:train_size], dtype=float)
    return fit_hybrid_in_sample_residuals(train_series, arima_order=arima_order, window_size=window_size)


def rolling_calibration_residuals(series, initial_train_size, calibration_end, block_size, arima_order, window_size=30):
    residuals = []

    for start in range(initial_train_size, calibration_end, block_size):
        stop = min(start + block_size, calibration_end)
        horizon = stop - start
        train_series = np.asarray(series[:start], dtype=float)
        actual_block = np.asarray(series[start:stop], dtype=float)

        forecast_bundle = hybrid_point_forecast(
            train_series,
            horizon,
            arima_order=arima_order,
            window_size=window_size
        )
        block_residuals = actual_block - forecast_bundle["final_forecast"]
        residuals.extend(block_residuals.tolist())

    return np.asarray(residuals, dtype=float)


def sort_bootstrap_residuals(residual_pool, horizon, rng):
    sampled_indices = rng.choice(len(residual_pool), size=horizon, replace=True)
    sampled_indices.sort()
    return residual_pool[sampled_indices]


def giordano_style_prediction_interval(
    point_forecast,
    calibration_residuals,
    horizon,
    alpha=0.05,
    bootstrap_replications=300,
    random_seed=123
):
    rng = np.random.default_rng(random_seed)
    bootstrap_paths = np.zeros((bootstrap_replications, horizon), dtype=float)

    for boot_idx in range(bootstrap_replications):
        sampled_residuals = sort_bootstrap_residuals(calibration_residuals, horizon, rng)
        bootstrap_paths[boot_idx, :] = point_forecast + sampled_residuals

    interval_lower = np.quantile(bootstrap_paths, alpha / 2, axis=0)
    interval_upper = np.quantile(bootstrap_paths, 1 - alpha / 2, axis=0)
    return interval_lower, interval_upper


def bootstrap_paths_for_interval(
    point_forecast,
    calibration_residuals,
    horizon,
    bootstrap_replications=300,
    random_seed=123
):
    rng = np.random.default_rng(random_seed)
    bootstrap_paths = np.zeros((bootstrap_replications, horizon), dtype=float)

    for boot_idx in range(bootstrap_replications):
        sampled_residuals = sort_bootstrap_residuals(calibration_residuals, horizon, rng)
        bootstrap_paths[boot_idx, :] = point_forecast + sampled_residuals

    return bootstrap_paths


def normal_bootstrap_interval(bootstrap_paths, z_value=1.959963984540054):
    path_mean = np.mean(bootstrap_paths, axis=0)
    path_std = np.std(bootstrap_paths, axis=0, ddof=1)
    interval_lower = path_mean - z_value * path_std
    interval_upper = path_mean + z_value * path_std
    return interval_lower, interval_upper


def summarize_estimator(forecasts, truths):
    forecasts = np.asarray(forecasts, dtype=float)
    truths = np.asarray(truths, dtype=float)
    errors = forecasts - truths
    bias = np.mean(errors, axis=0)
    variance = np.var(forecasts, axis=0, ddof=1)
    mse = np.mean(errors ** 2, axis=0)

    return {
        "bias_by_horizon": bias,
        "variance_by_horizon": variance,
        "mse_by_horizon": mse,
        "mean_abs_bias": float(np.mean(np.abs(bias))),
        "mean_variance": float(np.mean(variance)),
        "mean_mse": float(np.mean(mse)),
        "h1_bias": float(bias[0]),
        "h1_variance": float(variance[0]),
        "h1_mse": float(mse[0]),
        "errors_h1": errors[:, 0],
    }


def bootstrap_mean_test(sample, null_value=0.0, bootstrap_replications=2000, random_seed=321):
    sample = np.asarray(sample, dtype=float)
    observed_mean = float(np.mean(sample))
    centered_sample = sample - observed_mean + null_value
    rng = np.random.default_rng(random_seed)

    bootstrap_means = np.empty(bootstrap_replications, dtype=float)
    for idx in range(bootstrap_replications):
        draw = centered_sample[rng.choice(len(centered_sample), size=len(centered_sample), replace=True)]
        bootstrap_means[idx] = np.mean(draw)

    p_value = float(np.mean(np.abs(bootstrap_means - null_value) >= abs(observed_mean - null_value)))
    ci_draws = sample[rng.choice(len(sample), size=(bootstrap_replications, len(sample)), replace=True)].mean(axis=1)
    ci_lower, ci_upper = np.quantile(ci_draws, [0.025, 0.975])

    return {
        "observed_mean": observed_mean,
        "p_value": p_value,
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
    }


def simulate_babu_style_data(n_steps, noise_std=0.15):
    """Generate a Giordano-inspired synthetic series with linear M1 and nonlinear M9 components."""
    low_true = np.zeros(n_steps, dtype=float)
    high_true = np.zeros(n_steps, dtype=float)

    low_true[0] = np.random.normal(0, 0.2)
    high_true[0] = np.random.normal(0, 0.2)

    for idx in range(1, n_steps):
        low_true[idx] = 0.6 * low_true[idx - 1] + np.random.normal(0, 0.2)
        high_true[idx] = 0.8 * high_true[idx - 1] - 0.8 * high_true[idx - 1] / (1.0 + np.exp(-10.0 * high_true[idx - 1])) + np.random.normal(0, 0.2)

    noise = np.random.normal(0, noise_std, size=n_steps)
    measurements = low_true + high_true + noise
    return low_true, high_true, measurements


def run_single_experiment(
    train_size=300,
    horizon=1,
    window_size=15,
    bootstrap_replications=300,
    initial_train_size=None,
    calibration_block=None
):
    dt = 1.0
    n_steps = train_size + horizon
    t = np.arange(n_steps) * dt

    low_true, high_true, measurements = simulate_babu_style_data(n_steps)

    calibration_low, _ = decompose_with_simple_kf(measurements[:train_size])
    arima_order = find_arima_order(calibration_low, p_max=3, q_max=3)
    if initial_train_size is None:
        initial_train_size = max(window_size + 20, train_size // 2)
    if calibration_block is None:
        calibration_block = max(1, min(10, max(1, train_size - initial_train_size)))

    calibration_residuals = rolling_calibration_residuals(
        measurements,
        initial_train_size=initial_train_size,
        calibration_end=train_size,
        block_size=calibration_block,
        arima_order=arima_order,
        window_size=window_size
    )

    full_train_bundle = hybrid_point_forecast(
        measurements[:train_size],
        horizon,
        arima_order=arima_order,
        window_size=window_size
    )

    estimated_low_full, estimated_high_full = decompose_with_simple_kf(measurements)
    low_forecast = full_train_bundle["low_forecast"]
    high_forecast = full_train_bundle["high_forecast"]
    final_forecast = full_train_bundle["final_forecast"]

    bootstrap_paths = bootstrap_paths_for_interval(
        final_forecast,
        calibration_residuals,
        horizon,
        bootstrap_replications=bootstrap_replications
    )

    interval_lower, interval_upper = giordano_style_prediction_interval(
        final_forecast,
        calibration_residuals,
        horizon,
        alpha=0.05,
        bootstrap_replications=bootstrap_replications
    )
    normal_interval_lower, normal_interval_upper = normal_bootstrap_interval(bootstrap_paths)

    t_test = t[train_size:]
    measurements_test = measurements[train_size:]
    true_low_test = low_true[train_size:]
    true_high_test = high_true[train_size:]
    true_clean_test = true_low_test + true_high_test

    low_rmse = np.sqrt(np.mean((low_forecast - true_low_test) ** 2))
    high_rmse = np.sqrt(np.mean((high_forecast - true_high_test) ** 2))
    low_mse = np.mean((low_forecast - true_low_test) ** 2)
    high_mse = np.mean((high_forecast - true_high_test) ** 2)
    final_clean_rmse = np.sqrt(np.mean((final_forecast - true_clean_test) ** 2))
    final_clean_mse = np.mean((final_forecast - true_clean_test) ** 2)
    final_noisy_rmse = np.sqrt(np.mean((final_forecast - measurements_test) ** 2))
    final_noisy_mse = np.mean((final_forecast - measurements_test) ** 2)
    calibration_residual_mean = np.mean(calibration_residuals)
    calibration_residual_std = np.std(calibration_residuals, ddof=1)
    interval_widths = interval_upper - interval_lower
    mean_interval_width = np.mean(interval_widths)
    median_interval_width = np.median(interval_widths)
    normal_interval_widths = normal_interval_upper - normal_interval_lower
    normal_mean_interval_width = np.mean(normal_interval_widths)
    normal_median_interval_width = np.median(normal_interval_widths)
    clean_in_interval = (true_clean_test >= interval_lower) & (true_clean_test <= interval_upper)
    noisy_in_interval = (measurements_test >= interval_lower) & (measurements_test <= interval_upper)
    clean_coverage = np.mean(clean_in_interval) * 100
    noisy_coverage = np.mean(noisy_in_interval) * 100
    clean_in_normal_interval = (true_clean_test >= normal_interval_lower) & (true_clean_test <= normal_interval_upper)
    noisy_in_normal_interval = (measurements_test >= normal_interval_lower) & (measurements_test <= normal_interval_upper)
    normal_clean_coverage = np.mean(clean_in_normal_interval) * 100
    normal_noisy_coverage = np.mean(noisy_in_normal_interval) * 100

    return {
        "t": t,
        "t_test": t_test,
        "measurements": measurements,
        "measurements_test": measurements_test,
        "low_true": low_true,
        "high_true": high_true,
        "true_low_test": true_low_test,
        "true_high_test": true_high_test,
        "true_clean_test": true_clean_test,
        "estimated_low_full": estimated_low_full,
        "estimated_high_full": estimated_high_full,
        "low_forecast": low_forecast,
        "high_forecast": high_forecast,
        "final_forecast": final_forecast,
        "interval_lower": interval_lower,
        "interval_upper": interval_upper,
        "normal_interval_lower": normal_interval_lower,
        "normal_interval_upper": normal_interval_upper,
        "clean_in_interval": clean_in_interval,
        "noisy_in_interval": noisy_in_interval,
        "clean_in_normal_interval": clean_in_normal_interval,
        "noisy_in_normal_interval": noisy_in_normal_interval,
        "arima_order": arima_order,
        "calibration_residuals": calibration_residuals,
        "calibration_residual_mean": calibration_residual_mean,
        "calibration_residual_std": calibration_residual_std,
        "mean_interval_width": mean_interval_width,
        "median_interval_width": median_interval_width,
        "normal_mean_interval_width": normal_mean_interval_width,
        "normal_median_interval_width": normal_median_interval_width,
        "low_rmse": low_rmse,
        "high_rmse": high_rmse,
        "low_mse": low_mse,
        "high_mse": high_mse,
        "final_clean_rmse": final_clean_rmse,
        "final_clean_mse": final_clean_mse,
        "final_noisy_rmse": final_noisy_rmse,
        "final_noisy_mse": final_noisy_mse,
        "clean_coverage": clean_coverage,
        "noisy_coverage": noisy_coverage,
        "normal_clean_coverage": normal_clean_coverage,
        "normal_noisy_coverage": normal_noisy_coverage,
    }


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    warnings.filterwarnings("ignore", category=UserWarning)

    np.random.seed(42)
    start_time = time.perf_counter()
    n_monte_carlo = 100
    train_size = 500
    horizons = [60]
    bootstrap_replications = 500

    print(f"Monte Carlo runs               : {n_monte_carlo}")
    print(f"Observed sample size T         : {train_size}")
    print(f"Bootstrap replications B       : {bootstrap_replications}")
    print("")
    print("h | Mean final MSE | Percentile coverage | Normal coverage | Percentile width | Normal width")
    print("--|----------------|---------------------|-----------------|------------------|-------------")

    for horizon in horizons:
        experiment_results = []
        for run_idx in range(n_monte_carlo):
            np.random.seed(42 + run_idx)
            experiment_results.append(
                run_single_experiment(
                    train_size=train_size,
                    horizon=horizon,
                    bootstrap_replications=bootstrap_replications
                )
            )

        clean_mses = np.array([result["final_clean_mse"] for result in experiment_results], dtype=float)
        clean_coverages = np.array([result["clean_coverage"] for result in experiment_results], dtype=float)
        normal_clean_coverages = np.array([result["normal_clean_coverage"] for result in experiment_results], dtype=float)
        mean_interval_widths = np.array([result["mean_interval_width"] for result in experiment_results], dtype=float)
        normal_mean_interval_widths = np.array([result["normal_mean_interval_width"] for result in experiment_results], dtype=float)

        final_forecasts = np.array([result["final_forecast"] for result in experiment_results], dtype=float)
        final_truths = np.array([result["true_clean_test"] for result in experiment_results], dtype=float)
        estimator_summary = summarize_estimator(final_forecasts, final_truths)
        mean_error_test = bootstrap_mean_test(estimator_summary["errors_h1"], null_value=0.0)

        print(
            f"{horizon} | "
            f"{np.mean(clean_mses):.4f}         | "
            f"{np.mean(clean_coverages):.2f}%                | "
            f"{np.mean(normal_clean_coverages):.2f}%            | "
            f"{np.mean(mean_interval_widths):.4f}           | "
            f"{np.mean(normal_mean_interval_widths):.4f}"
        )

        print("")
        print("Final forecast estimator summary")
        print(f"Mean absolute bias              : {estimator_summary['mean_abs_bias']:.4f}")
        print(f"Mean variance                  : {estimator_summary['mean_variance']:.4f}")
        print(f"Mean MSE                       : {estimator_summary['mean_mse']:.4f}")
        print(f"Horizon 1 bias                 : {estimator_summary['h1_bias']:.4f}")
        print(f"Horizon 1 variance             : {estimator_summary['h1_variance']:.4f}")
        print(f"Horizon 1 MSE                  : {estimator_summary['h1_mse']:.4f}")
        print("")
        print("Bootstrap CI comparison")
        print(f"Percentile mean width          : {np.mean(mean_interval_widths):.4f}")
        print(f"Normal mean width              : {np.mean(normal_mean_interval_widths):.4f}")
        print(f"Percentile mean clean coverage : {np.mean(clean_coverages):.2f}%")
        print(f"Normal mean clean coverage     : {np.mean(normal_clean_coverages):.2f}%")
        print("")
        print("Bootstrap hypothesis test on horizon-1 forecast error")
        print("H0: E[e1] = 0")
        print(f"Observed mean error            : {mean_error_test['observed_mean']:.4f}")
        print(f"Bootstrap p-value              : {mean_error_test['p_value']:.4f}")
        print(f"95% bootstrap CI               : [{mean_error_test['ci_lower']:.4f}, {mean_error_test['ci_upper']:.4f}]")

    elapsed_seconds = time.perf_counter() - start_time
    print("")
    print(f"Elapsed time                   : {elapsed_seconds:.2f} seconds")
