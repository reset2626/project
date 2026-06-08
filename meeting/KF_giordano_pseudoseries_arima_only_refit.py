import matplotlib
import numpy as np
import time
import warnings

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tools.sm_exceptions import ConvergenceWarning
import torch
import torch.nn as nn


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


class DistPredMLP(nn.Module):
    def __init__(self, input_dim, ensemble_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 10),
            nn.ReLU(),
            nn.Linear(10, 10),
            nn.ReLU(),
            nn.Linear(10, 5),
            nn.ReLU(),
            nn.Linear(5, ensemble_size),
        )

    def forward(self, x):
        return self.net(x)


def distpred_ensemble_loss(pred_ensemble, y_true):
    y_true = y_true.unsqueeze(1)
    calibration_term = torch.abs(pred_ensemble - y_true).mean(dim=1)
    diversity_term = torch.abs(
        pred_ensemble.unsqueeze(2) - pred_ensemble.unsqueeze(1)
    ).mean(dim=(1, 2))
    mean_term = torch.square(pred_ensemble.mean(dim=1, keepdim=True) - y_true).squeeze(1)
    return (calibration_term - 0.5 * diversity_term + 0.1 * mean_term).mean()


def fit_distpred_high_bundle(high_train, window_size, ensemble_size=50, epochs=250, learning_rate=1e-3):
    x_train, y_train = create_windows(high_train, window_size)
    if len(x_train) == 0:
        raise ValueError("Not enough samples for DistPred windows. Increase training size or reduce window_size.")

    x_scaler = StandardScaler()
    y_scaler = StandardScaler()
    x_train_scaled = x_scaler.fit_transform(x_train)
    y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1, 1)).ravel().astype(np.float32)

    x_tensor = torch.tensor(x_train_scaled, dtype=torch.float32)
    y_tensor = torch.tensor(y_train_scaled, dtype=torch.float32)

    model = DistPredMLP(input_dim=window_size, ensemble_size=ensemble_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        pred_ensemble = model(x_tensor)
        loss = distpred_ensemble_loss(pred_ensemble, y_tensor)
        loss.backward()
        optimizer.step()

    lower_q, upper_q = np.quantile(high_train, [0.01, 0.99])
    iqr = np.subtract(*np.quantile(high_train, [0.75, 0.25]))
    margin = max(0.5 * iqr, 0.25)
    clip_bounds = (float(lower_q - margin), float(upper_q + margin))

    return {
        "model": model.eval(),
        "x_scaler": x_scaler,
        "y_scaler": y_scaler,
        "clip_bounds": clip_bounds,
        "x_train_scaled": x_train_scaled,
        "y_train": y_train,
        "window_size": window_size,
        "ensemble_size": ensemble_size,
    }


def recursive_distpred_high_forecast(bundle, history, horizon, window_size):
    ensemble_size = bundle["ensemble_size"]
    rolling_histories = np.tile(np.asarray(history[-window_size:], dtype=float), (ensemble_size, 1))
    ensemble_paths = np.zeros((ensemble_size, horizon), dtype=float)
    for step_idx in range(horizon):
        features_scaled = bundle["x_scaler"].transform(rolling_histories)
        with torch.no_grad():
            pred_scaled = bundle["model"](torch.tensor(features_scaled, dtype=torch.float32)).cpu().numpy()
        pred = bundle["y_scaler"].inverse_transform(pred_scaled.reshape(-1, 1)).reshape(ensemble_size, ensemble_size)
        next_vals = pred[np.arange(ensemble_size), np.arange(ensemble_size)]
        next_vals = np.clip(next_vals, bundle["clip_bounds"][0], bundle["clip_bounds"][1])
        ensemble_paths[:, step_idx] = next_vals
        rolling_histories = np.concatenate([rolling_histories[:, 1:], next_vals.reshape(-1, 1)], axis=1)
    return ensemble_paths


def fit_distpred_high_component(high_train, horizon, window_size, ensemble_size=50):
    distpred_bundle = fit_distpred_high_bundle(high_train, window_size, ensemble_size=ensemble_size)
    high_samples = recursive_distpred_high_forecast(
        distpred_bundle,
        high_train,
        horizon,
        window_size
    )
    high_forecast = np.mean(high_samples, axis=0)
    return high_forecast, high_samples, distpred_bundle


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

    distpred_bundle = fit_distpred_high_bundle(estimated_high, window_size)
    with torch.no_grad():
        high_fitted_scaled = distpred_bundle["model"](
            torch.tensor(distpred_bundle["x_train_scaled"], dtype=torch.float32)
        ).cpu().numpy()
    high_fitted_ensemble = distpred_bundle["y_scaler"].inverse_transform(
        high_fitted_scaled.reshape(-1, 1)
    ).reshape(high_fitted_scaled.shape)
    high_fitted = high_fitted_ensemble.mean(axis=1)

    combined_fitted = low_fitted[window_size:] + high_fitted
    actual_aligned = np.asarray(train_series, dtype=float)[window_size:]
    combined_residuals = actual_aligned - combined_fitted
    centered_combined_residuals = combined_residuals - np.mean(combined_residuals)
    high_residuals = distpred_bundle["y_train"] - high_fitted
    centered_high_residuals = high_residuals - np.mean(high_residuals)

    return {
        "combined_residuals": np.asarray(combined_residuals, dtype=float),
        "centered_combined_residuals": np.asarray(centered_combined_residuals, dtype=float),
        "centered_high_residuals": np.asarray(centered_high_residuals, dtype=float),
        "distpred_bundle": distpred_bundle,
        "estimated_high": estimated_high,
        "fitted_final": np.asarray(combined_fitted, dtype=float),
        "window_size": window_size,
    }


def hybrid_point_forecast(train_series, horizon, arima_order, window_size=30):
    estimated_low, estimated_high = decompose_with_simple_kf(train_series)
    low_forecast = fit_arima_low_component(estimated_low, horizon, arima_order=arima_order)
    high_forecast, high_samples, distpred_bundle = fit_distpred_high_component(
        estimated_high,
        horizon,
        window_size=window_size
    )

    return {
        "estimated_low": estimated_low,
        "estimated_high": estimated_high,
        "low_forecast": low_forecast,
        "high_forecast": high_forecast,
        "high_samples": high_samples,
        "distpred_bundle": distpred_bundle,
        "final_forecast": low_forecast + high_forecast,
    }


def timed_hybrid_point_forecast(train_series, horizon, arima_order, window_size=30):
    timing = {
        "kf_decomposition": 0.0,
        "arima_forecast": 0.0,
        "ann_fit_forecast": 0.0,
    }

    kf_start = time.perf_counter()
    estimated_low, estimated_high = decompose_with_simple_kf(train_series)
    timing["kf_decomposition"] += time.perf_counter() - kf_start

    arima_start = time.perf_counter()
    low_forecast = fit_arima_low_component(estimated_low, horizon, arima_order=arima_order)
    timing["arima_forecast"] += time.perf_counter() - arima_start

    ann_start = time.perf_counter()
    high_forecast, high_samples, distpred_bundle = fit_distpred_high_component(
        estimated_high,
        horizon,
        window_size=window_size
    )
    timing["ann_fit_forecast"] += time.perf_counter() - ann_start

    return {
        "estimated_low": estimated_low,
        "estimated_high": estimated_high,
        "low_forecast": low_forecast,
        "high_forecast": high_forecast,
        "high_samples": high_samples,
        "distpred_bundle": distpred_bundle,
        "final_forecast": low_forecast + high_forecast,
    }, timing


def timed_hybrid_point_forecast_arima_only(
    train_series,
    horizon,
    arima_order,
    fixed_high_forecast,
    fixed_high_samples
):
    timing = {
        "kf_decomposition": 0.0,
        "arima_forecast": 0.0,
        "ann_fit_forecast": 0.0,
    }

    kf_start = time.perf_counter()
    estimated_low, estimated_high = decompose_with_simple_kf(train_series)
    timing["kf_decomposition"] += time.perf_counter() - kf_start

    arima_start = time.perf_counter()
    low_forecast = fit_arima_low_component(estimated_low, horizon, arima_order=arima_order)
    timing["arima_forecast"] += time.perf_counter() - arima_start

    high_forecast = np.asarray(fixed_high_forecast, dtype=float)
    high_samples = np.asarray(fixed_high_samples, dtype=float)

    return {
        "estimated_low": estimated_low,
        "estimated_high": estimated_high,
        "low_forecast": low_forecast,
        "high_forecast": high_forecast,
        "high_samples": high_samples,
        "final_forecast": low_forecast + high_forecast,
    }, timing


def insample_calibration_residuals(series, train_size, arima_order, window_size=30):
    train_series = np.asarray(series[:train_size], dtype=float)
    return fit_hybrid_in_sample_residuals(train_series, arima_order=arima_order, window_size=window_size)


def sort_bootstrap_residuals(residual_pool, horizon, rng):
    sampled_indices = rng.choice(len(residual_pool), size=horizon, replace=True)
    sampled_indices.sort()
    return residual_pool[sampled_indices]


def build_pseudo_series(train_series, fitted_final, centered_combined_residuals, window_size, rng):
    train_series = np.asarray(train_series, dtype=float)
    pseudo_series = train_series.copy()
    sampled_residuals = sort_bootstrap_residuals(centered_combined_residuals, len(fitted_final), rng)
    pseudo_series[window_size:] = fitted_final + sampled_residuals
    return pseudo_series


def pseudoseries_prediction_interval(
    train_series,
    arima_order,
    window_size,
    fitted_final,
    centered_combined_residuals,
    fixed_high_forecast,
    fixed_high_samples,
    horizon,
    alpha=0.05,
    bootstrap_replications=300,
    random_seed=123
):
    rng = np.random.default_rng(random_seed)
    bootstrap_paths = []
    timing_info = {
        "pseudo_series_generation": 0.0,
        "bootstrap_refit_forecast": 0.0,
        "bootstrap_kf_decomposition": 0.0,
        "bootstrap_arima_forecast": 0.0,
        "bootstrap_ann_fit_forecast": 0.0,
    }

    for boot_idx in range(bootstrap_replications):
        boot_start = time.perf_counter()
        pseudo_series = build_pseudo_series(
            train_series,
            fitted_final,
            centered_combined_residuals,
            window_size,
            rng
        )
        timing_info["pseudo_series_generation"] += time.perf_counter() - boot_start

        refit_start = time.perf_counter()
        pseudo_forecast_bundle, bootstrap_detail_timing = timed_hybrid_point_forecast_arima_only(
            pseudo_series,
            horizon,
            arima_order=arima_order,
            fixed_high_forecast=fixed_high_forecast,
            fixed_high_samples=fixed_high_samples
        )
        timing_info["bootstrap_refit_forecast"] += time.perf_counter() - refit_start
        timing_info["bootstrap_kf_decomposition"] += bootstrap_detail_timing["kf_decomposition"]
        timing_info["bootstrap_arima_forecast"] += bootstrap_detail_timing["arima_forecast"]
        timing_info["bootstrap_ann_fit_forecast"] += bootstrap_detail_timing["ann_fit_forecast"]
        low_forecast = pseudo_forecast_bundle["low_forecast"]
        final_paths = np.asarray(fixed_high_samples, dtype=float) + low_forecast.reshape(1, -1)
        bootstrap_paths.append(final_paths)

    bootstrap_paths = np.concatenate(bootstrap_paths, axis=0)
    interval_lower = np.quantile(bootstrap_paths, alpha / 2, axis=0)
    interval_upper = np.quantile(bootstrap_paths, 1 - alpha / 2, axis=0)
    return interval_lower, interval_upper, timing_info


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


def run_single_experiment(train_size=300, horizon=1, window_size=15, bootstrap_replications=300):
    timing_info = {
        "data_generation": 0.0,
        "arima_order_selection": 0.0,
        "insample_residual_fit": 0.0,
        "point_forecast_fit": 0.0,
        "pseudo_series_generation": 0.0,
        "bootstrap_refit_forecast": 0.0,
        "bootstrap_kf_decomposition": 0.0,
        "bootstrap_arima_forecast": 0.0,
        "bootstrap_ann_fit_forecast": 0.0,
    }
    dt = 1.0
    n_steps = train_size + horizon
    t = np.arange(n_steps) * dt

    data_start = time.perf_counter()
    low_true, high_true, measurements = simulate_babu_style_data(n_steps)
    timing_info["data_generation"] += time.perf_counter() - data_start

    order_start = time.perf_counter()
    calibration_low, _ = decompose_with_simple_kf(measurements[:train_size])
    arima_order = find_arima_order(calibration_low, p_max=3, q_max=3)
    timing_info["arima_order_selection"] += time.perf_counter() - order_start

    insample_start = time.perf_counter()
    calibration_residuals = insample_calibration_residuals(
        measurements,
        train_size=train_size,
        arima_order=arima_order,
        window_size=window_size
    )
    timing_info["insample_residual_fit"] += time.perf_counter() - insample_start

    point_start = time.perf_counter()
    full_train_bundle = hybrid_point_forecast(
        measurements[:train_size],
        horizon,
        arima_order=arima_order,
        window_size=window_size
    )
    timing_info["point_forecast_fit"] += time.perf_counter() - point_start

    estimated_low_full, estimated_high_full = decompose_with_simple_kf(measurements)
    low_forecast = full_train_bundle["low_forecast"]
    high_forecast = full_train_bundle["high_forecast"]
    final_forecast = full_train_bundle["final_forecast"]

    interval_lower, interval_upper, bootstrap_timing = pseudoseries_prediction_interval(
        measurements[:train_size],
        arima_order,
        window_size,
        calibration_residuals["fitted_final"],
        calibration_residuals["centered_combined_residuals"],
        full_train_bundle["high_forecast"],
        full_train_bundle["high_samples"],
        horizon,
        alpha=0.05,
        bootstrap_replications=bootstrap_replications
    )
    timing_info["pseudo_series_generation"] += bootstrap_timing["pseudo_series_generation"]
    timing_info["bootstrap_refit_forecast"] += bootstrap_timing["bootstrap_refit_forecast"]
    timing_info["bootstrap_kf_decomposition"] += bootstrap_timing["bootstrap_kf_decomposition"]
    timing_info["bootstrap_arima_forecast"] += bootstrap_timing["bootstrap_arima_forecast"]
    timing_info["bootstrap_ann_fit_forecast"] += bootstrap_timing["bootstrap_ann_fit_forecast"]

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
    final_noisy_rmse = np.sqrt(np.mean((final_forecast - measurements_test) ** 2))
    final_noisy_mse = np.mean((final_forecast - measurements_test) ** 2)
    calibration_residual_mean = np.mean(calibration_residuals["centered_combined_residuals"])
    calibration_residual_std = np.std(calibration_residuals["centered_combined_residuals"], ddof=1)
    interval_widths = interval_upper - interval_lower
    mean_interval_width = np.mean(interval_widths)
    median_interval_width = np.median(interval_widths)
    clean_in_interval = (true_clean_test >= interval_lower) & (true_clean_test <= interval_upper)
    noisy_in_interval = (measurements_test >= interval_lower) & (measurements_test <= interval_upper)
    clean_coverage = np.mean(clean_in_interval) * 100
    noisy_coverage = np.mean(noisy_in_interval) * 100

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
        "clean_in_interval": clean_in_interval,
        "noisy_in_interval": noisy_in_interval,
        "arima_order": arima_order,
        "calibration_residuals": calibration_residuals["centered_combined_residuals"],
        "calibration_residual_mean": calibration_residual_mean,
        "calibration_residual_std": calibration_residual_std,
        "mean_interval_width": mean_interval_width,
        "median_interval_width": median_interval_width,
        "low_rmse": low_rmse,
        "high_rmse": high_rmse,
        "low_mse": low_mse,
        "high_mse": high_mse,
        "final_clean_rmse": final_clean_rmse,
        "final_noisy_rmse": final_noisy_rmse,
        "final_noisy_mse": final_noisy_mse,
        "clean_coverage": clean_coverage,
        "noisy_coverage": noisy_coverage,
        "timing_info": timing_info,
    }


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    warnings.filterwarnings("ignore", category=UserWarning)

    np.random.seed(42)
    start_time = time.perf_counter()
    print("Mode                          : ARIMA pseudo-series + DistPred-style ANN ensemble")
    n_monte_carlo = 70
    train_size = 300
    horizons = [1]
    bootstrap_replications = 70

    print(f"Monte Carlo runs               : {n_monte_carlo}")
    print(f"Observed sample size T         : {train_size}")
    print(f"Bootstrap replications B       : {bootstrap_replications}")
    print("")
    print("h | Mean low MSE | Mean high MSE | Mean noisy MSE | Mean noisy coverage | Mean interval width")
    print("--|--------------|---------------|----------------|---------------------|--------------------")

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

        low_mses = np.array([result["low_mse"] for result in experiment_results], dtype=float)
        high_mses = np.array([result["high_mse"] for result in experiment_results], dtype=float)
        noisy_mses = np.array([result["final_noisy_mse"] for result in experiment_results], dtype=float)
        noisy_coverages = np.array([result["noisy_coverage"] for result in experiment_results], dtype=float)
        mean_interval_widths = np.array([result["mean_interval_width"] for result in experiment_results], dtype=float)
        mean_data_generation = np.mean([result["timing_info"]["data_generation"] for result in experiment_results])
        mean_arima_order_selection = np.mean([result["timing_info"]["arima_order_selection"] for result in experiment_results])
        mean_insample_residual_fit = np.mean([result["timing_info"]["insample_residual_fit"] for result in experiment_results])
        mean_point_forecast_fit = np.mean([result["timing_info"]["point_forecast_fit"] for result in experiment_results])
        mean_pseudo_series_generation = np.mean([result["timing_info"]["pseudo_series_generation"] for result in experiment_results])
        mean_bootstrap_refit_forecast = np.mean([result["timing_info"]["bootstrap_refit_forecast"] for result in experiment_results])
        mean_bootstrap_kf_decomposition = np.mean([result["timing_info"]["bootstrap_kf_decomposition"] for result in experiment_results])
        mean_bootstrap_arima_forecast = np.mean([result["timing_info"]["bootstrap_arima_forecast"] for result in experiment_results])
        mean_bootstrap_ann_fit_forecast = np.mean([result["timing_info"]["bootstrap_ann_fit_forecast"] for result in experiment_results])

        print(
            f"{horizon} | "
            f"{np.mean(low_mses):.4f}       | "
            f"{np.mean(high_mses):.4f}        | "
            f"{np.mean(noisy_mses):.4f}         | "
            f"{np.mean(noisy_coverages):.2f}%               | "
            f"{np.mean(mean_interval_widths):.4f}"
        )
        print(
            "    Avg timing (s/run): "
            f"data={mean_data_generation:.2f}, "
            f"order={mean_arima_order_selection:.2f}, "
            f"insample_fit={mean_insample_residual_fit:.2f}, "
            f"point_fit={mean_point_forecast_fit:.2f}, "
            f"pseudo_series={mean_pseudo_series_generation:.2f}, "
            f"bootstrap_refit={mean_bootstrap_refit_forecast:.2f}"
        )
        print(
            "    Avg bootstrap breakdown (s/run): "
            f"kf={mean_bootstrap_kf_decomposition:.2f}, "
            f"arima={mean_bootstrap_arima_forecast:.2f}, "
            f"ann={mean_bootstrap_ann_fit_forecast:.2f}"
        )

    elapsed_seconds = time.perf_counter() - start_time
    print("")
    print(f"Elapsed time                   : {elapsed_seconds:.2f} seconds")
