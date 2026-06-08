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


def recursive_ann_forecast(model, x_scaler, y_scaler, history, horizon, window_size, clip_bounds=None):
    rolling = list(history[-window_size:])
    forecasts = []

    for _ in range(horizon):
        features = np.array(rolling[-window_size:], dtype=float).reshape(1, -1)
        features_scaled = x_scaler.transform(features)
        pred_scaled = model.predict(features_scaled).reshape(-1, 1)
        pred = y_scaler.inverse_transform(pred_scaled)[0, 0]
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


def fit_ann_high_component(high_train, horizon, window_size):
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

    high_forecast = recursive_ann_forecast(
        ann_model,
        x_scaler,
        y_scaler,
        high_train,
        horizon,
        window_size,
        clip_bounds=clip_bounds
    )

    return high_forecast


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


def rolling_calibration_residuals(series, initial_train_size, calibration_end, block_size, arima_order, window_size=30):
    residuals = []

    for start in range(initial_train_size, calibration_end, block_size):
        stop = min(start + block_size, calibration_end)
        horizon = stop - start
        train_series = series[:start]
        actual_block = series[start:stop]

        forecast_bundle = hybrid_point_forecast(
            train_series,
            horizon,
            arima_order=arima_order,
            window_size=window_size
        )
        block_residuals = actual_block - forecast_bundle["final_forecast"]
        residuals.extend(block_residuals.tolist())

    return np.array(residuals, dtype=float)


def empirical_prediction_interval(point_forecast, calibration_residuals, alpha=0.05):
    lower_resid = np.quantile(calibration_residuals, alpha / 2)
    upper_resid = np.quantile(calibration_residuals, 1 - alpha / 2)
    interval_lower = point_forecast + lower_resid
    interval_upper = point_forecast + upper_resid
    return interval_lower, interval_upper


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


def run_single_experiment(n_steps=300, train_size=240, window_size=15, initial_train_size=120, calibration_block=30):
    dt = 1.0
    horizon = n_steps - train_size
    t = np.arange(n_steps) * dt

    low_true, high_true, measurements = simulate_babu_style_data(n_steps)

    calibration_low, _ = decompose_with_simple_kf(measurements[:train_size])
    arima_order = find_arima_order(calibration_low, p_max=3, q_max=3)

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

    interval_lower, interval_upper = empirical_prediction_interval(
        final_forecast,
        calibration_residuals,
        alpha=0.05
    )

    t_test = t[train_size:]
    measurements_test = measurements[train_size:]
    true_low_test = low_true[train_size:]
    true_high_test = high_true[train_size:]
    true_clean_test = true_low_test + true_high_test

    low_rmse = np.sqrt(np.mean((low_forecast - true_low_test) ** 2))
    high_rmse = np.sqrt(np.mean((high_forecast - true_high_test) ** 2))
    final_clean_rmse = np.sqrt(np.mean((final_forecast - true_clean_test) ** 2))
    final_noisy_rmse = np.sqrt(np.mean((final_forecast - measurements_test) ** 2))
    calibration_residual_mean = np.mean(calibration_residuals)
    calibration_residual_std = np.std(calibration_residuals, ddof=1)
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
        "calibration_residuals": calibration_residuals,
        "calibration_residual_mean": calibration_residual_mean,
        "calibration_residual_std": calibration_residual_std,
        "low_rmse": low_rmse,
        "high_rmse": high_rmse,
        "final_clean_rmse": final_clean_rmse,
        "final_noisy_rmse": final_noisy_rmse,
        "clean_coverage": clean_coverage,
        "noisy_coverage": noisy_coverage,
    }


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    warnings.filterwarnings("ignore", category=UserWarning)

    np.random.seed(42)
    start_time = time.perf_counter()
    n_monte_carlo = 100
    single_run_id_raw = os.getenv("SINGLE_RUN_ID")
    experiment_results = []

    if single_run_id_raw is not None:
        single_run_id = int(single_run_id_raw)
        if not 1 <= single_run_id <= n_monte_carlo:
            raise ValueError(f"SINGLE_RUN_ID must be between 1 and {n_monte_carlo}.")
        np.random.seed(42 + (single_run_id - 1))
        experiment_results.append(run_single_experiment())
        representative_run_id = single_run_id
    else:
        for run_idx in range(n_monte_carlo):
            np.random.seed(42 + run_idx)
            experiment_results.append(run_single_experiment())
        representative_run_id = 1

    representative = experiment_results[0]

    clean_coverages = np.array([result["clean_coverage"] for result in experiment_results], dtype=float)
    noisy_coverages = np.array([result["noisy_coverage"] for result in experiment_results], dtype=float)
    low_rmses = np.array([result["low_rmse"] for result in experiment_results], dtype=float)
    high_rmses = np.array([result["high_rmse"] for result in experiment_results], dtype=float)
    clean_rmses = np.array([result["final_clean_rmse"] for result in experiment_results], dtype=float)
    noisy_rmses = np.array([result["final_noisy_rmse"] for result in experiment_results], dtype=float)
    worst_high_idx = int(np.argmax(high_rmses))
    worst_clean_idx = int(np.argmax(clean_rmses))
    worst_noisy_idx = int(np.argmax(noisy_rmses))
    elapsed_seconds = time.perf_counter() - start_time

    completed_runs = len(experiment_results)
    print(f"Monte Carlo runs               : {completed_runs}")
    print(f"Representative ARIMA order     : {representative['arima_order']}")
    print(f"Representative run id          : {representative_run_id}")
    print(f"Mean clean coverage            : {np.mean(clean_coverages):.2f}%")
    print(f"Median clean coverage          : {np.median(clean_coverages):.2f}%")
    print(f"Mean noisy coverage            : {np.mean(noisy_coverages):.2f}%")
    print(f"Median noisy coverage          : {np.median(noisy_coverages):.2f}%")
    print(f"Mean low component RMSE        : {np.mean(low_rmses):.4f}")
    print(f"Median low component RMSE      : {np.median(low_rmses):.4f}")
    print(f"Max low component RMSE         : {np.max(low_rmses):.4f}")
    print(f"Mean high component RMSE       : {np.mean(high_rmses):.4f}")
    print(f"Median high component RMSE     : {np.median(high_rmses):.4f}")
    print(f"Max high component RMSE        : {np.max(high_rmses):.4f} (run {worst_high_idx + 1})")
    print(f"Mean final clean RMSE          : {np.mean(clean_rmses):.4f}")
    print(f"Median final clean RMSE        : {np.median(clean_rmses):.4f}")
    print(f"Max final clean RMSE           : {np.max(clean_rmses):.4f} (run {worst_clean_idx + 1})")
    print(f"Mean final noisy RMSE          : {np.mean(noisy_rmses):.4f}")
    print(f"Median final noisy RMSE        : {np.median(noisy_rmses):.4f}")
    print(f"Max final noisy RMSE           : {np.max(noisy_rmses):.4f} (run {worst_noisy_idx + 1})")
    print(f"Representative clean coverage  : {representative['clean_coverage']:.2f}%")
    print(f"Representative noisy coverage  : {representative['noisy_coverage']:.2f}%")
    print(f"Elapsed time                   : {elapsed_seconds:.2f} seconds")

    fig, axes = plt.subplots(4, 1, figsize=(12, 13), constrained_layout=True)

    axes[0].plot(representative["t"], representative["measurements"], label="Mixed Data", alpha=0.7)
    axes[0].plot(representative["t"], representative["low_true"], label="True AR(1) Component", linewidth=2)
    axes[0].plot(representative["t"], representative["estimated_low_full"], label="KF Estimated Low Part", linewidth=2)
    axes[0].axvline(representative["t_test"][0], color="gray", linestyle="--", label="Forecast Start")
    axes[0].set_ylabel("Signal")
    axes[0].legend(loc="upper left")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(representative["t"], representative["high_true"], label="True Nonlinear Part", linewidth=2)
    axes[1].plot(representative["t"], representative["estimated_high_full"], label="KF Estimated High Part", alpha=0.9)
    axes[1].axvline(representative["t_test"][0], color="gray", linestyle="--")
    axes[1].set_ylabel("Residual")
    axes[1].legend(loc="upper left")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(representative["t_test"], representative["true_low_test"], label="True Low Component", linewidth=2)
    axes[2].plot(representative["t_test"], representative["low_forecast"], label="ARIMA Forecast", linewidth=2)
    axes[2].plot(representative["t_test"], representative["true_high_test"], label="True High Component", linewidth=2)
    axes[2].plot(representative["t_test"], representative["high_forecast"], label="ANN Forecast", linewidth=2)
    axes[2].set_ylabel("Components")
    axes[2].legend(loc="center right")
    axes[2].grid(True, alpha=0.3)

    axes[3].plot(representative["t_test"], representative["true_clean_test"], label="True Future Clean Signal", linewidth=2)
    axes[3].plot(representative["t_test"], representative["measurements_test"], label="True Future Mixed Data", linewidth=2)
    axes[3].plot(representative["t_test"], representative["final_forecast"], label="Combined Point Forecast", linewidth=2)
    axes[3].fill_between(
        representative["t_test"],
        representative["interval_lower"],
        representative["interval_upper"],
        color="gray",
        alpha=0.25,
        label="Empirical Rolling PI"
    )
    axes[3].set_ylabel("Forecast")
    axes[3].set_xlabel("Time (s)")
    axes[3].legend(loc="upper left")
    axes[3].grid(True, alpha=0.3)

    plot_filename = (
        f"kf_giordano_m1_m9_run_{representative_run_id}_forecast.png"
        if single_run_id_raw is not None
        else "kf_giordano_m1_m9_forecast.png"
    )
    plt.savefig(plot_filename, dpi=150)
    print(f"\nPlot saved to {plot_filename}")
