import matplotlib
import numpy as np
import time
import warnings

matplotlib.use("Agg")
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tools.sm_exceptions import ConvergenceWarning
import torch
import torch.nn as nn


def kalman_filter(observations, process_variance=0.5, measurement_variance=10.0, initial_error=1.0):
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
        initial_error=1.0,
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
                    enforce_invertibility=False,
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


def fit_arima_component_with_samples(low_train, horizon, arima_order, ensemble_size=50):
    model = ARIMA(
        np.asarray(low_train, dtype=float),
        order=arima_order,
        trend="c",
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        warnings.simplefilter("ignore", UserWarning)
        result = model.fit()

    forecast_res = result.get_forecast(horizon)
    low_forecast = np.asarray(forecast_res.predicted_mean, dtype=float)
    low_var = np.asarray(forecast_res.var_pred_mean, dtype=float)
    low_std = np.sqrt(np.maximum(low_var, 1e-8))

    low_samples = np.random.normal(
        loc=low_forecast.reshape(1, -1),
        scale=low_std.reshape(1, -1),
        size=(ensemble_size, horizon),
    )

    return low_forecast, low_samples, result


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


def fit_distpred_bundle(series_train, window_size, ensemble_size=50, epochs=250, learning_rate=1e-3):
    x_train, y_train = create_windows(series_train, window_size)
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

    lower_q, upper_q = np.quantile(series_train, [0.01, 0.99])
    iqr = np.subtract(*np.quantile(series_train, [0.75, 0.25]))
    margin = max(0.5 * iqr, 0.25)
    clip_bounds = (float(lower_q - margin), float(upper_q + margin))

    return {
        "model": model.eval(),
        "x_scaler": x_scaler,
        "y_scaler": y_scaler,
        "clip_bounds": clip_bounds,
        "x_train_scaled": x_train_scaled,
        "window_size": window_size,
        "ensemble_size": ensemble_size,
    }


def recursive_distpred_forecast(bundle, history, horizon, window_size):
    ensemble_size = bundle["ensemble_size"]
    rolling_histories = np.tile(np.asarray(history[-window_size:], dtype=float), (ensemble_size, 1))
    ensemble_paths = np.zeros((ensemble_size, horizon), dtype=float)

    for step_idx in range(horizon):
        features_scaled = bundle["x_scaler"].transform(rolling_histories)
        with torch.no_grad():
            pred_scaled = bundle["model"](torch.tensor(features_scaled, dtype=torch.float32)).cpu().numpy()
        pred = bundle["y_scaler"].inverse_transform(pred_scaled.reshape(-1, 1)).reshape(pred_scaled.shape)
        next_vals = pred[np.arange(ensemble_size), np.arange(ensemble_size)]
        next_vals = np.clip(next_vals, bundle["clip_bounds"][0], bundle["clip_bounds"][1])
        ensemble_paths[:, step_idx] = next_vals
        rolling_histories = np.concatenate([rolling_histories[:, 1:], next_vals.reshape(-1, 1)], axis=1)

    return ensemble_paths


def fit_distpred_component(series_train, horizon, window_size, ensemble_size=50):
    bundle = fit_distpred_bundle(series_train, window_size, ensemble_size=ensemble_size)
    samples = recursive_distpred_forecast(bundle, series_train, horizon, window_size)
    forecast = np.mean(samples, axis=0)
    return forecast, samples, bundle


def hybrid_point_forecast(train_series, horizon, window_size=15, ensemble_size=50):
    estimated_low, estimated_high = decompose_with_simple_kf(train_series)
    arima_order = find_arima_order(estimated_low, p_max=3, q_max=3)
    low_forecast, low_samples, low_result = fit_arima_component_with_samples(
        estimated_low,
        horizon,
        arima_order=arima_order,
        ensemble_size=ensemble_size,
    )
    high_forecast, high_samples, high_bundle = fit_distpred_component(
        estimated_high,
        horizon,
        window_size=window_size,
        ensemble_size=ensemble_size,
    )
    final_samples = low_samples + high_samples
    final_forecast = np.mean(final_samples, axis=0)

    return {
        "arima_order": arima_order,
        "estimated_low": estimated_low,
        "estimated_high": estimated_high,
        "low_forecast": low_forecast,
        "low_samples": low_samples,
        "high_forecast": high_forecast,
        "high_samples": high_samples,
        "high_bundle": high_bundle,
        "low_result": low_result,
        "final_samples": final_samples,
        "final_forecast": final_forecast,
    }


def timed_hybrid_point_forecast(train_series, horizon, window_size=15, ensemble_size=50):
    timing = {
        "kf_decomposition": 0.0,
        "order_selection": 0.0,
        "low_arima_fit_forecast": 0.0,
        "high_distpred_fit_forecast": 0.0,
    }

    kf_start = time.perf_counter()
    estimated_low, estimated_high = decompose_with_simple_kf(train_series)
    timing["kf_decomposition"] += time.perf_counter() - kf_start

    order_start = time.perf_counter()
    arima_order = find_arima_order(estimated_low, p_max=3, q_max=3)
    timing["order_selection"] += time.perf_counter() - order_start

    low_start = time.perf_counter()
    low_forecast, low_samples, low_result = fit_arima_component_with_samples(
        estimated_low,
        horizon,
        arima_order=arima_order,
        ensemble_size=ensemble_size,
    )
    timing["low_arima_fit_forecast"] += time.perf_counter() - low_start

    high_start = time.perf_counter()
    high_forecast, high_samples, high_bundle = fit_distpred_component(
        estimated_high,
        horizon,
        window_size=window_size,
        ensemble_size=ensemble_size,
    )
    timing["high_distpred_fit_forecast"] += time.perf_counter() - high_start

    final_samples = low_samples + high_samples
    final_forecast = np.mean(final_samples, axis=0)

    return {
        "arima_order": arima_order,
        "estimated_low": estimated_low,
        "estimated_high": estimated_high,
        "low_forecast": low_forecast,
        "low_samples": low_samples,
        "high_forecast": high_forecast,
        "high_samples": high_samples,
        "high_bundle": high_bundle,
        "low_result": low_result,
        "final_samples": final_samples,
        "final_forecast": final_forecast,
    }, timing


def simulate_babu_style_data(n_steps, noise_std=0.15):
    low_true = np.zeros(n_steps, dtype=float)
    high_true = np.zeros(n_steps, dtype=float)
    low_true[0] = np.random.normal(0, 0.2)
    high_true[0] = np.random.normal(0, 0.2)

    for idx in range(1, n_steps):
        low_true[idx] = 0.6 * low_true[idx - 1] + np.random.normal(0, 0.2)
        high_true[idx] = (
            0.8 * high_true[idx - 1]
            - 0.8 * high_true[idx - 1] / (1.0 + np.exp(-10.0 * high_true[idx - 1]))
            + np.random.normal(0, 0.2)
        )

    noise = np.random.normal(0, noise_std, size=n_steps)
    measurements = low_true + high_true + noise
    return low_true, high_true, measurements


def run_single_experiment(train_size=300, horizon=1, window_size=15, ensemble_size=50):
    timing_info = {
        "data_generation": 0.0,
        "point_forecast_fit": 0.0,
        "kf_decomposition": 0.0,
        "order_selection": 0.0,
        "low_arima_fit_forecast": 0.0,
        "high_distpred_fit_forecast": 0.0,
    }

    n_steps = train_size + horizon

    data_start = time.perf_counter()
    low_true, high_true, measurements = simulate_babu_style_data(n_steps)
    timing_info["data_generation"] += time.perf_counter() - data_start

    point_start = time.perf_counter()
    full_train_bundle, point_timing = timed_hybrid_point_forecast(
        measurements[:train_size],
        horizon,
        window_size=window_size,
        ensemble_size=ensemble_size,
    )
    timing_info["point_forecast_fit"] += time.perf_counter() - point_start
    timing_info["kf_decomposition"] += point_timing["kf_decomposition"]
    timing_info["order_selection"] += point_timing["order_selection"]
    timing_info["low_arima_fit_forecast"] += point_timing["low_arima_fit_forecast"]
    timing_info["high_distpred_fit_forecast"] += point_timing["high_distpred_fit_forecast"]

    low_forecast = full_train_bundle["low_forecast"]
    high_forecast = full_train_bundle["high_forecast"]
    final_forecast = full_train_bundle["final_forecast"]
    final_samples = full_train_bundle["final_samples"]

    interval_lower = np.quantile(final_samples, 0.025, axis=0)
    interval_upper = np.quantile(final_samples, 0.975, axis=0)

    measurements_test = measurements[train_size:]
    true_low_test = low_true[train_size:]
    true_high_test = high_true[train_size:]

    low_mse = np.mean((low_forecast - true_low_test) ** 2)
    high_mse = np.mean((high_forecast - true_high_test) ** 2)
    final_noisy_mse = np.mean((final_forecast - measurements_test) ** 2)
    noisy_in_interval = (measurements_test >= interval_lower) & (measurements_test <= interval_upper)
    noisy_coverage = np.mean(noisy_in_interval) * 100
    mean_interval_width = np.mean(interval_upper - interval_lower)

    return {
        "low_mse": low_mse,
        "high_mse": high_mse,
        "final_noisy_mse": final_noisy_mse,
        "noisy_coverage": noisy_coverage,
        "mean_interval_width": mean_interval_width,
        "timing_info": timing_info,
        "arima_order": full_train_bundle["arima_order"],
    }


if __name__ == "__main__":
    np.random.seed(42)
    start_time = time.perf_counter()
    print("Mode                          : ARIMA predictive ensemble + DistPred-style ANN ensemble")
    n_monte_carlo = 50
    train_size = 300
    horizons = [1]
    ensemble_size = 50

    print(f"Monte Carlo runs               : {n_monte_carlo}")
    print(f"Observed sample size T         : {train_size}")
    print(f"Ensemble samples K             : {ensemble_size}")
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
                    ensemble_size=ensemble_size,
                )
            )

        low_mses = np.array([result["low_mse"] for result in experiment_results], dtype=float)
        high_mses = np.array([result["high_mse"] for result in experiment_results], dtype=float)
        noisy_mses = np.array([result["final_noisy_mse"] for result in experiment_results], dtype=float)
        noisy_coverages = np.array([result["noisy_coverage"] for result in experiment_results], dtype=float)
        mean_interval_widths = np.array([result["mean_interval_width"] for result in experiment_results], dtype=float)
        mean_data_generation = np.mean([result["timing_info"]["data_generation"] for result in experiment_results])
        mean_point_forecast_fit = np.mean([result["timing_info"]["point_forecast_fit"] for result in experiment_results])
        mean_kf_decomposition = np.mean([result["timing_info"]["kf_decomposition"] for result in experiment_results])
        mean_order_selection = np.mean([result["timing_info"]["order_selection"] for result in experiment_results])
        mean_low_arima_fit_forecast = np.mean([result["timing_info"]["low_arima_fit_forecast"] for result in experiment_results])
        mean_high_distpred_fit_forecast = np.mean([result["timing_info"]["high_distpred_fit_forecast"] for result in experiment_results])
        mean_order = np.mean([result["arima_order"][0] + result["arima_order"][2] for result in experiment_results])

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
            f"point_fit={mean_point_forecast_fit:.2f}"
        )
        print(
            "    Avg model breakdown (s/run): "
            f"kf={mean_kf_decomposition:.2f}, "
            f"order={mean_order_selection:.2f}, "
            f"arima={mean_low_arima_fit_forecast:.2f}, "
            f"high={mean_high_distpred_fit_forecast:.2f}, "
            f"pq_sum={mean_order:.2f}"
        )

    elapsed_seconds = time.perf_counter() - start_time
    print("")
    print(f"Elapsed time                   : {elapsed_seconds:.2f} seconds")
