import numpy as np

import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.arima.model import ARIMA


class SimpleKalmanFilter:
    def __init__(self, process_variance=0.5, measurement_variance=10.0, initial_estimate=0.0, initial_error=1.0):
        self.process_variance = float(process_variance)
        self.measurement_variance = float(measurement_variance)
        self.estimated_state = float(initial_estimate)
        self.estimate_error = float(initial_error)

    def update(self, measurement):
        predicted_state = self.estimated_state
        predicted_error = self.estimate_error + self.process_variance

        kalman_gain = predicted_error / (predicted_error + self.measurement_variance)
        self.estimated_state = predicted_state + kalman_gain * (measurement - predicted_state)
        self.estimate_error = (1.0 - kalman_gain) * predicted_error

        return self.estimated_state


def create_windows(series, window_size):
    x_data = []
    y_data = []

    for idx in range(window_size, len(series)):
        x_data.append(series[idx - window_size:idx])
        y_data.append(series[idx])

    return np.array(x_data), np.array(y_data)


def recursive_ann_forecast(model, x_scaler, y_scaler, history, horizon, window_size):
    rolling = list(history[-window_size:])
    forecasts = []

    for _ in range(horizon):
        features = np.array(rolling[-window_size:], dtype=float).reshape(1, -1)
        features_scaled = x_scaler.transform(features)
        pred_scaled = model.predict(features_scaled).reshape(-1, 1)
        pred = y_scaler.inverse_transform(pred_scaled)[0, 0]
        forecasts.append(pred)
        rolling.append(pred)

    return np.array(forecasts)


def bootstrap_ann_forecasts(history, horizon, window_size, n_bootstrap=100):
    boot_predictions = []
    n_samples = len(history)

    for seed in range(n_bootstrap):
        rng = np.random.default_rng(seed)
        sampled_indices = np.sort(rng.choice(n_samples, size=n_samples, replace=True))
        boot_series = history[sampled_indices]
        x_boot, y_boot = create_windows(boot_series, window_size)

        x_scaler = StandardScaler()
        y_scaler = StandardScaler()
        x_boot_scaled = x_scaler.fit_transform(x_boot)
        y_boot_scaled = y_scaler.fit_transform(y_boot.reshape(-1, 1)).ravel()

        ann_model = MLPRegressor(
            hidden_layer_sizes=(128, 64, 32),
            activation="tanh",
            solver="adam",
            alpha=1e-5,
            learning_rate_init=0.0005,
            max_iter=10000,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=50,
            random_state=seed
        )
        ann_model.fit(x_boot_scaled, y_boot_scaled)

        boot_predictions.append(
            recursive_ann_forecast(
                ann_model,
                x_scaler,
                y_scaler,
                boot_series,
                horizon,
                window_size
            )
        )

    return np.array(boot_predictions)


if __name__ == "__main__":
    np.random.seed(42)
    n_bootstrap = 100

    dt = 0.1
    n_steps = 240
    train_size = 200
    horizon = n_steps - train_size
    t = np.arange(n_steps) * dt

    phi = 0.8
    low_true = np.zeros(n_steps, dtype=float)
    for idx in range(1, n_steps):
        low_true[idx] = phi * low_true[idx - 1] + np.random.normal(0, 0.25)

    high_true = 3.0 * np.sin(t)
    noise = np.random.normal(0, 0.2, size=n_steps)
    measurements = low_true + high_true + noise

    kf = SimpleKalmanFilter(
        process_variance=0.5,
        measurement_variance=10.0,
        initial_estimate=measurements[0],
        initial_error=1.0
    )

    estimated_low = []

    for value in measurements:
        low_estimate = kf.update(value)
        estimated_low.append(low_estimate)

    estimated_low = np.array(estimated_low)
    estimated_high = measurements - estimated_low

    low_train = estimated_low[:train_size]
    high_train = estimated_high[:train_size]

    low_test_estimated = estimated_low[train_size:]
    arima_order = (1, 0, 0)
    low_forecast = []
    low_interval_lower = []
    low_interval_upper = []
    for idx in range(horizon):
        rolling_low_train = np.concatenate((low_train, low_test_estimated[:idx]))
        arima_model = ARIMA(rolling_low_train, order=arima_order, trend="c")
        arima_result = arima_model.fit()
        arima_forecast = arima_result.get_forecast(1)
        low_forecast.append(arima_forecast.predicted_mean[0])
        conf_int = arima_forecast.conf_int(alpha=0.05)[0]
        low_interval_lower.append(conf_int[0])
        low_interval_upper.append(conf_int[1])
    low_forecast = np.array(low_forecast)
    low_interval_lower = np.array(low_interval_lower)
    low_interval_upper = np.array(low_interval_upper)

    window_size = 40
    ann_bootstrap_forecasts = bootstrap_ann_forecasts(
        high_train,
        horizon,
        window_size,
        n_bootstrap=n_bootstrap
    )
    high_forecast = ann_bootstrap_forecasts.mean(axis=0)
    high_interval_lower = np.percentile(ann_bootstrap_forecasts, 2.5, axis=0)
    high_interval_upper = np.percentile(ann_bootstrap_forecasts, 97.5, axis=0)

    final_forecast = low_forecast + high_forecast

    t_test = t[train_size:]
    measurements_test = measurements[train_size:]
    true_low_test = low_true[train_size:]
    true_high_test = high_true[train_size:]
    true_clean_test = true_low_test + true_high_test

    low_rmse = np.sqrt(np.mean((low_forecast - true_low_test) ** 2))
    high_rmse = np.sqrt(np.mean((high_forecast - true_high_test) ** 2))
    final_clean_rmse = np.sqrt(np.mean((final_forecast - true_clean_test) ** 2))
    final_noisy_rmse = np.sqrt(np.mean((final_forecast - measurements_test) ** 2))
    final_mse = np.mean((final_forecast - measurements_test) ** 2)
    low_coverage = np.mean(
        (true_low_test >= low_interval_lower) & (true_low_test <= low_interval_upper)
    ) * 100
    high_coverage = np.mean(
        (true_high_test >= high_interval_lower) & (true_high_test <= high_interval_upper)
    ) * 100

    print(f"Low component RMSE  : {low_rmse:.4f}")
    print(f"High component RMSE : {high_rmse:.4f}")
    print(f"Final clean RMSE    : {final_clean_rmse:.4f}")
    print(f"Final noisy RMSE    : {final_noisy_rmse:.4f}")
    print(f"Final noisy MSE     : {final_mse:.4f}")
    print(f"ARIMA low coverage  : {low_coverage:.2f}%")
    print(f"ANN high coverage   : {high_coverage:.2f}%")

    print("\nStep\tTrue\tForecast\tLow\tHigh\tARIMA_L\tARIMA_U\tANN_L\tANN_U")
    for idx in range(horizon):
        print(
            f"{idx + 1}\t"
            f"{measurements_test[idx]:.3f}\t"
            f"{final_forecast[idx]:.3f}\t\t"
            f"{low_forecast[idx]:.3f}\t"
            f"{high_forecast[idx]:.3f}\t"
            f"{low_interval_lower[idx]:.3f}\t\t"
            f"{low_interval_upper[idx]:.3f}\t\t"
            f"{high_interval_lower[idx]:.3f}\t"
            f"{high_interval_upper[idx]:.3f}"
        )

    fig, axes = plt.subplots(4, 1, figsize=(12, 13), constrained_layout=True)

    axes[0].plot(t, measurements, label="Mixed Data", alpha=0.7)
    axes[0].plot(t, low_true, label="True AR(1) Component", linewidth=2)
    axes[0].plot(t, estimated_low, label="KF Estimated Low Part", linewidth=2)
    axes[0].axvline(t[train_size], color="gray", linestyle="--", label="Forecast Start")
    axes[0].set_ylabel("Signal")
    axes[0].legend(loc="upper left")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t, high_true, label="True 3*sin(x)", linewidth=2)
    axes[1].plot(t, estimated_high, label="KF Estimated High Part", alpha=0.9)
    axes[1].axvline(t[train_size], color="gray", linestyle="--")
    axes[1].set_ylabel("Residual")
    axes[1].legend(loc="upper left")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(t_test, true_low_test, label="True Low Component", linewidth=2)
    axes[2].plot(t_test, low_forecast, label="ARIMA Forecast", linewidth=2)
    axes[2].plot(t_test, true_high_test, label="True High Component", linewidth=2)
    axes[2].plot(t_test, high_forecast, label="ANN Forecast", linewidth=2)
    axes[2].set_ylabel("Components")
    axes[2].legend(loc="center right")
    axes[2].grid(True, alpha=0.3)

    axes[3].plot(t_test, true_clean_test, label="True Future Clean Signal", linewidth=2)
    axes[3].plot(t_test, measurements_test, label="True Future Mixed Data", linewidth=2)
    axes[3].plot(t_test, final_forecast, label="Combined Point Forecast", linewidth=2)
    axes[3].fill_between(
        t_test,
        high_interval_lower,
        high_interval_upper,
        color="orange",
        alpha=0.25,
        label="ANN Bootstrap Band"
    )
    axes[3].set_ylabel("Forecast")
    axes[3].set_xlabel("Time (s)")
    axes[3].legend(loc="upper left")
    axes[3].grid(True, alpha=0.3)

    plt.show()
