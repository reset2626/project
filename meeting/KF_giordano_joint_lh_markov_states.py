import matplotlib
import numpy as np
import time

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
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


def create_joint_windows(low_series, high_series, window_size):
    features = []
    targets = []
    for idx in range(window_size, len(low_series)):
        low_window = np.asarray(low_series[idx - window_size:idx], dtype=float)
        high_window = np.asarray(high_series[idx - window_size:idx], dtype=float)
        joint_window = np.column_stack([low_window, high_window]).reshape(-1)
        features.append(joint_window)
        targets.append([float(low_series[idx]), float(high_series[idx])])
    return np.asarray(features, dtype=float), np.asarray(targets, dtype=float)


class JointDistPredMLP(nn.Module):
    def __init__(self, input_dim, ensemble_size):
        super().__init__()
        self.ensemble_size = ensemble_size
        self.net = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, ensemble_size * 2),
        )

    def forward(self, x):
        out = self.net(x)
        return out.view(-1, self.ensemble_size, 2)


def joint_distpred_loss(pred_ensemble, y_true):
    y_true = y_true.unsqueeze(1)
    calibration_term = torch.abs(pred_ensemble - y_true).mean(dim=(1, 2))
    pairwise = pred_ensemble.unsqueeze(2) - pred_ensemble.unsqueeze(1)
    diversity_term = torch.abs(pairwise).mean(dim=(1, 2, 3))
    mean_term = torch.square(pred_ensemble.mean(dim=1) - y_true.squeeze(1)).mean(dim=1)
    # Encourage sample spread more strongly so the ensemble does not collapse
    # into a narrow point-mass around the conditional mean.
    return (calibration_term - 0.5 * diversity_term + 0.05 * mean_term).mean()


def fit_joint_distpred_bundle(low_train, high_train, window_size, ensemble_size=50, epochs=250, learning_rate=1e-3):
    x_train, y_train = create_joint_windows(low_train, high_train, window_size)
    if len(x_train) == 0:
        raise ValueError("Not enough samples for joint DistPred windows. Increase training size or reduce window_size.")

    x_scaler = StandardScaler()
    y_scaler = StandardScaler()
    x_train_scaled = x_scaler.fit_transform(x_train)
    y_train_scaled = y_scaler.fit_transform(y_train).astype(np.float32)

    x_tensor = torch.tensor(x_train_scaled, dtype=torch.float32)
    y_tensor = torch.tensor(y_train_scaled, dtype=torch.float32)

    model = JointDistPredMLP(input_dim=x_train.shape[1], ensemble_size=ensemble_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        pred_ensemble = model(x_tensor)
        loss = joint_distpred_loss(pred_ensemble, y_tensor)
        loss.backward()
        optimizer.step()

    low_q = np.quantile(low_train, [0.01, 0.99])
    high_q = np.quantile(high_train, [0.01, 0.99])
    low_iqr = np.subtract(*np.quantile(low_train, [0.75, 0.25]))
    high_iqr = np.subtract(*np.quantile(high_train, [0.75, 0.25]))
    low_margin = max(0.5 * low_iqr, 0.25)
    high_margin = max(0.5 * high_iqr, 0.25)
    clip_bounds = (
        (float(low_q[0] - low_margin), float(low_q[1] + low_margin)),
        (float(high_q[0] - high_margin), float(high_q[1] + high_margin)),
    )

    return {
        "model": model.eval(),
        "x_scaler": x_scaler,
        "y_scaler": y_scaler,
        "clip_bounds": clip_bounds,
        "window_size": window_size,
        "ensemble_size": ensemble_size,
    }


def recursive_joint_distpred_forecast(bundle, low_history, high_history, horizon, window_size):
    ensemble_size = bundle["ensemble_size"]
    low_hist = np.asarray(low_history[-window_size:], dtype=float)
    high_hist = np.asarray(high_history[-window_size:], dtype=float)
    base_history = np.column_stack([low_hist, high_hist])
    rolling_histories = np.tile(base_history.reshape(1, window_size, 2), (ensemble_size, 1, 1))
    low_paths = np.zeros((ensemble_size, horizon), dtype=float)
    high_paths = np.zeros((ensemble_size, horizon), dtype=float)

    for step_idx in range(horizon):
        features = rolling_histories.reshape(ensemble_size, window_size * 2)
        features_scaled = bundle["x_scaler"].transform(features)
        with torch.no_grad():
            pred_scaled = bundle["model"](torch.tensor(features_scaled, dtype=torch.float32)).cpu().numpy()

        # Pick the diagonal ensemble member to keep one path per sample.
        diag_scaled = pred_scaled[np.arange(ensemble_size), np.arange(ensemble_size), :]
        diag_pred = bundle["y_scaler"].inverse_transform(diag_scaled)
        next_low = np.clip(diag_pred[:, 0], bundle["clip_bounds"][0][0], bundle["clip_bounds"][0][1])
        next_high = np.clip(diag_pred[:, 1], bundle["clip_bounds"][1][0], bundle["clip_bounds"][1][1])

        low_paths[:, step_idx] = next_low
        high_paths[:, step_idx] = next_high

        next_pair = np.stack([next_low, next_high], axis=1).reshape(ensemble_size, 1, 2)
        rolling_histories = np.concatenate([rolling_histories[:, 1:, :], next_pair], axis=1)

    return low_paths, high_paths


def fit_joint_distpred_component(low_train, high_train, horizon, window_size, ensemble_size=50):
    bundle = fit_joint_distpred_bundle(
        low_train,
        high_train,
        window_size,
        ensemble_size=ensemble_size,
    )
    low_samples, high_samples = recursive_joint_distpred_forecast(
        bundle,
        low_train,
        high_train,
        horizon,
        window_size,
    )
    low_forecast = np.mean(low_samples, axis=0)
    high_forecast = np.mean(high_samples, axis=0)
    return low_forecast, high_forecast, low_samples, high_samples, bundle


def hybrid_point_forecast(train_series, horizon, window_size=15, ensemble_size=50):
    estimated_low, estimated_high = decompose_with_simple_kf(train_series)
    low_forecast, high_forecast, low_samples, high_samples, bundle = fit_joint_distpred_component(
        estimated_low,
        estimated_high,
        horizon,
        window_size=window_size,
        ensemble_size=ensemble_size,
    )
    final_samples = low_samples + high_samples
    final_forecast = np.mean(final_samples, axis=0)

    return {
        "estimated_low": estimated_low,
        "estimated_high": estimated_high,
        "low_forecast": low_forecast,
        "high_forecast": high_forecast,
        "low_samples": low_samples,
        "high_samples": high_samples,
        "joint_bundle": bundle,
        "final_samples": final_samples,
        "final_forecast": final_forecast,
    }


def timed_hybrid_point_forecast(train_series, horizon, window_size=15, ensemble_size=50):
    timing = {
        "kf_decomposition": 0.0,
        "joint_distpred_fit_forecast": 0.0,
    }

    kf_start = time.perf_counter()
    estimated_low, estimated_high = decompose_with_simple_kf(train_series)
    timing["kf_decomposition"] += time.perf_counter() - kf_start

    joint_start = time.perf_counter()
    low_forecast, high_forecast, low_samples, high_samples, bundle = fit_joint_distpred_component(
        estimated_low,
        estimated_high,
        horizon,
        window_size=window_size,
        ensemble_size=ensemble_size,
    )
    timing["joint_distpred_fit_forecast"] += time.perf_counter() - joint_start

    final_samples = low_samples + high_samples
    final_forecast = np.mean(final_samples, axis=0)

    return {
        "estimated_low": estimated_low,
        "estimated_high": estimated_high,
        "low_forecast": low_forecast,
        "high_forecast": high_forecast,
        "low_samples": low_samples,
        "high_samples": high_samples,
        "joint_bundle": bundle,
        "final_samples": final_samples,
        "final_forecast": final_forecast,
    }, timing


def save_interaction_plot(bundle, low_history, high_history, output_path, grid_points=41):
    window_size = bundle["window_size"]
    ensemble_size = bundle["ensemble_size"]
    base_low = np.asarray(low_history[-window_size:], dtype=float)
    base_high = np.asarray(high_history[-window_size:], dtype=float)

    low_min, low_max = bundle["clip_bounds"][0]
    high_min, high_max = bundle["clip_bounds"][1]
    low_grid = np.linspace(low_min, low_max, grid_points)
    high_grid = np.linspace(high_min, high_max, grid_points)

    mean_surface = np.zeros((grid_points, grid_points), dtype=float)
    width_surface = np.zeros((grid_points, grid_points), dtype=float)

    for i, low_val in enumerate(low_grid):
        for j, high_val in enumerate(high_grid):
            low_window = base_low.copy()
            high_window = base_high.copy()
            low_window[-1] = low_val
            high_window[-1] = high_val

            features = np.column_stack([low_window, high_window]).reshape(1, -1)
            features = np.repeat(features, ensemble_size, axis=0)
            features_scaled = bundle["x_scaler"].transform(features)
            with torch.no_grad():
                pred_scaled = bundle["model"](torch.tensor(features_scaled, dtype=torch.float32)).cpu().numpy()

            diag_scaled = pred_scaled[np.arange(ensemble_size), np.arange(ensemble_size), :]
            diag_pred = bundle["y_scaler"].inverse_transform(diag_scaled)
            next_low = np.clip(diag_pred[:, 0], bundle["clip_bounds"][0][0], bundle["clip_bounds"][0][1])
            next_high = np.clip(diag_pred[:, 1], bundle["clip_bounds"][1][0], bundle["clip_bounds"][1][1])
            final_samples = next_low + next_high

            mean_surface[j, i] = np.mean(final_samples)
            width_surface[j, i] = np.quantile(final_samples, 0.975) - np.quantile(final_samples, 0.025)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    mean_im = axes[0].imshow(
        mean_surface,
        origin="lower",
        aspect="auto",
        extent=[low_grid[0], low_grid[-1], high_grid[0], high_grid[-1]],
        cmap="viridis",
    )
    axes[0].set_title("Final Mean vs Last (L,H)")
    axes[0].set_xlabel("Last L")
    axes[0].set_ylabel("Last H")
    fig.colorbar(mean_im, ax=axes[0], shrink=0.85)

    width_im = axes[1].imshow(
        width_surface,
        origin="lower",
        aspect="auto",
        extent=[low_grid[0], low_grid[-1], high_grid[0], high_grid[-1]],
        cmap="magma",
    )
    axes[1].set_title("Final 95% Width vs Last (L,H)")
    axes[1].set_xlabel("Last L")
    axes[1].set_ylabel("Last H")
    fig.colorbar(width_im, ax=axes[1], shrink=0.85)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def simulate_babu_style_data(n_steps, noise_std=0.15, p11=0.9, p22=0.9):
    low_true = np.zeros(n_steps, dtype=float)
    high_true = np.zeros(n_steps, dtype=float)
    states = np.zeros(n_steps, dtype=int)
    low_true[0] = np.random.normal(0, 0.2)
    high_true[0] = np.random.normal(0, 0.2)
    states[0] = 0

    for idx in range(1, n_steps):
        prev_state = states[idx - 1]
        if prev_state == 0:
            states[idx] = 0 if np.random.rand() < p11 else 1
        else:
            states[idx] = 1 if np.random.rand() < p22 else 0

        if states[idx] == 0:
            low_phi = 0.6
            high_coef = 0.8
        else:
            low_phi = 0.8
            high_coef = 0.6

        shared_shock = np.random.normal(0, 0.2)
        low_true[idx] = low_phi * low_true[idx - 1] + shared_shock
        high_true[idx] = (
            high_coef * high_true[idx - 1]
            - high_coef * high_true[idx - 1] / (1.0 + np.exp(-10.0 * high_true[idx - 1]))
            + shared_shock
        )

    noise = np.random.normal(0, noise_std, size=n_steps)
    measurements = low_true + high_true + noise
    return low_true, high_true, measurements, states


def run_single_experiment(train_size=300, horizon=1, window_size=15, ensemble_size=50):
    timing_info = {
        "data_generation": 0.0,
        "point_forecast_fit": 0.0,
        "kf_decomposition": 0.0,
        "joint_distpred_fit_forecast": 0.0,
    }

    n_steps = train_size + horizon

    data_start = time.perf_counter()
    low_true, high_true, measurements, states = simulate_babu_style_data(n_steps)
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
    timing_info["joint_distpred_fit_forecast"] += point_timing["joint_distpred_fit_forecast"]

    low_forecast = full_train_bundle["low_forecast"]
    high_forecast = full_train_bundle["high_forecast"]
    final_forecast = full_train_bundle["final_forecast"]
    low_samples = full_train_bundle["low_samples"]
    high_samples = full_train_bundle["high_samples"]
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
    low_sample_width = np.mean(np.quantile(low_samples, 0.975, axis=0) - np.quantile(low_samples, 0.025, axis=0))
    high_sample_width = np.mean(np.quantile(high_samples, 0.975, axis=0) - np.quantile(high_samples, 0.025, axis=0))
    final_sample_width = np.mean(np.quantile(final_samples, 0.975, axis=0) - np.quantile(final_samples, 0.025, axis=0))

    return {
        "low_mse": low_mse,
        "high_mse": high_mse,
        "final_noisy_mse": final_noisy_mse,
        "noisy_coverage": noisy_coverage,
        "mean_interval_width": mean_interval_width,
        "low_sample_width": low_sample_width,
        "high_sample_width": high_sample_width,
        "final_sample_width": final_sample_width,
        "timing_info": timing_info,
        "joint_bundle": full_train_bundle["joint_bundle"],
        "estimated_low_train": full_train_bundle["estimated_low"],
        "estimated_high_train": full_train_bundle["estimated_high"],
        "states": states,
    }


if __name__ == "__main__":
    np.random.seed(42)
    start_time = time.perf_counter()
    print("Mode                          : Joint (L,H) DistPred-style ensemble with 2-state Markov switching")
    n_monte_carlo = 50
    train_size = 300
    horizons = [1 , 3]
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
        low_sample_widths = np.array([result["low_sample_width"] for result in experiment_results], dtype=float)
        high_sample_widths = np.array([result["high_sample_width"] for result in experiment_results], dtype=float)
        final_sample_widths = np.array([result["final_sample_width"] for result in experiment_results], dtype=float)
        mean_data_generation = np.mean([result["timing_info"]["data_generation"] for result in experiment_results])
        mean_point_forecast_fit = np.mean([result["timing_info"]["point_forecast_fit"] for result in experiment_results])
        mean_kf_decomposition = np.mean([result["timing_info"]["kf_decomposition"] for result in experiment_results])
        mean_joint_distpred_fit_forecast = np.mean(
            [result["timing_info"]["joint_distpred_fit_forecast"] for result in experiment_results]
        )

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
            f"joint={mean_joint_distpred_fit_forecast:.2f}"
        )
        print(
            "    Avg sample width: "
            f"low={np.mean(low_sample_widths):.4f}, "
            f"high={np.mean(high_sample_widths):.4f}, "
            f"final={np.mean(final_sample_widths):.4f}"
        )

        representative = experiment_results[0]
        plot_path = f"meeting\\joint_lh_markov_interaction_h{horizon}.png"
        save_interaction_plot(
            representative["joint_bundle"],
            representative["estimated_low_train"],
            representative["estimated_high_train"],
            plot_path,
        )
        print(f"    Saved interaction plot     : {plot_path}")

    elapsed_seconds = time.perf_counter() - start_time
    print("")
    print(f"Elapsed time                   : {elapsed_seconds:.2f} seconds")
