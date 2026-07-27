from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import traceback

import numpy as np
import pandas as pd

from kf_forecasting.models.kf_enbpi import (
    EnbPIConfig,
    monte_carlo_oob_residual_diagnostics,
    simulate_and_run,
)


OUTPUT_DIR = Path("tmp_m1m9_oob_diagnostics")
OUTPUT_DIR.mkdir(exist_ok=True)

config = EnbPIConfig(
    window_size=15,
    alpha=0.05,
    n_bootstrap=30,
    block_length=None,
    batch_size=1,
    beta_grid_size=101,
    oob_bias_correction=True,
    oob_bias_correction_mode="combined",
    arima_order=None,
    arima_max_p=4,
    arima_max_q=4,
    ann_hidden_layers=(32, 16),
    ann_max_iter=500,
    ann_alpha=1e-4,
    ann_learning_rate_init=1e-3,
    ann_target_standardization=True,
    ann_early_stopping=False,
    ann_rolling_validation=True,
    ann_rolling_splits=3,
    ann_validation_fraction=0.10,
    ann_iteration_candidates=(125, 250, 500),
    ann_tol=1e-3,
    random_state=1234,
)


def run_one(run_and_seed: tuple[int, int]) -> tuple[dict, dict]:
    run, data_seed = run_and_seed
    result = simulate_and_run(
        "m1m9",
        train_size=650,
        horizon=200,
        config=config,
        data_seed=data_seed,
    )
    run_row = {"run": run, "data_seed": data_seed, **result.metrics()}
    one_run = pd.DataFrame([run_row])
    diagnostic_row, _ = monte_carlo_oob_residual_diagnostics(
        one_run, [result]
    )
    return run_row, diagnostic_row.iloc[0].to_dict()


def main() -> None:
    try:
        (OUTPUT_DIR / "status.txt").write_text("running", encoding="utf-8")
        children = np.random.SeedSequence(2026).spawn(20)
        jobs = [
            (run, int(child.generate_state(1)[0]))
            for run, child in enumerate(children, start=1)
        ]
        with ProcessPoolExecutor(max_workers=4) as executor:
            output = list(executor.map(run_one, jobs))
        runs = pd.DataFrame([item[0] for item in output])
        diagnostics = pd.DataFrame([item[1] for item in output])

        forecast_fields = [
            "mse", "rmse", "coverage", "mean_width",
            "low_kf_mse", "low_kf_rmse", "high_kf_mse", "high_kf_rmse",
            "low_true_mse", "low_true_rmse", "high_true_mse", "high_true_rmse",
            "low_enbpi_coverage", "low_enbpi_mean_width", "low_mean_beta",
            "high_enbpi_coverage", "high_enbpi_mean_width", "high_mean_beta",
            "low_true_coverage", "high_true_coverage",
            "mean_beta", "mean_bias_correction",
            "low_mean_bias_correction", "high_mean_bias_correction",
            "ann_selected_max_iter_mean", "ann_rolling_validation_mse",
            "ann_nonconverged_models", "elapsed_seconds",
        ]
        summary = (
            runs[forecast_fields]
            .agg(["mean", "std"])
            .T.reset_index(names="metric")
        )
        diagnostic_fields = [
            "final_fitted_residual_slope",
            "low_branch_residual_slope",
            "high_branch_residual_slope",
            "final_residual_by_low_slope",
            "final_residual_by_high_slope",
            "component_error_corr",
            "component_error_cross_term",
            "component_error_cross_term_share",
            "combined_oob_mse",
            "high_abs_error_by_abs_kf_low_corr",
            "high_squared_error_by_abs_kf_low_corr",
            "high_oob_mse_extreme_to_calm_low_state_ratio",
            "true_innovation_squared_by_abs_true_low_corr",
            "true_innovation_mse_extreme_to_calm_low_state_ratio",
            "fitted_residual_cov_total",
            "covariance_cancellation_fraction",
        ]
        diagnostic_summary = (
            diagnostics[diagnostic_fields]
            .agg(["mean", "std"])
            .T.reset_index(names="metric")
        )
        diagnostic_summary = pd.concat(
            [
                diagnostic_summary,
                pd.DataFrame(
                    [
                        {
                            "metric": "opposite_branch_slope_sign_rate",
                            "mean": diagnostics[
                                "opposite_branch_slope_signs"
                            ].mean(),
                            "std": np.nan,
                        },
                        {
                            "metric": (
                                "opposite_final_component_slope_sign_rate"
                            ),
                            "mean": diagnostics[
                                "opposite_final_component_slope_signs"
                            ].mean(),
                            "std": np.nan,
                        },
                    ]
                ),
            ],
            ignore_index=True,
        )
        runs.to_csv(OUTPUT_DIR / "runs.csv", index=False)
        summary.to_csv(OUTPUT_DIR / "forecast_summary.csv", index=False)
        diagnostics.to_csv(OUTPUT_DIR / "oob_diagnostics.csv", index=False)
        diagnostic_summary.to_csv(
            OUTPUT_DIR / "oob_diagnostic_summary.csv", index=False
        )
        (OUTPUT_DIR / "status.txt").write_text("complete", encoding="utf-8")
    except Exception:
        (OUTPUT_DIR / "status.txt").write_text(
            "failed\n" + traceback.format_exc(), encoding="utf-8"
        )
        raise


if __name__ == "__main__":
    main()
