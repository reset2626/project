"""Controlled six-method M1+M9 GARCH benchmark.

This complements, rather than replaces, the native-settings benchmark.  It
aligns all settings that can be shared without deleting the defining method
difference (DistPred, quantile regression, Minusformer, sort bootstrap, or
out-of-time conformal calibration).
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from kf_forecasting.models.kf_enbpi import EnbPIConfig
from kf_forecasting.models.kf_out_of_time_enbpi import OutOfTimeEnbPIConfig
from run_m1m9_garch_six_method_benchmark import (
    BenchmarkConfig,
    load_sister_definitions,
    run_one_seed,
    summarize,
    summarize_runtime,
)


OUTPUT_DIR = PROJECT_ROOT / "results" / "m1m9_garch_six_method_controlled"

# Shared settings for M1--M5.  DistPred output bins, quantile levels and the
# Minusformer-specific dimensions remain method-defining settings.
SISTER_CONTROL_OVERRIDES: dict[str, object] = {
    "seq_len": 30,
    "bootstrap_window": 30,
    "hidden_dims": [128, 256, 128],
    "dropout_ann": 0.0,
    "learning_rate": 0.001,
    "train_epochs": 50,
    "patience": 5,
    "bootstrap_iters": 100,
    "kalman_process_var": 0.5,
    "kalman_measurement_var": 10.0,
    "arima_max_p": 5,
    "arima_max_q": 5,
}

# M6 uses the same lag length, ANN widths, learning rate, nominal training
# budget, Kalman parameters, ARIMA search bounds and ensemble count.  Batch
# size stays 1 because sequential OOT validity requires it.  Moving-block
# resampling and cross-fitting are retained because they define M6.
OOT_CONTROL_CONFIG = OutOfTimeEnbPIConfig(
    base=EnbPIConfig(
        window_size=30,
        alpha=0.05,
        n_bootstrap=100,
        block_length=30,
        batch_size=1,
        arima_order=None,
        arima_max_p=5,
        arima_max_q=5,
        ann_hidden_layers=(128, 256, 128),
        ann_max_iter=50,
        ann_learning_rate_init=0.001,
        ann_target_standardization=False,
        ann_early_stopping=False,
        ann_rolling_validation=False,
        ann_iteration_candidates=(50,),
        process_variance=0.5,
        measurement_variance=10.0,
    ),
    initial_train_fraction=0.50,
    n_crossfit_blocks=4,
    residual_window_candidates=(100, 200, 300, 400, None),
    window_validation_fraction=0.40,
    min_window_history=40,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="*", type=int)
    parser.add_argument("--train-size", type=int, default=650)
    parser.add_argument("--horizon", type=int, default=200)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.smoke:
        seeds = (2020,)
        train_size, horizon = 80, 8
        sister_overrides = {
            **SISTER_CONTROL_OVERRIDES,
            "hidden_dims": [8],
            "d_model": 8,
            "d_ff": 8,
            "e_layers": 1,
            "bins": 10,
            "train_epochs": 1,
            "bootstrap_iters": 2,
            "arima_max_p": 1,
            "arima_max_q": 1,
        }
        oot_config = replace(
            OOT_CONTROL_CONFIG,
            base=replace(
                OOT_CONTROL_CONFIG.base,
                window_size=5,
                n_bootstrap=20,
                block_length=3,
                arima_order=(1, 0, 0),
                ann_hidden_layers=(6,),
                ann_max_iter=10,
                ann_iteration_candidates=(10,),
            ),
            initial_train_fraction=0.60,
            n_crossfit_blocks=2,
            residual_window_candidates=(10, None),
            min_window_history=6,
        )
    else:
        seeds = tuple(args.seeds) if args.seeds else tuple(range(2020, 2040))
        train_size, horizon = args.train_size, args.horizon
        sister_overrides = SISTER_CONTROL_OVERRIDES
        oot_config = OOT_CONTROL_CONFIG
    output_dir = (
        PROJECT_ROOT / "results" / "m1m9_garch_six_method_controlled_smoke"
        if args.smoke
        else OUTPUT_DIR
    )
    config = BenchmarkConfig(
        train_size=train_size,
        horizon=horizon,
        data_seeds=seeds,
        output_dir="results/m1m9_garch_six_method_controlled",
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    ns = load_sister_definitions()
    rows: list[dict[str, object]] = []
    predictions: list[pd.DataFrame] = []

    for seed in seeds:
        print(f"Running controlled paired seed {seed} ...", flush=True)
        seed_rows, seed_predictions = run_one_seed(
            seed,
            config,
            ns=ns,
            output_dir=output_dir,
            sister_overrides=sister_overrides,
            oot_config=oot_config,
        )
        rows.extend(seed_rows)
        predictions.append(seed_predictions)
        # Checkpoint after every seed so a long Monte Carlo run is auditable.
        pd.DataFrame(rows).to_csv(output_dir / "metrics_by_seed.csv", index=False)

    metrics = pd.DataFrame(rows)
    prediction_table = pd.concat(predictions, ignore_index=True)
    metric_summary = summarize(metrics)
    runtime_summary = summarize_runtime(metrics)
    metrics.to_csv(output_dir / "metrics_by_seed.csv", index=False)
    prediction_table.to_csv(output_dir / "predictions.csv", index=False)
    metric_summary.to_csv(output_dir / "summary.csv")
    runtime_summary.to_csv(output_dir / "runtime_summary.csv")
    settings = {
        "experiment": asdict(config),
        "shared": {
            "lag_window": 30,
            "ann_hidden_layers": [128, 256, 128],
            "ann_activation": "ReLU",
            "learning_rate": 0.001,
            "training_budget": 50,
            "bootstrap_or_ensemble_count_M5_M6": 100,
            "kalman_process_variance": 0.5,
            "kalman_measurement_variance": 10.0,
            "arima_max_p": 5,
            "arima_max_q": 5,
            "nominal_coverage": 0.95,
        },
        "sister_overrides": sister_overrides,
        "oot_config": asdict(oot_config),
        "smoke_test": args.smoke,
        "unavoidably_method_specific": {
            "M1_M2": "CRPS DistPred with 100 output samples",
            "M2_M4": "Minusformer backbone",
            "M3_M4": "pinball loss at 0.025, 0.5, 0.975",
            "M5": "100 index-sorted IID bootstrap ANN fits",
            "M6": "100 moving-block hybrid fits plus rolling-origin OOT calibration",
        },
    }
    (output_dir / "config.json").write_text(
        json.dumps(settings, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(metric_summary.to_string())
    print("\nRuntime summary")
    print(runtime_summary.to_string())


if __name__ == "__main__":
    main()
