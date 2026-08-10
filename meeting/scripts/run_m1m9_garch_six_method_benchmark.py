"""Paired six-method benchmark on the current M1+M9 GARCH DGP.

Methods M1--M5 retain the implementation and hyperparameters in Hung Wei-Ling's
thesis archive.  M5 is the senior thesis' sort bootstrap: resample (value,
original-time-index) pairs with replacement, sort by the sampled indices, fit
one ANN per resample, average its forecasts for the point prediction, and use
the 2.5/97.5 percentiles for the nonlinear interval.  M6 is the repository's
rolling-origin out-of-time adaptive EnbPI-type method.

Every method receives the exact same observed path for each data seed.  The
default DGP is ``m1m9_garch_low_constant_high_variance`` with T=650 and H=200.
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from kf_forecasting.models.kf_enbpi import (  # noqa: E402
    EnbPIConfig,
    simulate_m1_m9_garch_low_constant_high_variance_data,
)
from kf_forecasting.models.kf_out_of_time_enbpi import (  # noqa: E402
    OutOfTimeEnbPIConfig,
    run_out_of_time_enbpi,
)


METHOD_NAMES = (
    "M1_DistPred_ANN",
    "M2_DistPred_Minusformer",
    "M3_Quantile_ANN",
    "M4_Quantile_Minusformer",
    "M5_SortBootstrap_ANN",
    "M6_OOT_Adaptive_EnbPI",
)


@dataclass
class BenchmarkConfig:
    train_size: int = 650
    horizon: int = 200
    data_seeds: tuple[int, ...] = tuple(range(2020, 2040))
    alpha: float = 0.05
    output_dir: str = "results/m1m9_garch_six_method"


def _find_sister_source() -> Path:
    matches = list(
        (PROJECT_ROOT / "tmp_hong_thesis_archive").rglob(
            "AR1_GARCH_kalman_arima_5methods.py"
        )
    )
    if len(matches) != 1:
        raise FileNotFoundError(
            "Expected one extracted AR1_GARCH_kalman_arima_5methods.py; "
            "extract the supplied thesis archive under tmp_hong_thesis_archive"
        )
    return matches[0]


def load_sister_definitions(source: Path | None = None) -> dict[str, object]:
    """Load definitions without executing the thesis' 300-run top-level job."""
    source = source or _find_sister_source()
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    safe_nodes = [
        node
        for node in tree.body
        if isinstance(
            node,
            (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.ClassDef),
        )
        and not (isinstance(node, ast.FunctionDef) and node.name == "run_one_experiment")
    ]
    module = ast.Module(body=safe_nodes, type_ignores=[])
    namespace: dict[str, object] = {"__name__": "sister_thesis_methods"}
    exec(compile(module, str(source), "exec"), namespace)
    return namespace


def interval_metrics(
    truth: np.ndarray,
    point: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    alpha: float,
) -> dict[str, float]:
    truth = np.asarray(truth, dtype=float)
    point = np.asarray(point, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    if not (truth.shape == point.shape == lower.shape == upper.shape):
        raise ValueError("truth, point, lower and upper must have equal shapes")
    if np.any(lower > upper):
        raise ValueError("Prediction interval has lower > upper")
    error = truth - point
    width = upper - lower
    below = truth < lower
    above = truth > upper
    score = width + (2.0 / alpha) * (
        (lower - truth) * below + (truth - upper) * above
    )
    return {
        "mse": float(np.mean(error**2)),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
        "coverage": float(np.mean(~below & ~above)),
        "coverage_percent": float(100.0 * np.mean(~below & ~above)),
        "mean_width": float(np.mean(width)),
        "median_width": float(np.median(width)),
        "mean_interval_score": float(np.mean(score)),
        "lower_miss_rate": float(np.mean(below)),
        "upper_miss_rate": float(np.mean(above)),
    }


def _sister_config(
    ns: dict[str, object],
    *,
    smoke: bool,
    overrides: dict[str, object] | None = None,
) -> object:
    cfg = ns["ExperimentConfig"]()
    # Method hyperparameters remain the thesis defaults.  Smoke mode is an
    # explicit computational test only and is never used for reported results.
    if smoke:
        cfg.hidden_dims = [8]
        cfg.d_model = 8
        cfg.d_ff = 8
        cfg.e_layers = 1
        cfg.bins = 10
        cfg.train_epochs = 1
        cfg.patience = 1
        cfg.bootstrap_iters = 2
        cfg.arima_max_p = 1
        cfg.arima_max_q = 1
    for name, value in (overrides or {}).items():
        if not hasattr(cfg, name):
            raise AttributeError(f"Unknown sister configuration field: {name}")
        setattr(cfg, name, value)
    cfg.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return cfg


def run_sister_methods(
    observed: np.ndarray,
    train_size: int,
    *,
    seed: int,
    ns: dict[str, object],
    checkpoint_dir: Path,
    smoke: bool = False,
    sister_overrides: dict[str, object] | None = None,
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, float]]:
    common_started = perf_counter()
    cfg = _sister_config(ns, smoke=smoke, overrides=sister_overrides)
    ns["seed_everything"](seed)
    y = np.asarray(observed, dtype=float)
    train_y, test_y = y[:train_size], y[train_size:]

    low_all = ns["kalman_filter"](
        y, cfg.kalman_process_var, cfg.kalman_measurement_var
    )
    high_all = y - low_all
    low_train, low_test = low_all[:train_size], low_all[train_size:]
    high_train, high_test = high_all[:train_size], high_all[train_size:]
    order = ns["find_arima_order"](
        low_train, max_p=cfg.arima_max_p, max_q=cfg.arima_max_q
    )
    arima_point, arima_lower, arima_upper = ns["arima_rolling_predict"](
        low_train, low_test, order
    )

    ann_train, ann_val, ann_test = ns["make_loaders_ann"](
        high_train, high_test, cfg
    )
    tf_train, tf_val, tf_test = ns["make_loaders_tf"](
        high_train, high_test, cfg
    )
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    common_elapsed = perf_counter() - common_started

    def train(model_name: str, model: object, tr: object, va: object, loss: object):
        ckpt = checkpoint_dir / f"{model_name}_seed{seed}.pth"
        return ns["_train_loop"](model, tr, va, cfg, loss, str(ckpt))

    method_started = perf_counter()
    m1 = train(
        "m1",
        ns["DistPredANN"](cfg).to(cfg.device),
        ann_train,
        ann_val,
        ns["make_loss_distpred_ann"](cfg),
    )
    bins1 = ns["predict_distpred_ann"](m1, ann_test, cfg)
    elapsed1 = common_elapsed + perf_counter() - method_started
    method_started = perf_counter()
    m2 = train(
        "m2",
        ns["DistPredMinusformer"](cfg).to(cfg.device),
        tf_train,
        tf_val,
        ns["make_loss_distpred_tf"](cfg),
    )
    bins2 = ns["predict_distpred_tf"](m2, tf_test, cfg)
    elapsed2 = common_elapsed + perf_counter() - method_started
    method_started = perf_counter()
    m3 = train(
        "m3",
        ns["QuantileANN_Monotone"](cfg).to(cfg.device),
        ann_train,
        ann_val,
        ns["make_loss_quantile_ann"](cfg),
    )
    quant3 = ns["predict_quantile_ann"](m3, ann_test, cfg)
    elapsed3 = common_elapsed + perf_counter() - method_started
    method_started = perf_counter()
    m4 = train(
        "m4",
        ns["QuantileMinusformer_Monotone"](cfg).to(cfg.device),
        tf_train,
        tf_val,
        ns["make_loss_quantile_tf"](cfg),
    )
    quant4 = ns["predict_quantile_tf"](m4, tf_test, cfg)
    elapsed4 = common_elapsed + perf_counter() - method_started
    method_started = perf_counter()
    boot_point, boot_lower, boot_upper = ns["predict_sort_bootstrap_ann"](
        high_train, high_test, cfg, base_seed=seed
    )
    elapsed5 = common_elapsed + perf_counter() - method_started

    def combine(point: np.ndarray, lower: np.ndarray, upper: np.ndarray):
        return arima_point + point, arima_lower + lower, arima_upper + upper

    return {
        METHOD_NAMES[0]: (*combine(
            np.percentile(bins1, 50.0, axis=1),
            np.percentile(bins1, 2.5, axis=1),
            np.percentile(bins1, 97.5, axis=1),
        ), elapsed1),
        METHOD_NAMES[1]: (*combine(
            np.percentile(bins2, 50.0, axis=1),
            np.percentile(bins2, 2.5, axis=1),
            np.percentile(bins2, 97.5, axis=1),
        ), elapsed2),
        METHOD_NAMES[2]: (*combine(quant3[:, 1], quant3[:, 0], quant3[:, 2]), elapsed3),
        METHOD_NAMES[3]: (*combine(quant4[:, 1], quant4[:, 0], quant4[:, 2]), elapsed4),
        METHOD_NAMES[4]: (*combine(boot_point, boot_lower, boot_upper), elapsed5),
    }


def run_one_seed(
    seed: int,
    config: BenchmarkConfig,
    *,
    ns: dict[str, object],
    output_dir: Path,
    smoke: bool = False,
    sister_overrides: dict[str, object] | None = None,
    oot_config: OutOfTimeEnbPIConfig | None = None,
) -> tuple[list[dict[str, object]], pd.DataFrame]:
    rng = np.random.default_rng(seed)
    true_low, true_high, observed = (
        simulate_m1_m9_garch_low_constant_high_variance_data(
            config.train_size + config.horizon, rng=rng
        )
    )
    truth = observed[config.train_size :]
    rows: list[dict[str, object]] = []
    predictions: list[pd.DataFrame] = []

    sister = run_sister_methods(
        observed,
        config.train_size,
        seed=seed,
        ns=ns,
        checkpoint_dir=output_dir / "checkpoints",
        smoke=smoke,
        sister_overrides=sister_overrides,
    )
    for method, (point, lower, upper, elapsed_seconds) in sister.items():
        metric = interval_metrics(truth, point, lower, upper, alpha=config.alpha)
        rows.append(
            {"seed": seed, "method": method, **metric, "elapsed_seconds": elapsed_seconds}
        )
        predictions.append(
            pd.DataFrame(
                {
                    "seed": seed,
                    "time": np.arange(config.train_size, len(observed)),
                    "method": method,
                    "truth": truth,
                    "point": point,
                    "lower": lower,
                    "upper": upper,
                }
            )
        )

    if smoke:
        base = EnbPIConfig(
            window_size=5,
            n_bootstrap=20,
            block_length=3,
            ann_hidden_layers=(6,),
            ann_max_iter=10,
            ann_rolling_validation=False,
            ann_iteration_candidates=(10,),
            arima_order=(1, 0, 0),
            random_state=seed,
        )
        oot_cfg = OutOfTimeEnbPIConfig(
            base=base,
            initial_train_fraction=0.6,
            n_crossfit_blocks=2,
            residual_window_candidates=(10, None),
            min_window_history=6,
        )
    elif oot_config is None:
        oot_cfg = OutOfTimeEnbPIConfig(
            base=EnbPIConfig(alpha=config.alpha, random_state=seed)
        )
    else:
        oot_cfg = replace(
            oot_config,
            base=replace(oot_config.base, alpha=config.alpha, random_state=seed),
        )
    oot = run_out_of_time_enbpi(
        observed,
        config.train_size,
        config=oot_cfg,
        model_name=METHOD_NAMES[5],
        true_low=true_low,
        true_high=true_high,
        data_seed=seed,
    )
    forecast = oot.forecast
    metric = interval_metrics(
        forecast.truth,
        forecast.point,
        forecast.lower,
        forecast.upper,
        alpha=config.alpha,
    )
    rows.append(
        {
            "seed": seed,
            "method": METHOD_NAMES[5],
            **metric,
            "elapsed_seconds": forecast.elapsed_seconds,
            "selected_residual_window": (
                "all" if oot.selected_residual_window is None else oot.selected_residual_window
            ),
        }
    )
    predictions.append(
        pd.DataFrame(
            {
                "seed": seed,
                "time": forecast.test_times,
                "method": METHOD_NAMES[5],
                "truth": forecast.truth,
                "point": forecast.point,
                "lower": forecast.lower,
                "upper": forecast.upper,
            }
        )
    )
    return rows, pd.concat(predictions, ignore_index=True)


def summarize(metrics: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "mse",
        "rmse",
        "mae",
        "coverage_percent",
        "mean_width",
        "median_width",
        "mean_interval_score",
        "lower_miss_rate",
        "upper_miss_rate",
    ]
    return metrics.groupby("method", sort=False)[columns].agg(["mean", "std"])


def summarize_runtime(metrics: pd.DataFrame) -> pd.DataFrame:
    runtime = metrics.groupby("method", sort=False)["elapsed_seconds"].agg(
        total_seconds="sum",
        mean_seconds="mean",
        std_seconds="std",
        min_seconds="min",
        max_seconds="max",
        completed_seeds="count",
    )
    runtime["total_minutes"] = runtime["total_seconds"] / 60.0
    runtime["total_hours"] = runtime["total_seconds"] / 3600.0
    return runtime


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="tiny integration test")
    parser.add_argument("--seeds", nargs="*", type=int)
    parser.add_argument("--train-size", type=int, default=650)
    parser.add_argument("--horizon", type=int, default=200)
    args = parser.parse_args()

    if args.smoke:
        # Enough rows for every architecture while keeping the test quick.
        train_size = 80
        horizon = 8
        seeds = (2020,)
    else:
        train_size = args.train_size
        horizon = args.horizon
        seeds = tuple(args.seeds) if args.seeds else tuple(range(2020, 2040))
    result_subdir = (
        "results/m1m9_garch_six_method_smoke"
        if args.smoke
        else "results/m1m9_garch_six_method"
    )
    config = BenchmarkConfig(
        train_size=train_size,
        horizon=horizon,
        data_seeds=seeds,
        output_dir=result_subdir,
    )
    output_dir = PROJECT_ROOT / config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    ns = load_sister_definitions()

    all_rows: list[dict[str, object]] = []
    all_predictions: list[pd.DataFrame] = []
    for seed in config.data_seeds:
        print(f"Running paired seed {seed} ...", flush=True)
        rows, predictions = run_one_seed(
            seed, config, ns=ns, output_dir=output_dir, smoke=args.smoke
        )
        all_rows.extend(rows)
        all_predictions.append(predictions)
        pd.DataFrame(all_rows).to_csv(output_dir / "metrics_by_seed.csv", index=False)

    metrics = pd.DataFrame(all_rows)
    prediction_table = pd.concat(all_predictions, ignore_index=True)
    summary = summarize(metrics)
    runtime_summary = summarize_runtime(metrics)
    metrics.to_csv(output_dir / "metrics_by_seed.csv", index=False)
    prediction_table.to_csv(output_dir / "predictions.csv", index=False)
    summary.to_csv(output_dir / "summary.csv")
    runtime_summary.to_csv(output_dir / "runtime_summary.csv")
    (output_dir / "config.json").write_text(
        json.dumps(asdict(config), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(summary.to_string())
    print("\nRuntime summary")
    print(runtime_summary.to_string())


if __name__ == "__main__":
    main()
