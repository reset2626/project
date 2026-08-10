"""Paired diagnostic for M6 on AR(1) versus AR(1) plus a sinusoid.

The senior thesis states ``AR(1), phi=0.8 + 3*sin(x)`` but does not specify the
grid for x.  This diagnostic therefore uses the explicit, auditable definition

    x_t = 2*pi*t/period,  y_t = ar1_t + amplitude*sin(x_t).

For every seed, both conditions share the identical AR(1) innovations.  Any
paired change in interval width, coverage, interval score, or selected residual
window is therefore attributable to adding the deterministic periodic term.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from kf_forecasting.models.kf_enbpi import EnbPIConfig  # noqa: E402
from kf_forecasting.models.kf_out_of_time_enbpi import (  # noqa: E402
    OutOfTimeEnbPIConfig,
    run_out_of_time_enbpi,
)

from run_m1m9_garch_six_method_benchmark import interval_metrics  # noqa: E402


@dataclass(frozen=True)
class DiagnosticConfig:
    train_size: int = 650
    horizon: int = 200
    seeds: tuple[int, ...] = tuple(range(2020, 2040))
    phi: float = 0.8
    innovation_sd: float = 1.0
    amplitude: float = 3.0
    periodic_offset: float = 0.0
    heteroskedastic_min_sd: float = 1.0
    heteroskedastic_max_sd: float = 15.0
    conditions: tuple[str, ...] = ("AR1", "AR1_plus_sine")
    # Generate this many earlier observations and then retain only the final
    # train_size+horizon values. This permits several training sizes to share
    # exactly the same test suffix from one long seeded path.
    series_offset: int = 0
    period: int = 132
    alpha: float = 0.05
    output_dir: str = "results/m6_periodic_ar1_diagnostic"


def paired_series(config: DiagnosticConfig, seed: int) -> dict[str, np.ndarray]:
    """Generate paired paths with exactly the same stochastic innovations."""
    retained_n = config.train_size + config.horizon
    n = config.series_offset + retained_n
    rng = np.random.default_rng(seed)
    innovations = rng.normal(0.0, config.innovation_sd, size=n)
    ar1 = np.empty(n, dtype=float)
    # Draw from the stationary marginal distribution to avoid a zero-start artifact.
    ar1[0] = innovations[0] / np.sqrt(1.0 - config.phi**2)
    for t in range(1, n):
        ar1[t] = config.phi * ar1[t - 1] + innovations[t]
    time = np.arange(n, dtype=float)
    sine = np.sin(
        2.0 * np.pi * time / config.period
    )
    periodic = config.periodic_offset + config.amplitude * sine

    # Same standardized shocks as the homoskedastic AR(1), but innovation
    # variance increases smoothly from trough to peak.  This models the fact
    # that monthly sunspot fluctuations are much larger near cycle peaks.
    phase_level = (sine + 1.0) / 2.0
    innovation_sd = (
        config.heteroskedastic_min_sd
        + (config.heteroskedastic_max_sd - config.heteroskedastic_min_sd)
        * phase_level
    )
    hetero_ar1 = np.empty(n, dtype=float)
    hetero_ar1[0] = innovation_sd[0] * innovations[0] / np.sqrt(
        1.0 - config.phi**2
    )
    for t in range(1, n):
        hetero_ar1[t] = (
            config.phi * hetero_ar1[t - 1] + innovation_sd[t] * innovations[t]
        )
    hetero_observed = np.maximum(0.0, periodic + hetero_ar1)
    paths = {
        "AR1": ar1,
        "AR1_plus_sine": ar1 + periodic,
        "AR1_plus_sine_heteroskedastic": hetero_observed,
    }
    if config.series_offset:
        paths = {name: values[-retained_n:] for name, values in paths.items()}
    return paths


def m6_config(seed: int, alpha: float, *, smoke: bool) -> OutOfTimeEnbPIConfig:
    if smoke:
        base = EnbPIConfig(
            window_size=5,
            alpha=alpha,
            n_bootstrap=20,
            block_length=3,
            ann_hidden_layers=(6,),
            ann_max_iter=10,
            ann_rolling_validation=False,
            ann_iteration_candidates=(10,),
            arima_order=(1, 0, 0),
            random_state=seed,
        )
        return OutOfTimeEnbPIConfig(
            base=base,
            initial_train_fraction=0.6,
            n_crossfit_blocks=2,
            residual_window_candidates=(10, None),
            min_window_history=6,
        )
    return OutOfTimeEnbPIConfig(
        base=EnbPIConfig(alpha=alpha, random_state=seed)
    )


def run_condition(
    observed: np.ndarray,
    *,
    condition: str,
    seed: int,
    config: DiagnosticConfig,
    smoke: bool,
):
    oot = run_out_of_time_enbpi(
        observed,
        config.train_size,
        config=m6_config(seed, config.alpha, smoke=smoke),
        model_name=f"M6_{condition}",
        data_seed=seed,
    )
    f = oot.forecast
    metrics = interval_metrics(
        f.truth, f.point, f.lower, f.upper, alpha=config.alpha
    )
    test_range = float(np.max(f.truth) - np.min(f.truth))
    row = {
        "seed": seed,
        "condition": condition,
        **metrics,
        "test_range": test_range,
        "relative_mean_width": metrics["mean_width"] / test_range,
        "relative_median_width": metrics["median_width"] / test_range,
        "elapsed_seconds": f.elapsed_seconds,
        "selected_residual_window": (
            "all" if oot.selected_residual_window is None
            else oot.selected_residual_window
        ),
    }
    predictions = pd.DataFrame(
        {
            "seed": seed,
            "condition": condition,
            "time": f.test_times,
            "truth": f.truth,
            "point": f.point,
            "lower": f.lower,
            "upper": f.upper,
        }
    )
    window_selection = oot.window_selection.copy()
    window_selection.insert(0, "condition", condition)
    window_selection.insert(0, "seed", seed)
    return row, predictions, window_selection


def paired_differences(metrics: pd.DataFrame) -> pd.DataFrame:
    numeric = [
        "rmse",
        "mae",
        "coverage_percent",
        "mean_width",
        "median_width",
        "mean_interval_score",
        "test_range",
        "relative_mean_width",
        "relative_median_width",
        "elapsed_seconds",
    ]
    wide = metrics.pivot(index="seed", columns="condition", values=numeric)
    result = pd.DataFrame(index=wide.index)
    for column in numeric:
        result[f"delta_{column}"] = (
            wide[(column, "AR1_plus_sine")] - wide[(column, "AR1")]
        )
    return result.reset_index()


def save_plot(predictions: pd.DataFrame, path: Path) -> None:
    seeds = predictions["seed"].drop_duplicates().tolist()
    conditions = predictions["condition"].drop_duplicates().tolist()
    fig, axes = plt.subplots(
        len(seeds), len(conditions),
        figsize=(7.5 * len(conditions), 4 * len(seeds)), squeeze=False
    )
    for row_index, seed in enumerate(seeds):
        for column_index, condition in enumerate(conditions):
            frame = predictions[
                (predictions["seed"] == seed)
                & (predictions["condition"] == condition)
            ]
            ax = axes[row_index, column_index]
            ax.fill_between(
                frame["time"], frame["lower"], frame["upper"],
                color="tab:blue", alpha=0.18, label="95% PI"
            )
            ax.plot(frame["time"], frame["truth"], color="black", lw=1.1, label="Truth")
            ax.plot(frame["time"], frame["point"], color="tab:blue", lw=1.0, label="Point")
            ax.set_title(f"seed {seed}: {condition}")
            ax.grid(alpha=0.2)
            if row_index == 0 and column_index == 0:
                ax.legend(ncol=3)
    fig.suptitle("M6 paired periodicity diagnostic", y=1.0)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="*", type=int)
    parser.add_argument("--train-size", type=int, default=650)
    parser.add_argument("--horizon", type=int, default=200)
    parser.add_argument("--period", type=int, default=132)
    parser.add_argument("--amplitude", type=float, default=3.0)
    parser.add_argument("--periodic-offset", type=float, default=0.0)
    parser.add_argument("--heteroskedastic-min-sd", type=float, default=1.0)
    parser.add_argument("--heteroskedastic-max-sd", type=float, default=15.0)
    parser.add_argument("--output-dir", type=str)
    parser.add_argument("--series-offset", type=int, default=0)
    parser.add_argument(
        "--conditions",
        nargs="+",
        choices=("AR1", "AR1_plus_sine", "AR1_plus_sine_heteroskedastic"),
        default=("AR1", "AR1_plus_sine"),
    )
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    if args.smoke:
        config = DiagnosticConfig(
            train_size=80,
            horizon=8,
            seeds=(2020,),
            period=20,
            output_dir="results/m6_periodic_ar1_diagnostic_smoke",
        )
    else:
        if args.output_dir:
            output_dir = args.output_dir
        elif args.conditions == ["AR1_plus_sine_heteroskedastic"]:
            output_dir = "results/m6_periodic_ar1_heteroskedastic_diagnostic"
        elif args.amplitude == 3.0 and args.periodic_offset == 0.0:
            output_dir = "results/m6_periodic_ar1_diagnostic"
        else:
            output_dir = "results/m6_periodic_ar1_large_range_diagnostic"
        config = DiagnosticConfig(
            train_size=args.train_size,
            horizon=args.horizon,
            seeds=tuple(args.seeds) if args.seeds else tuple(range(2020, 2040)),
            period=args.period,
            amplitude=args.amplitude,
            periodic_offset=args.periodic_offset,
            heteroskedastic_min_sd=args.heteroskedastic_min_sd,
            heteroskedastic_max_sd=args.heteroskedastic_max_sd,
            conditions=tuple(args.conditions),
            series_offset=args.series_offset,
            output_dir=output_dir,
        )

    output = PROJECT_ROOT / config.output_dir
    output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []
    selection_frames: list[pd.DataFrame] = []

    for seed in config.seeds:
        paths = paired_series(config, seed)
        for condition in config.conditions:
            observed = paths[condition]
            print(f"Running seed {seed}, {condition} ...", flush=True)
            row, predictions, selection = run_condition(
                observed,
                condition=condition,
                seed=seed,
                config=config,
                smoke=args.smoke,
            )
            rows.append(row)
            prediction_frames.append(predictions)
            selection_frames.append(selection)
            pd.DataFrame(rows).to_csv(output / "metrics_by_seed.csv", index=False)

    metrics = pd.DataFrame(rows)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    selections = pd.concat(selection_frames, ignore_index=True)
    summary = metrics.groupby("condition", sort=False).agg(
        {
            "rmse": ["mean", "std"],
            "mae": ["mean", "std"],
            "coverage_percent": ["mean", "std"],
            "mean_width": ["mean", "std"],
            "median_width": ["mean", "std"],
            "mean_interval_score": ["mean", "std"],
            "test_range": ["mean", "std"],
            "relative_mean_width": ["mean", "std"],
            "relative_median_width": ["mean", "std"],
            "elapsed_seconds": ["mean", "sum"],
        }
    )
    if {"AR1", "AR1_plus_sine"}.issubset(set(metrics["condition"])):
        differences = paired_differences(metrics)
        difference_summary = differences.drop(columns="seed").agg(["mean", "std"])
    else:
        differences = pd.DataFrame()
        difference_summary = pd.DataFrame()

    metrics.to_csv(output / "metrics_by_seed.csv", index=False)
    predictions.to_csv(output / "predictions.csv", index=False)
    selections.to_csv(output / "window_selection.csv", index=False)
    summary.to_csv(output / "summary.csv")
    if not differences.empty:
        differences.to_csv(output / "paired_differences.csv", index=False)
        difference_summary.to_csv(output / "paired_difference_summary.csv")
    (output / "config.json").write_text(
        json.dumps(asdict(config), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    save_plot(predictions, output / "paired_predictions.png")
    print("\nCondition summary:\n", summary.to_string())
    if not difference_summary.empty:
        print("\nPaired difference (AR1_plus_sine - AR1):\n", difference_summary.to_string())
    print(f"\nSaved to {output}")


if __name__ == "__main__":
    main()
