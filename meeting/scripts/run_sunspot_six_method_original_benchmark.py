"""Six original-setting methods on monthly total sunspot numbers.

The data range and chronological 80/20 split reproduce the senior thesis:
1900-01-01 through 2023-01-01.  Models are trained on log1p(sunspot),
then point forecasts and interval endpoints are transformed back with expm1.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

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
from run_m1m9_garch_six_method_benchmark import (  # noqa: E402
    METHOD_NAMES,
    interval_metrics,
    load_sister_definitions,
    run_sister_methods,
)


DATA_PATH = PROJECT_ROOT / "data" / "raw" / "SN_m_tot_V2.0.csv"
START_DATE = "1900-01-01"
END_DATE = "2023-01-01"
TRAIN_FRACTION = 0.80
MODEL_SEED = 2026
OUTPUT_DIR = PROJECT_ROOT / "results" / "sunspot_six_method_log1p"


def load_sunspot_data() -> pd.Series:
    frame = pd.read_csv(
        DATA_PATH,
        sep=";",
        header=None,
        names=(
            "year",
            "month",
            "decimal_date",
            "sunspot",
            "std",
            "observations",
            "provisional",
        ),
    )
    frame["date"] = pd.to_datetime(
        {"year": frame["year"], "month": frame["month"], "day": 1}
    )
    frame = frame.loc[
        frame["date"].between(START_DATE, END_DATE)
        & frame["sunspot"].ge(0)
    ].copy()
    series = frame.set_index("date")["sunspot"].astype(float).sort_index()
    expected = pd.date_range(series.index[0], series.index[-1], freq="MS")
    if not series.index.equals(expected):
        raise ValueError("Filtered sunspot series is not complete monthly data")
    return series


def save_method_plot(frame: pd.DataFrame, method: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(frame["date"], frame["truth"], color="black", linewidth=1.5, label="Actual")
    ax.plot(frame["date"], frame["point"], color="tab:blue", linewidth=1.2, label="Point forecast")
    ax.fill_between(
        frame["date"], frame["lower"], frame["upper"],
        color="tab:blue", alpha=0.20, label="95% prediction interval",
    )
    ax.set(title=f"Monthly sunspots — {method}", xlabel="Date", ylabel="Sunspot number")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_panel_plot(predictions: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(18, 15), sharex=True)
    for ax, method in zip(axes.flat, METHOD_NAMES):
        frame = predictions.loc[predictions["method"] == method]
        ax.plot(frame["date"], frame["truth"], color="black", linewidth=1.1, label="Actual")
        ax.plot(frame["date"], frame["point"], color="tab:blue", linewidth=1.0, label="Point")
        ax.fill_between(
            frame["date"], frame["lower"], frame["upper"],
            color="tab:blue", alpha=0.20,
        )
        ax.set_title(method)
        ax.grid(alpha=0.22)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.965), ncol=2)
    fig.suptitle(
        "Monthly sunspots: final combined point forecasts and 95% prediction intervals\n"
        "Chronological 80/20 split of 1900-01 through 2023-01",
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_plots(predictions: pd.DataFrame, figure_dir: Path) -> None:
    for method in METHOD_NAMES:
        safe_name = method.lower().replace(" ", "_")
        save_method_plot(
            predictions.loc[predictions["method"] == method], method,
            figure_dir / f"{safe_name}.png",
        )
    save_panel_plot(predictions, figure_dir / "all_six_methods.png")


def inverse_log1p(values: np.ndarray) -> np.ndarray:
    """Return forecasts to the non-negative sunspot-number support."""
    return np.maximum(0.0, np.expm1(np.asarray(values, dtype=float)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    figure_dir = OUTPUT_DIR / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    if args.plot_only:
        predictions = pd.read_csv(OUTPUT_DIR / "predictions.csv", parse_dates=["date"])
        write_plots(predictions, figure_dir)
        print(f"Regenerated figures in {figure_dir}")
        return

    series = load_sunspot_data()
    train_size = int(TRAIN_FRACTION * len(series))
    train, test = series.iloc[:train_size], series.iloc[train_size:]
    observed = np.log1p(series.to_numpy(dtype=float))
    truth = test.to_numpy(dtype=float)
    test_dates = pd.to_datetime(test.index)
    ns = load_sister_definitions()
    rows: list[dict[str, object]] = []
    frames: list[pd.DataFrame] = []

    sister = run_sister_methods(
        observed, train_size, seed=MODEL_SEED, ns=ns,
        checkpoint_dir=OUTPUT_DIR / "checkpoints",
    )
    for method, (point_log, lower_log, upper_log, elapsed) in sister.items():
        point = inverse_log1p(point_log)
        lower = inverse_log1p(lower_log)
        upper = inverse_log1p(upper_log)
        metric = interval_metrics(truth, point, lower, upper, alpha=0.05)
        rows.append({"method": method, **metric, "elapsed_seconds": elapsed})
        frames.append(pd.DataFrame({
            "date": test_dates, "method": method, "truth": truth,
            "point": point, "lower": lower, "upper": upper,
        }))

    oot_started = perf_counter()
    oot = run_out_of_time_enbpi(
        observed, train_size,
        config=OutOfTimeEnbPIConfig(
            base=EnbPIConfig(alpha=0.05, random_state=MODEL_SEED)
        ),
        model_name=METHOD_NAMES[5], data_seed=MODEL_SEED,
    )
    forecast = oot.forecast
    point = inverse_log1p(forecast.point)
    lower = inverse_log1p(forecast.lower)
    upper = inverse_log1p(forecast.upper)
    metric = interval_metrics(
        truth, point, lower, upper, alpha=0.05
    )
    rows.append({
        "method": METHOD_NAMES[5], **metric,
        "elapsed_seconds": perf_counter() - oot_started,
        "selected_residual_window": (
            "all" if oot.selected_residual_window is None else oot.selected_residual_window
        ),
    })
    frames.append(pd.DataFrame({
        "date": test_dates, "method": METHOD_NAMES[5], "truth": truth,
        "point": point, "lower": lower, "upper": upper,
    }))

    predictions = pd.concat(frames, ignore_index=True)
    metrics = pd.DataFrame(rows)
    predictions.to_csv(OUTPUT_DIR / "predictions.csv", index=False)
    metrics.to_csv(OUTPUT_DIR / "metrics.csv", index=False)
    write_plots(predictions, figure_dir)
    config = {
        "data_file": str(DATA_PATH.relative_to(PROJECT_ROOT)),
        "frequency": "monthly",
        "training_scale": "log1p(monthly total sunspot number)",
        "evaluation_scale": "monthly total sunspot number after expm1",
        "support_constraint": "point/lower/upper clipped at zero after expm1",
        "source_range": [str(series.index[0].date()), str(series.index[-1].date())],
        "split": "chronological 80/20",
        "train_range": [str(train.index[0].date()), str(train.index[-1].date())],
        "test_range": [str(test.index[0].date()), str(test.index[-1].date())],
        "n_total": len(series), "n_train": len(train), "n_test": len(test),
        "model_seed": MODEL_SEED, "methods": list(METHOD_NAMES),
        "settings": "Original M1-M5 thesis settings and original M6 OOT settings",
    }
    (OUTPUT_DIR / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(config, ensure_ascii=False, indent=2))
    print(metrics.to_string(index=False))
    print(f"Saved figures to {figure_dir}")


if __name__ == "__main__":
    main()
