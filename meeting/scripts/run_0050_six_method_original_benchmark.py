"""Six original-setting methods on 0050.TW adjusted close.

Train: 2020-01-01 through 2024-12-31.
Test:  2025-01-01 through 2026-07-31.

All six methods receive the same log adjusted-close series.  Their final
combined point forecasts and interval endpoints are exponentiated back to
price units before metrics and plots are produced.
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
import yfinance as yf


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


TICKER = "0050.TW"
TRAIN_START = "2020-01-01"
TRAIN_END = "2024-12-31"
TEST_START = "2025-01-01"
TEST_END = "2026-07-31"
MODEL_SEED = 2026
OUTPUT_DIR = PROJECT_ROOT / "results" / "0050_six_method_original"


def download_market_data() -> tuple[pd.Series, pd.Series]:
    cache_dir = OUTPUT_DIR / ".yfinance_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    yf.set_tz_cache_location(str(cache_dir))
    # Yahoo's end date is exclusive.
    download_end = (pd.Timestamp(TEST_END) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    raw = yf.download(
        TICKER,
        start=TRAIN_START,
        end=download_end,
        auto_adjust=True,
        progress=False,
    )
    if raw.empty:
        raise RuntimeError("Yahoo Finance returned no 0050.TW observations")
    close = raw["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close = close.dropna().astype(float)
    close.index = pd.to_datetime(close.index).tz_localize(None)
    train = close.loc[TRAIN_START:TRAIN_END]
    test = close.loc[TEST_START:TEST_END]
    if train.empty or test.empty:
        raise RuntimeError("Requested 0050 train or test period is empty")
    if test.index[-1] < pd.Timestamp("2026-07-30"):
        raise RuntimeError(
            f"Downloaded test data end at {test.index[-1].date()}, not 2026-07-31"
        )
    return train, test


def save_method_plot(frame: pd.DataFrame, method: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(frame["date"], frame["truth"], color="black", linewidth=1.6, label="Actual")
    ax.plot(frame["date"], frame["point"], color="tab:blue", linewidth=1.3, label="Point forecast")
    ax.fill_between(
        frame["date"],
        frame["lower"],
        frame["upper"],
        color="tab:blue",
        alpha=0.20,
        label="95% prediction interval",
    )
    ax.axvline(frame["date"].iloc[0], color="tab:red", linestyle="--", alpha=0.7)
    ax.set(title=f"0050.TW — {method}", xlabel="Date", ylabel="Adjusted close")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_panel_plot(predictions: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(18, 15), sharex=True)
    for ax, method in zip(axes.flat, METHOD_NAMES):
        frame = predictions.loc[predictions["method"] == method]
        ax.plot(frame["date"], frame["truth"], color="black", linewidth=1.2, label="Actual")
        ax.plot(frame["date"], frame["point"], color="tab:blue", linewidth=1.0, label="Point")
        ax.fill_between(
            frame["date"], frame["lower"], frame["upper"], color="tab:blue", alpha=0.20
        )
        ax.set_title(method)
        ax.grid(alpha=0.22)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.965), ncol=2)
    fig.suptitle(
        "0050.TW final combined point forecasts and 95% prediction intervals\n"
        "Train 2020–2024; test 2025–2026-07-31",
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    figure_dir = OUTPUT_DIR / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    if args.plot_only:
        predictions = pd.read_csv(OUTPUT_DIR / "predictions.csv", parse_dates=["date"])
        for method in METHOD_NAMES:
            safe_name = method.lower().replace(" ", "_")
            save_method_plot(
                predictions.loc[predictions["method"] == method],
                method,
                figure_dir / f"{safe_name}.png",
            )
        save_panel_plot(predictions, figure_dir / "all_six_methods.png")
        print(f"Regenerated figures in {figure_dir}")
        return
    train_close, test_close = download_market_data()
    full_close = pd.concat([train_close, test_close])
    full_close.rename("adjusted_close").to_csv(OUTPUT_DIR / "0050_adjusted_close.csv")

    observed_log = np.log(full_close.to_numpy(dtype=float))
    train_size = len(train_close)
    test_dates = pd.to_datetime(test_close.index)
    truth_price = test_close.to_numpy(dtype=float)
    ns = load_sister_definitions()
    rows: list[dict[str, object]] = []
    frames: list[pd.DataFrame] = []

    sister = run_sister_methods(
        observed_log,
        train_size,
        seed=MODEL_SEED,
        ns=ns,
        checkpoint_dir=OUTPUT_DIR / "checkpoints",
    )
    for method, (point_log, lower_log, upper_log, elapsed) in sister.items():
        point, lower, upper = np.exp(point_log), np.exp(lower_log), np.exp(upper_log)
        metric = interval_metrics(truth_price, point, lower, upper, alpha=0.05)
        rows.append({"method": method, **metric, "elapsed_seconds": elapsed})
        frames.append(
            pd.DataFrame(
                {
                    "date": test_dates,
                    "method": method,
                    "truth": truth_price,
                    "point": point,
                    "lower": lower,
                    "upper": upper,
                }
            )
        )

    oot_started = perf_counter()
    oot = run_out_of_time_enbpi(
        observed_log,
        train_size,
        config=OutOfTimeEnbPIConfig(
            base=EnbPIConfig(alpha=0.05, random_state=MODEL_SEED)
        ),
        model_name=METHOD_NAMES[5],
        data_seed=MODEL_SEED,
    )
    forecast = oot.forecast
    point, lower, upper = np.exp(forecast.point), np.exp(forecast.lower), np.exp(forecast.upper)
    metric = interval_metrics(truth_price, point, lower, upper, alpha=0.05)
    rows.append(
        {
            "method": METHOD_NAMES[5],
            **metric,
            "elapsed_seconds": perf_counter() - oot_started,
            "selected_residual_window": (
                "all" if oot.selected_residual_window is None else oot.selected_residual_window
            ),
        }
    )
    frames.append(
        pd.DataFrame(
            {
                "date": test_dates,
                "method": METHOD_NAMES[5],
                "truth": truth_price,
                "point": point,
                "lower": lower,
                "upper": upper,
            }
        )
    )

    predictions = pd.concat(frames, ignore_index=True)
    metrics = pd.DataFrame(rows)
    predictions.to_csv(OUTPUT_DIR / "predictions.csv", index=False)
    metrics.to_csv(OUTPUT_DIR / "metrics.csv", index=False)
    for method in METHOD_NAMES:
        safe_name = method.lower().replace(" ", "_")
        save_method_plot(
            predictions.loc[predictions["method"] == method],
            method,
            figure_dir / f"{safe_name}.png",
        )
    save_panel_plot(predictions, figure_dir / "all_six_methods.png")
    config = {
        "ticker": TICKER,
        "price": "Yahoo Finance auto-adjusted Close",
        "model_scale": "log adjusted close",
        "plot_scale": "adjusted close price",
        "train_start": TRAIN_START,
        "train_end": TRAIN_END,
        "test_start": TEST_START,
        "test_end": TEST_END,
        "n_train": train_size,
        "n_test": len(test_close),
        "model_seed": MODEL_SEED,
        "methods": list(METHOD_NAMES),
        "settings": "Original M1-M5 thesis settings and original M6 OOT settings",
    }
    (OUTPUT_DIR / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(metrics.to_string(index=False))
    print(f"Saved figures to {figure_dir}")


if __name__ == "__main__":
    main()
