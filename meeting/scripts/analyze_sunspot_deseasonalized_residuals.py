"""Diagnose monthly sunspot residuals after removing the ~11-year cycle.

This is a descriptive diagnostic, not a forecasting backtest. Robust STL is
applied to the observed series with a 132-month period. The script examines
whether remainder variance depends on the estimated cycle level, which helps
explain why a global residual interval can be excessively wide near zero.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.seasonal import STL


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "SN_m_tot_V2.0.csv"
OUTPUT_DIR = PROJECT_ROOT / "results" / "sunspot_deseasonalized_residuals"
START_DATE = "1900-01-01"
END_DATE = "2023-01-01"
PERIOD_MONTHS = 132
ROLLING_MONTHS = 36


def load_sunspots() -> pd.Series:
    frame = pd.read_csv(
        DATA_PATH,
        sep=";",
        header=None,
        names=(
            "year", "month", "decimal_date", "sunspot", "std",
            "observations", "provisional",
        ),
    )
    frame["date"] = pd.to_datetime(
        {"year": frame["year"], "month": frame["month"], "day": 1}
    )
    frame = frame.loc[
        frame["date"].between(START_DATE, END_DATE) & frame["sunspot"].ge(0)
    ]
    series = frame.set_index("date")["sunspot"].astype(float).sort_index()
    expected = pd.date_range(series.index[0], series.index[-1], freq="MS")
    if not series.index.equals(expected):
        raise ValueError("Sunspot series is not a complete monthly sequence")
    return series


def activity_summary(frame: pd.DataFrame) -> pd.DataFrame:
    # Equal-count bins isolate how remainder variability changes with cycle level.
    labels = ["low", "middle", "high"]
    frame = frame.copy()
    frame["activity_group"] = pd.qcut(
        frame["cycle_level"], q=3, labels=labels, duplicates="drop"
    )
    return (
        frame.groupby("activity_group", observed=True)
        .agg(
            n=("remainder", "size"),
            cycle_level_mean=("cycle_level", "mean"),
            observed_mean=("observed", "mean"),
            remainder_mean=("remainder", "mean"),
            remainder_sd=("remainder", "std"),
            remainder_mae=("remainder", lambda x: float(np.mean(np.abs(x)))),
            remainder_q025=("remainder", lambda x: float(np.quantile(x, 0.025))),
            remainder_q975=("remainder", lambda x: float(np.quantile(x, 0.975))),
        )
        .reset_index()
    )


def save_plots(frame: pd.DataFrame, output_dir: Path) -> None:
    fig, axes = plt.subplots(4, 1, figsize=(16, 15), sharex=True)
    axes[0].plot(frame.index, frame["observed"], color="black", lw=0.8)
    axes[0].set_title("Monthly total sunspots")
    axes[0].set_ylabel("Sunspot number")

    axes[1].plot(frame.index, frame["cycle_level"], color="tab:orange", lw=1.2)
    axes[1].set_title("Estimated cycle level (STL trend + 132-month seasonal component)")
    axes[1].set_ylabel("Cycle level")

    axes[2].axhline(0.0, color="grey", lw=0.8)
    axes[2].plot(frame.index, frame["remainder"], color="tab:blue", lw=0.7)
    axes[2].set_title("Remainder after removing the estimated cycle")
    axes[2].set_ylabel("Remainder")

    axes[3].plot(
        frame.index, frame["rolling_remainder_sd"], color="tab:red", lw=1.1
    )
    axes[3].set_title(f"Rolling remainder SD ({ROLLING_MONTHS} months)")
    axes[3].set_ylabel("SD")
    axes[3].set_xlabel("Date")
    for ax in axes:
        ax.grid(alpha=0.22)
    fig.tight_layout()
    fig.savefig(output_dir / "cycle_removal_and_residuals.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    axes[0].scatter(
        frame["cycle_level"], np.abs(frame["remainder"]),
        s=10, alpha=0.35, color="tab:blue",
    )
    axes[0].set(
        title="Absolute remainder versus estimated cycle level",
        xlabel="Estimated cycle level", ylabel="Absolute remainder",
    )
    axes[0].grid(alpha=0.22)
    plot_acf(frame["remainder"], lags=264, ax=axes[1], zero=False)
    axes[1].set_title("ACF of deseasonalized remainder")
    fig.tight_layout()
    fig.savefig(output_dir / "residual_scale_and_acf.png", dpi=180)
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    observed = load_sunspots()
    fit = STL(
        observed,
        period=PERIOD_MONTHS,
        seasonal=13,
        trend=199,
        robust=True,
    ).fit()
    frame = pd.DataFrame(
        {
            "observed": observed,
            "trend": fit.trend,
            "seasonal_132": fit.seasonal,
            "cycle_level": fit.trend + fit.seasonal,
            "remainder": fit.resid,
        }
    )
    frame["abs_remainder"] = frame["remainder"].abs()
    frame["rolling_remainder_sd"] = frame["remainder"].rolling(
        ROLLING_MONTHS, min_periods=12
    ).std()
    by_activity = activity_summary(frame)

    pearson_abs = float(frame["cycle_level"].corr(frame["abs_remainder"]))
    spearman_abs = float(
        frame["cycle_level"].corr(frame["abs_remainder"], method="spearman")
    )
    low_sd = float(by_activity.loc[by_activity["activity_group"] == "low", "remainder_sd"].iloc[0])
    high_sd = float(by_activity.loc[by_activity["activity_group"] == "high", "remainder_sd"].iloc[0])
    diagnostics = {
        "data_range": [str(observed.index[0].date()), str(observed.index[-1].date())],
        "n_months": int(len(observed)),
        "decomposition": "robust STL on original scale",
        "period_months": PERIOD_MONTHS,
        "rolling_sd_months": ROLLING_MONTHS,
        "corr_cycle_level_abs_remainder_pearson": pearson_abs,
        "corr_cycle_level_abs_remainder_spearman": spearman_abs,
        "high_to_low_remainder_sd_ratio": high_sd / low_sd,
        "interpretation_scope": "descriptive full-series diagnostic; not causal OOT validation",
    }

    frame.to_csv(OUTPUT_DIR / "deseasonalized_components.csv")
    by_activity.to_csv(OUTPUT_DIR / "residual_by_activity.csv", index=False)
    (OUTPUT_DIR / "diagnostics.json").write_text(
        json.dumps(diagnostics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    save_plots(frame, OUTPUT_DIR)
    print(json.dumps(diagnostics, ensure_ascii=False, indent=2))
    print("\nResidual dispersion by estimated activity level:")
    print(by_activity.to_string(index=False))
    print(f"\nSaved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
