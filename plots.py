"""Draw the report figures from a finished run: python plots.py

Reads results/metrics.csv, results/loss_log.csv and results/predictions/test.npz,
and writes seven PNG files to results/figures.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import forecast as fc

FIGURES = Path("results/figures")
BLUE, ORANGE, INK = "#2979D9", "#EB6834", "#111111"  # LSTM, persistence, observed
STYLE = {"persistence": dict(color=ORANGE, linestyle="--", marker="s", label="Persistence"),
         "lstm": dict(color=BLUE, linestyle="-", marker="o", label="LSTM")}
UNIT = "W/m$^2$"


def save(fig, name: str) -> None:
    for ax in fig.axes:
        ax.grid(True, color="#E6E6E6")
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGURES / name, dpi=150)
    plt.close(fig)


def one_day(run, df, pick, title: str, name: str) -> None:
    """Observed against both forecasts for the clearest or the most clouded test day."""
    times = pd.DatetimeIndex(run["times"].ravel())
    cloud = df["CLOUD_AMT"].reindex(times).to_numpy().reshape(run["actual"].shape)
    cloud = np.where(run["actual"] > fc.DAYLIGHT, cloud, np.nan)  # daylight hours only
    i = int(pick(np.nanmean(cloud, axis=1)))
    hours = np.arange(1, 25)
    fig, ax = plt.subplots(figsize=(7.1, 3.8))
    ax.plot(hours, run["actual"][i], color=INK, linewidth=2, label="Observed")
    for model, style in STYLE.items():
        ax.plot(hours, run[model][i], linewidth=2, markersize=4, **style)
    date = pd.Timestamp(run["times"][i][0]).date()
    ax.set(title=f"Next-day forecast on the {title} test day, {date}", xlabel="Hour of day (local solar time)",
           ylabel=f"GHI ({UNIT})", xticks=range(0, 25, 3), ylim=(0, None))
    ax.legend(frameon=False)
    save(fig, name)


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    df = fc.load_data()
    run = np.load("results/predictions/test.npz")
    metrics = pd.read_csv("results/metrics.csv")
    whole_year = metrics[(metrics["split"] == "test") & (metrics["season"] == "all")]
    test = whole_year.set_index(["model", "metric"])["value"].sort_index()

    one_day(run, df, np.nanargmin, "clearest", "fig1_clear_day.png")
    one_day(run, df, np.nanargmax, "most clouded", "fig2_cloudy_day.png")

    # Error by hour of day.
    fig, ax = plt.subplots(figsize=(7.1, 3.8))
    for model, style in STYLE.items():
        hours = [h for h in range(24) if (model, f"mae_masked_hour_{h:02d}") in test.index]
        ax.plot(hours, [test[model, f"mae_masked_hour_{h:02d}"] for h in hours], linewidth=2, **style)
    ax.set(title="Forecast error by hour of day, daylight hours of the test year",
           xlabel="Hour of day (local solar time)", ylabel=f"MAE ({UNIT})", ylim=(0, None))
    ax.legend(frameon=False)
    save(fig, "fig3_error_by_hour.png")

    # Error by month, with the wet season shaded.
    fig, ax = plt.subplots(figsize=(7.5, 3.9))
    months = np.arange(1, 13)
    ax.axvspan(min(fc.WET_MONTHS) - 0.5, 12.5, color="#EFEEEC", label="Wet season")
    for offset, (model, style) in zip((-0.19, 0.19), STYLE.items()):
        ax.bar(months + offset, [test[model, f"rmse_masked_month_{m:02d}"] for m in months], width=0.38,
               color=style["color"], label=style["label"])
    ax.set(title="Forecast error by month, daylight hours of the test year", xlabel="Month of the test year",
           ylabel=f"RMSE ({UNIT})", xticks=months,
           xticklabels=["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
    ax.set_ylim(0, ax.get_ylim()[1] * 1.25)  # headroom so the legend clears the bars
    ax.legend(frameon=False, ncols=3, loc="upper left")
    save(fig, "fig4_seasonal.png")

    # Predicted against observed, one panel per model.
    day = run["actual"] > fc.DAYLIGHT
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 4.4), sharex=True, sharey=True)
    for ax, (model, style) in zip(axes, STYLE.items()):
        ax.scatter(run["actual"][day], run[model][day], s=4, alpha=0.3, color=style["color"])
        ax.plot([0, 1100], [0, 1100], color=INK, linewidth=1)
        ax.set(title=style["label"], xlabel=f"Observed GHI ({UNIT})")
    axes[0].set_ylabel(f"Predicted GHI ({UNIT})")
    save(fig, "fig5_scatter.png")

    # Training history.
    history = pd.read_csv("results/loss_log.csv")
    best = history.loc[history["val_loss"].idxmin(), "epoch"]
    fig, ax = plt.subplots(figsize=(7.1, 3.8))
    ax.plot(history["epoch"], history["train_loss"], color=BLUE, marker="o", label="Training loss")
    ax.plot(history["epoch"], history["val_loss"], color=ORANGE, marker="s", linestyle="--", label="Validation loss")
    ax.axvline(best, color=INK, linestyle=":", label=f"Best epoch ({best})")
    ax.set(title="Training and validation loss", xlabel="Epoch", ylabel="MSE on scaled GHI")
    ax.legend(frameon=False)
    save(fig, "fig6_loss_curve.png")

    # What a typical day looks like across all six years.
    by_hour = df["ghi"].groupby(df.index.hour)
    mean, spread = by_hour.mean(), by_hour.std()
    fig, ax = plt.subplots(figsize=(7.1, 3.8))
    ax.fill_between(mean.index, mean - spread, mean + spread, color="#DCE6F7", label="Plus or minus one standard deviation")
    ax.plot(mean.index, mean, color=BLUE, marker="o", linewidth=2, label="Mean GHI")
    ax.axhline(fc.DAYLIGHT, color=INK, linestyle="--", linewidth=1, label=f"Daylight threshold ({fc.DAYLIGHT:.0f} {UNIT})")
    ax.set(title="GHI by hour of day, Port of Spain, 2019 to 2024", xlabel="Hour of day (local solar time)",
           ylabel=f"GHI ({UNIT})", xticks=range(0, 24, 2), ylim=(0, None))
    ax.legend(frameon=False, loc="upper left")
    save(fig, "fig7_ghi_climatology.png")
    print(f"Wrote 7 figures to {FIGURES}")


if __name__ == "__main__":
    main()
