"""The seven report figures.

Design decisions that apply to all seven, and why:

**Two categorical hues, blue for the LSTM and orange for persistence, assigned by
entity and never by rank.** The pair was checked with a colour-vision-deficiency
validator rather than by eye: worst-pair deltaE is 24.7 under protanopia against a
target of 8, and both clear 3:1 contrast against the figure surface. Observed GHI is
drawn in neutral ink rather than a third hue, because it is the ground truth, not a
competing model, and that distinction should be visible at a glance.

**Line style and marker shape carry the same information as colour.** An IEEE report
is frequently printed in greyscale, and a figure whose series are distinguishable only
by hue becomes unreadable there. The LSTM is a solid line with round markers,
persistence is dashed with square markers, so every series survives the loss of colour.

**Recessive chrome.** Solid hairline gridlines one shade off the surface, no top or
right spine, no boxed legend. The data is the only loud thing in the frame.

**One axis per figure, always.** No figure here plots two different scales against each
other, which would invent a correlation the data does not contain.
"""

from __future__ import annotations

import argparse
import logging

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")   # headless backend, chosen before pyplot is imported
import matplotlib.dates as mdates
import matplotlib.pyplot as plt

from src.baseline import MODEL_NAME as PERSISTENCE
from src.config import Config, configure_logging
from src.evaluate import load_predictions, target_timestamps
from src.preprocess import load_features
from src.train import MODEL_NAME as LSTM
from src.windowing import SplitWindows
from src.windowing import load as load_windows

LOGGER = logging.getLogger(__name__)

# Palette, validated for colour-vision deficiency. See the module docstring.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e5e4e0"

SERIES_STYLE: dict[str, dict[str, object]] = {
    LSTM: {
        "color": "#2a78d6",
        "linestyle": "-",
        "marker": "o",
        "label": "LSTM",
    },
    PERSISTENCE: {
        "color": "#eb6834",
        "linestyle": (0, (4, 2)),
        "marker": "s",
        "label": "Persistence",
    },
}
ACTUAL_STYLE: dict[str, object] = {
    "color": INK,
    "linestyle": "-",
    "marker": None,
    "label": "Observed",
}

LINE_WIDTH = 2.0
MARKER_SIZE = 5.5
GHI_LABEL = "GHI (W/m$^2$)"


def _style_axes(ax: plt.Axes, ylabel: str, xlabel: str, title: str) -> None:
    """Apply the shared recessive chrome to one axes.

    Args:
        ax: the axes to style.
        ylabel: y axis label, including units.
        xlabel: x axis label.
        title: axes title.
    """
    ax.set_facecolor(SURFACE)
    ax.set_title(title, color=INK, fontsize=11, pad=10)
    ax.set_xlabel(xlabel, color=INK_SECONDARY, fontsize=10)
    ax.set_ylabel(ylabel, color=INK_SECONDARY, fontsize=10)
    ax.grid(True, color=GRID, linewidth=0.8, linestyle="-")
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=INK_SECONDARY, labelsize=9, length=3, width=0.8)


def _new_figure(width: float, height: float) -> tuple[plt.Figure, plt.Axes]:
    """Create a figure and axes on the shared surface colour."""
    figure, ax = plt.subplots(figsize=(width, height))
    figure.patch.set_facecolor(SURFACE)
    return figure, ax


def _headroom(ax: plt.Axes, factor: float = 1.20) -> None:
    """Add space above the data so an unboxed legend cannot collide with it.

    Args:
        ax: the axes to expand.
        factor: multiplier applied to the current upper limit.

    An unboxed legend has no opaque background to hide behind, so the room has to be
    made in the axes rather than painted over the marks.
    """
    ax.set_ylim(bottom=0, top=ax.get_ylim()[1] * factor)


def _legend(ax: plt.Axes, **kwargs: object) -> None:
    """Add an unboxed legend, which every figure with two or more series carries."""
    ax.legend(
        frameon=False,
        fontsize=9,
        labelcolor=INK_SECONDARY,
        **kwargs,
    )


def _save(config: Config, figure: plt.Figure, filename: str) -> None:
    """Write a figure at the configured resolution and close it.

    Args:
        config: loaded pipeline configuration.
        figure: the figure to write.
        filename: exact filename required by the spec, including the .png suffix.
    """
    figure.tight_layout()
    path = config.dir_path("figures_dir") / filename
    figure.savefig(
        path,
        dpi=int(config.figures["dpi"]),
        facecolor=figure.get_facecolor(),
        bbox_inches="tight",
    )
    plt.close(figure)
    LOGGER.info("wrote %s", path)


def select_days(
    config: Config,
    features: pd.DataFrame,
    window: SplitWindows,
    method: str,
) -> tuple[int, int]:
    """Choose the clearest and the most heavily clouded test forecast day.

    Args:
        config: loaded pipeline configuration.
        features: the feature frame in physical units.
        window: the test split's windows.
        method: "cloud" to rank by observed cloud amount, "clearness" to rank by a
            clearness index derived from GHI alone.

    Returns:
        Sample indices of the clearest and cloudiest windows.

    Raises:
        ValueError: for an unknown method.

    WHY a clearness fallback exists: day selection ranks by CLOUD_AMT, but Gate 0 drops
    any column exceeding five percent missing, so that column is not guaranteed to be
    present. The fallback ranks by each day's daylight GHI relative to the climatological
    mean for that day of year, which measures the same thing the figure is after, how
    much cloud attenuated the day, using only the target variable.
    """
    if method not in ("cloud", "clearness"):
        raise ValueError(
            f"unknown day selection method {method!r}, expected 'cloud' or 'clearness'"
        )

    threshold = float(config.evaluation["daylight_threshold"])
    daylight = window.target_actual > threshold

    if method == "cloud" and "CLOUD_AMT" in features.columns:
        cloud = features["CLOUD_AMT"].to_numpy(dtype="float64")[window.target_positions]
        score = np.where(daylight, cloud, np.nan)
        per_day = np.nanmean(score, axis=1)
        LOGGER.info("ranking test days by observed daylight mean CLOUD_AMT")
        return int(np.nanargmin(per_day)), int(np.nanargmax(per_day))

    # Clearness index: each day's daylight mean GHI divided by the climatological
    # daylight mean for the same day of year, so the seasonal envelope cancels out.
    times = pd.DatetimeIndex(target_timestamps(features, window).ravel())
    day_of_year = times.dayofyear.to_numpy().reshape(window.target_actual.shape)

    ghi = features[config.target_column]
    daylight_ghi = ghi[ghi > threshold]
    climatology = daylight_ghi.groupby(daylight_ghi.index.dayofyear).mean()
    expected = np.where(
        daylight,
        climatology.reindex(day_of_year.ravel())
        .to_numpy()
        .reshape(day_of_year.shape),
        np.nan,
    )
    observed = np.where(daylight, window.target_actual, np.nan)
    clearness = np.nanmean(observed, axis=1) / np.nanmean(expected, axis=1)
    LOGGER.info("ranking test days by clearness index, CLOUD_AMT unavailable or not requested")
    return int(np.nanargmax(clearness)), int(np.nanargmin(clearness))


def figure_day(
    config: Config,
    features: pd.DataFrame,
    window: SplitWindows,
    predictions: dict[str, np.ndarray],
    sample: int,
    filename: str,
    descriptor: str,
) -> None:
    """Plot observed and both forecasts across one 24 hour test day.

    Args:
        config: loaded pipeline configuration.
        features: the feature frame in physical units.
        window: the test split's windows.
        predictions: predictions in W/m^2 keyed by model name.
        sample: which window to plot.
        filename: exact output filename.
        descriptor: short phrase describing the day, used in the title.
    """
    times = pd.DatetimeIndex(target_timestamps(features, window)[sample])
    actual = window.target_actual[sample]

    figure, ax = _new_figure(7.2, 3.9)
    ax.plot(
        times,
        actual,
        linewidth=1.6,
        color=ACTUAL_STYLE["color"],
        linestyle=ACTUAL_STYLE["linestyle"],
        label=ACTUAL_STYLE["label"],
        zorder=3,
    )
    for model in (PERSISTENCE, LSTM):
        style = SERIES_STYLE[model]
        ax.plot(
            times,
            predictions[model][sample],
            linewidth=LINE_WIDTH,
            color=style["color"],
            linestyle=style["linestyle"],
            marker=style["marker"],
            markersize=MARKER_SIZE,
            markeredgecolor=SURFACE,
            markeredgewidth=1.0,
            label=style["label"],
            zorder=4,
        )

    origin = window.origins[sample]
    _style_axes(
        ax,
        ylabel=GHI_LABEL,
        xlabel="Hour of day (local solar time)",
        title=(
            f"Next-day forecast on the {descriptor} test day, "
            f"{times[0].date()}\nForecast issued at {origin:%Y-%m-%d %H:%M}"
        ),
    )
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    # Ticks on multiples of three from midnight, so the final tick reads as the 00:00 that
    # closes the horizon rather than as an unexplained repeat of the first label.
    ax.xaxis.set_major_locator(mdates.HourLocator(byhour=range(0, 24, 3)))
    _headroom(ax, 1.14)
    _legend(ax, loc="upper left")
    _save(config, figure, filename)


def figure_error_by_hour(config: Config, metrics: pd.DataFrame) -> None:
    """Plot daylight MAE against hour of day for both models.

    Args:
        config: loaded pipeline configuration.
        metrics: the long-format metrics frame.

    A line is used rather than grouped bars because hour of day is an ordered continuum,
    and the shape of the curve across the day is the thing worth reading.
    """
    figure, ax = _new_figure(7.2, 3.9)

    for model in (PERSISTENCE, LSTM):
        style = SERIES_STYLE[model]
        rows = metrics[
            (metrics["model"] == model)
            & (metrics["split"] == "test")
            & (metrics["metric"].str.startswith("mae_masked_hour_"))
        ].copy()
        rows["hour"] = rows["metric"].str.removeprefix("mae_masked_hour_").astype(int)
        rows = rows.sort_values("hour")
        ax.plot(
            rows["hour"],
            rows["value"],
            linewidth=LINE_WIDTH,
            color=style["color"],
            linestyle=style["linestyle"],
            marker=style["marker"],
            markersize=MARKER_SIZE,
            markeredgecolor=SURFACE,
            markeredgewidth=1.0,
            label=style["label"],
        )

    _style_axes(
        ax,
        ylabel="MAE (W/m$^2$)",
        xlabel="Hour of day (local solar time)",
        title="Forecast error by hour of day, daylight hours of the test year",
    )
    _headroom(ax, 1.16)
    _legend(ax, loc="upper left")
    _save(config, figure, "fig3_error_by_hour.png")


def figure_seasonal(config: Config, metrics: pd.DataFrame) -> None:
    """Plot daylight RMSE by calendar month for both models.

    Args:
        config: loaded pipeline configuration.
        metrics: the long-format metrics frame.

    Grouped bars suit twelve discrete months, with a surface-coloured gap between
    adjacent bars rather than a drawn border.
    """
    figure, ax = _new_figure(7.6, 4.0)
    month_labels = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun",
        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    ]
    bar_width = 0.38

    for offset, model in ((-0.5, PERSISTENCE), (0.5, LSTM)):
        style = SERIES_STYLE[model]
        rows = metrics[
            (metrics["model"] == model)
            & (metrics["split"] == "test")
            & (metrics["metric"].str.startswith("rmse_masked_month_"))
        ].copy()
        rows["month"] = rows["metric"].str.removeprefix("rmse_masked_month_").astype(int)
        rows = rows.sort_values("month")
        ax.bar(
            rows["month"] + offset * bar_width,
            rows["value"],
            width=bar_width,
            color=style["color"],
            label=style["label"],
            edgecolor=SURFACE,
            linewidth=1.2,
        )

    wet_months = sorted(int(m) for m in config.evaluation["wet_season_months"])
    ax.axvspan(
        min(wet_months) - 0.5,
        max(wet_months) + 0.5,
        color=GRID,
        alpha=0.55,
        zorder=0,
        label="Wet season",
    )

    _style_axes(
        ax,
        ylabel="RMSE (W/m$^2$)",
        xlabel="Month of the test year",
        title="Forecast error by month, daylight hours of the test year",
    )
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(month_labels)
    ax.set_xlim(0.4, 12.6)
    _headroom(ax, 1.24)
    _legend(ax, loc="upper left", ncol=3)
    _save(config, figure, "fig4_seasonal.png")


def figure_scatter(
    config: Config,
    window: SplitWindows,
    predictions: dict[str, np.ndarray],
    metrics: pd.DataFrame,
) -> None:
    """Plot predicted against observed GHI for the masked test hours.

    Args:
        config: loaded pipeline configuration.
        window: the test split's windows.
        predictions: predictions in W/m^2 keyed by model name.
        metrics: the long-format metrics frame, for the annotated RMSE.

    Two panels rather than one shared scatter, because 4229 points per model overplot
    into an unreadable mass when superimposed. Small multiples keep the comparison honest:
    both panels share identical axes and an identical identity line.
    """
    threshold = float(config.evaluation["daylight_threshold"])
    mask = window.target_actual > threshold
    observed = window.target_actual[mask]

    limit = float(
        max(observed.max(), *(predictions[m][mask].max() for m in (PERSISTENCE, LSTM)))
    )
    upper = 100.0 * np.ceil(limit / 100.0)

    figure, axes = plt.subplots(1, 2, figsize=(9.0, 4.4), sharex=True, sharey=True)
    figure.patch.set_facecolor(SURFACE)

    for ax, model in zip(axes, (PERSISTENCE, LSTM)):
        style = SERIES_STYLE[model]
        ax.plot(
            [0, upper],
            [0, upper],
            linewidth=1.2,
            color=INK_SECONDARY,
            linestyle=(0, (4, 3)),
            label="Identity",
            zorder=3,
        )
        ax.scatter(
            observed,
            predictions[model][mask],
            s=7,
            color=style["color"],
            alpha=0.22,
            linewidths=0,
            label=style["label"],
            zorder=2,
        )
        rmse = float(
            metrics[
                (metrics["model"] == model)
                & (metrics["split"] == "test")
                & (metrics["season"] == "all")
                & (metrics["metric"] == "rmse_masked")
            ]["value"].iloc[0]
        )
        # A single selective annotation per panel, not a label on every point.
        ax.annotate(
            f"RMSE {rmse:.1f} W/m$^2$",
            xy=(0.04, 0.94),
            xycoords="axes fraction",
            fontsize=9,
            color=INK_SECONDARY,
        )
        _style_axes(
            ax,
            ylabel=f"Predicted {GHI_LABEL}",
            xlabel=f"Observed {GHI_LABEL}",
            title=str(style["label"]),
        )
        ax.set_xlim(0, upper)
        ax.set_ylim(0, upper)
        ax.set_aspect("equal", adjustable="box")
        _legend(ax, loc="lower right")

    axes[1].set_ylabel("")
    figure.suptitle(
        "Predicted against observed GHI, daylight test hours",
        color=INK,
        fontsize=11,
    )
    _save(config, figure, "fig5_scatter.png")


def figure_loss_curve(config: Config) -> None:
    """Plot training and validation loss per epoch.

    Args:
        config: loaded pipeline configuration.

    Both series are scaled-space MSE on one axis, so no second scale is introduced. The
    epoch that early stopping selected is marked, because that is the model actually
    evaluated and a reader should be able to see which epoch produced the reported numbers.
    """
    history = pd.read_csv(config.path("loss_log"))
    best_epoch = int(history.loc[history["val_loss"].idxmin(), "epoch"])

    figure, ax = _new_figure(7.2, 3.9)
    ax.plot(
        history["epoch"],
        history["train_loss"],
        linewidth=LINE_WIDTH,
        color=SERIES_STYLE[LSTM]["color"],
        linestyle="-",
        marker="o",
        markersize=MARKER_SIZE,
        markeredgecolor=SURFACE,
        markeredgewidth=1.0,
        label="Training loss",
    )
    ax.plot(
        history["epoch"],
        history["val_loss"],
        linewidth=LINE_WIDTH,
        color=SERIES_STYLE[PERSISTENCE]["color"],
        linestyle=(0, (4, 2)),
        marker="s",
        markersize=MARKER_SIZE,
        markeredgecolor=SURFACE,
        markeredgewidth=1.0,
        label="Validation loss",
    )
    ax.axvline(
        best_epoch,
        color=INK_SECONDARY,
        linewidth=1.0,
        linestyle=(0, (2, 2)),
        label=f"Best epoch ({best_epoch}), weights restored",
    )

    _style_axes(
        ax,
        ylabel="MSE (scaled GHI)",
        xlabel="Epoch",
        title="Training and validation loss, early stopping on validation loss",
    )
    _headroom(ax, 1.26)
    _legend(ax, loc="upper right")
    _save(config, figure, "fig6_loss_curve.png")


def figure_climatology(config: Config, features: pd.DataFrame) -> None:
    """Plot mean GHI by hour of day across the whole dataset.

    Args:
        config: loaded pipeline configuration.
        features: the full feature frame in physical units.

    A single series, so there is no legend box: the title names what is plotted. The
    daylight mask threshold is drawn as a reference, because it is the line that decides
    which hours enter every headline metric in the project.
    """
    ghi = features[config.target_column]
    hourly = ghi.groupby(ghi.index.hour).agg(["mean", "std"])
    threshold = float(config.evaluation["daylight_threshold"])

    figure, ax = _new_figure(7.2, 3.9)
    ax.fill_between(
        hourly.index,
        np.clip(hourly["mean"] - hourly["std"], 0.0, None),
        hourly["mean"] + hourly["std"],
        color=SERIES_STYLE[LSTM]["color"],
        alpha=0.16,
        linewidth=0,
        label="Plus or minus one standard deviation",
    )
    ax.plot(
        hourly.index,
        hourly["mean"],
        linewidth=LINE_WIDTH,
        color=SERIES_STYLE[LSTM]["color"],
        marker="o",
        markersize=MARKER_SIZE,
        markeredgecolor=SURFACE,
        markeredgewidth=1.0,
        label="Mean GHI",
    )
    ax.axhline(
        threshold,
        color=INK_SECONDARY,
        linewidth=1.0,
        linestyle=(0, (4, 3)),
        label=f"Daylight mask threshold ({threshold:.0f} W/m$^2$)",
    )

    years = f"{features.index.year.min()} to {features.index.year.max()}"
    _style_axes(
        ax,
        ylabel=GHI_LABEL,
        xlabel="Hour of day (local solar time)",
        title=f"GHI climatology by hour of day, {config.location['name']}, {years}",
    )
    ax.set_xticks(range(0, 24, 2))
    ax.set_xlim(-0.5, 23.5)
    _headroom(ax, 1.22)
    _legend(ax, loc="upper left")
    _save(config, figure, "fig7_ghi_climatology.png")


def run(
    config: Config,
    day_selection: str = "cloud",
    splits: dict[str, SplitWindows] | None = None,
    features: pd.DataFrame | None = None,
    metrics: pd.DataFrame | None = None,
) -> list[str]:
    """Produce all seven figures.

    Args:
        config: loaded pipeline configuration.
        day_selection: "cloud" or "clearness", how figures 1 and 2 pick their days.
        splits: optional pre-loaded windows.
        features: optional pre-loaded feature frame.
        metrics: optional pre-loaded metrics frame.

    Returns:
        The filenames written, in figure order.
    """
    windows = load_windows(config) if splits is None else splits
    frame = load_features(config) if features is None else features
    metric_frame = (
        pd.read_csv(config.path("metrics_csv")) if metrics is None else metrics
    )
    predictions = load_predictions(config, "test")
    test = windows["test"]

    clearest, cloudiest = select_days(config, frame, test, day_selection)
    LOGGER.info(
        "clearest test day is sample %d (%s), cloudiest is sample %d (%s)",
        clearest,
        test.origins[clearest].date(),
        cloudiest,
        test.origins[cloudiest].date(),
    )

    figure_day(
        config, frame, test, predictions, clearest, "fig1_clear_day.png", "clearest"
    )
    figure_day(
        config, frame, test, predictions, cloudiest, "fig2_cloudy_day.png", "most clouded"
    )
    figure_error_by_hour(config, metric_frame)
    figure_seasonal(config, metric_frame)
    figure_scatter(config, test, predictions, metric_frame)
    figure_loss_curve(config)
    figure_climatology(config, frame)

    return [
        "fig1_clear_day.png",
        "fig2_cloudy_day.png",
        "fig3_error_by_hour.png",
        "fig4_seasonal.png",
        "fig5_scatter.png",
        "fig6_loss_curve.png",
        "fig7_ghi_climatology.png",
    ]


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for `python -m src.analysis`."""
    parser = argparse.ArgumentParser(description="Produce the seven report figures.")
    parser.add_argument(
        "--config", default=None, help="path to config.yaml, defaults to project root"
    )
    parser.add_argument(
        "--day-selection",
        choices=("cloud", "clearness"),
        default="cloud",
        help="how figures 1 and 2 rank test days, defaults to observed cloud amount",
    )
    args = parser.parse_args(argv)

    configure_logging()
    config = Config.load(args.config)
    written = run(config, day_selection=args.day_selection)

    print(f"Wrote {len(written)} figures to {config.dir_path('figures_dir')}:")
    for name in written:
        print(f"  {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
