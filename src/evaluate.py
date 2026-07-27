"""Daylight-masked evaluation of every model on identical test windows.

The daylight mask is the central methodological choice here, so it is worth stating why
it exists before reading the code.

Roughly half of any 24 hour window at this latitude is night, when GHI is zero and every
model predicts it correctly. Including those hours in an average does not measure
forecasting skill, it dilutes it: a model could halve its reported MAE by doing nothing
except getting night right, which it gets right for free. Masking to hours where the
observed GHI exceeds 20 W/m^2 restricts the average to the hours a grid operator actually
schedules against. The mask is computed from the observed series, not from either model's
output, so it selects exactly the same hours for every model and cannot favour one.

Unmasked variants are computed and reported alongside, clearly labelled, as the spec's hard
rules require. Masked is the headline. One exception is stated explicitly rather than
quietly dropped: MAPE has no unmasked counterpart, because unmasked hours include exact
zeros at night and a percentage error against zero is undefined.

The metrics file is long format, one number per row, columns model, split, season, metric,
value. Hour-of-day and per-month breakdowns are encoded in the metric name, since the
prescribed schema has no column for them.
"""

from __future__ import annotations

import argparse
import logging

import numpy as np
import pandas as pd

from src.baseline import MODEL_NAME as PERSISTENCE
from src.config import Config, configure_logging
from src.preprocess import load_features
from src.train import MODEL_NAME as LSTM
from src.train import UPPER_BOUND_NAME as UPPER_BOUND
from src.windowing import SplitWindows
from src.windowing import load as load_windows

LOGGER = logging.getLogger(__name__)

HEADLINE_SPLIT = "test"


def load_predictions(config: Config, split: str) -> dict[str, np.ndarray]:
    """Load every model's predictions for one split.

    Args:
        config: loaded pipeline configuration.
        split: split name.

    Returns:
        Predictions in W/m^2 keyed by model name, shape (samples, horizon) each. Models
        whose prediction file is absent are simply omitted, which is how the optional
        upper bound of spec section 13 stays optional.

    Raises:
        FileNotFoundError: if neither the baseline nor the primary model has been run.
    """
    found: dict[str, np.ndarray] = {}
    for model in (PERSISTENCE, LSTM, UPPER_BOUND):
        path = config.prediction_path(model, split)
        if path.exists():
            found[model] = np.load(path)
            LOGGER.info("loaded %s %s predictions %s", model, split, found[model].shape)

    if PERSISTENCE not in found or LSTM not in found:
        raise FileNotFoundError(
            f"split {split} needs both {PERSISTENCE} and {LSTM} predictions. "
            "Run python -m src.baseline and python -m src.train first."
        )
    return found


def target_timestamps(features: pd.DataFrame, window: SplitWindows) -> np.ndarray:
    """Timestamp of every target hour.

    Args:
        features: the feature frame the windows were cut from.
        window: one split's windows.

    Returns:
        Datetime array of shape (samples, horizon_hours).
    """
    return features.index.to_numpy()[window.target_positions]


def _metrics(
    predicted: np.ndarray, actual: np.ndarray, mask: np.ndarray, with_mape: bool
) -> dict[str, float]:
    """Error metrics over the selected cells.

    Args:
        predicted: predictions in W/m^2.
        actual: observations in W/m^2, same shape.
        mask: boolean selector of the same shape.
        with_mape: whether to include MAPE. Only valid when the mask excludes zeros.

    Returns:
        Mapping of metric name to value. MAE and RMSE are in W/m^2, MAPE in percent.
        An empty selection yields NaN rather than raising, so a month or hour with no
        daylight simply reports no value.
    """
    if not mask.any():
        empty = {"mae": float("nan"), "rmse": float("nan")}
        if with_mape:
            empty["mape"] = float("nan")
        return empty

    error = predicted[mask] - actual[mask]
    result = {
        "mae": float(np.abs(error).mean()),
        "rmse": float(np.sqrt(np.mean(error**2))),
    }
    if with_mape:
        result["mape"] = float((np.abs(error) / actual[mask]).mean() * 100.0)
    return result


def evaluate_split(
    config: Config,
    split: str,
    window: SplitWindows,
    features: pd.DataFrame,
    predictions: dict[str, np.ndarray],
) -> pd.DataFrame:
    """Compute every reported metric for one split.

    Args:
        config: loaded pipeline configuration.
        split: split name.
        window: the split's windows, supplying observed GHI and target positions.
        features: the feature frame, supplying target timestamps.
        predictions: predictions in W/m^2 keyed by model name.

    Returns:
        Long-format frame with columns model, split, season, metric, value.

    Raises:
        ValueError: if any model's predictions do not match the observed array shape, which
            would mean the models were not evaluated on identical windows.
    """
    threshold = float(config.evaluation["daylight_threshold"])
    wet_months = set(int(m) for m in config.evaluation["wet_season_months"])

    actual = window.target_actual
    times = pd.DatetimeIndex(target_timestamps(features, window).ravel())
    months = times.month.to_numpy().reshape(actual.shape)
    hours = times.hour.to_numpy().reshape(actual.shape)

    for model, forecast in predictions.items():
        if forecast.shape != actual.shape:
            raise ValueError(
                f"model {model} produced {forecast.shape} for split {split} but the "
                f"observations are {actual.shape}. Both models must be evaluated on "
                "identical windows."
            )

    daylight = actual > threshold
    everything = np.ones_like(daylight, dtype=bool)
    wet = np.isin(months, sorted(wet_months))

    season_masks = {"all": everything, "wet": wet, "dry": ~wet}

    rows: list[dict[str, object]] = []

    # Persistence RMSE per group is needed as the skill score denominator, so it is
    # computed first and reused. Each group's skill uses that group's own denominator,
    # never the overall one, otherwise a seasonal skill score would be meaningless.
    persistence_rmse: dict[tuple[str, bool], float] = {}
    for season, season_mask in season_masks.items():
        for masked in (True, False):
            selector = season_mask & daylight if masked else season_mask
            persistence_rmse[(season, masked)] = _metrics(
                predictions[PERSISTENCE], actual, selector, with_mape=False
            )["rmse"]

    for model, forecast in predictions.items():
        for season, season_mask in season_masks.items():
            masked_selector = season_mask & daylight
            masked = _metrics(forecast, actual, masked_selector, with_mape=True)
            unmasked = _metrics(forecast, actual, season_mask, with_mape=False)

            values: dict[str, float] = {
                "mae_masked": masked["mae"],
                "rmse_masked": masked["rmse"],
                "mape_masked": masked["mape"],
                "mae_unmasked": unmasked["mae"],
                "rmse_unmasked": unmasked["rmse"],
                "n_hours_masked": float(masked_selector.sum()),
                "n_hours_total": float(season_mask.sum()),
                "n_forecast_days": float(actual.shape[0]),
            }
            for label, is_masked in (("masked", True), ("unmasked", False)):
                denominator = persistence_rmse[(season, is_masked)]
                numerator = masked["rmse"] if is_masked else unmasked["rmse"]
                values[f"skill_score_{label}"] = (
                    float("nan")
                    if not np.isfinite(denominator) or denominator == 0.0
                    else 1.0 - numerator / denominator
                )

            rows.extend(
                {
                    "model": model,
                    "split": split,
                    "season": season,
                    "metric": metric,
                    "value": value,
                }
                for metric, value in values.items()
            )

        # Hour-of-day breakdown, daylight hours only. Encoded in the metric name because
        # the prescribed schema has no hour column.
        for hour in range(int(config.windowing["horizon_hours"])):
            selector = (hours == hour) & daylight
            if not selector.any():
                continue
            hourly = _metrics(forecast, actual, selector, with_mape=False)
            rows.extend(
                {
                    "model": model,
                    "split": split,
                    "season": "all",
                    "metric": f"{name}_masked_hour_{hour:02d}",
                    "value": value,
                }
                for name, value in hourly.items()
            )

        # Per-month breakdown, daylight hours only, for the seasonal figure.
        for month in range(1, 13):
            selector = (months == month) & daylight
            if not selector.any():
                continue
            monthly = _metrics(forecast, actual, selector, with_mape=False)
            rows.extend(
                {
                    "model": model,
                    "split": split,
                    "season": "all",
                    "metric": f"{name}_masked_month_{month:02d}",
                    "value": value,
                }
                for name, value in monthly.items()
            )

    LOGGER.info(
        "split %s: %d daylight hours of %d total across %d forecast days, %d metric rows",
        split,
        int(daylight.sum()),
        daylight.size,
        actual.shape[0],
        len(rows),
    )
    return pd.DataFrame(rows)


def lookup(metrics: pd.DataFrame, model: str, split: str, season: str, metric: str) -> float:
    """Read one value out of the long-format metrics frame.

    Args:
        metrics: the long-format metrics frame.
        model: model name.
        split: split name.
        season: season label, one of "all", "wet", "dry".
        metric: metric name.

    Returns:
        The value, or NaN when the row is absent.
    """
    match = metrics[
        (metrics["model"] == model)
        & (metrics["split"] == split)
        & (metrics["season"] == season)
        & (metrics["metric"] == metric)
    ]
    return float("nan") if match.empty else float(match["value"].iloc[0])


def format_summary(config: Config, metrics: pd.DataFrame) -> str:
    """Render the human-readable results table.

    Args:
        config: loaded pipeline configuration.
        metrics: the long-format metrics frame.

    Returns:
        A box-drawn table as a single string, ending in one closing line.

    The spec asks for a printout that is pleasant to read, and no more than that. A box
    table and one friendly line, nothing further.
    """
    split = HEADLINE_SPLIT
    models = [
        model
        for model in (PERSISTENCE, LSTM, UPPER_BOUND)
        if model in set(metrics["model"])
    ]
    test_year = ", ".join(str(year) for year in config.test_years)
    days = int(lookup(metrics, PERSISTENCE, split, "all", "n_forecast_days"))
    masked_hours = int(lookup(metrics, PERSISTENCE, split, "all", "n_hours_masked"))
    total_hours = int(lookup(metrics, PERSISTENCE, split, "all", "n_hours_total"))
    threshold = float(config.evaluation["daylight_threshold"])

    width = 78
    lines: list[str] = []
    lines.append("┌" + "─" * width + "┐")
    lines.append("│" + f"  Next-day GHI forecast, {config.location['name']}".ljust(width) + "│")
    lines.append(
        "│"
        + f"  Held-out test year {test_year}, {days} forecast days, "
        f"origin at local midnight".ljust(width)
        + "│"
    )
    lines.append(
        "│"
        + f"  Headline metrics are daylight-masked: {masked_hours} of {total_hours} "
        f"hours above {threshold:.0f} W/m^2".ljust(width)
        + "│"
    )
    lines.append("├" + "─" * width + "┤")

    header = (
        f"│ {'Model':<16}{'MAE':>10}{'RMSE':>10}{'MAPE':>10}{'Skill':>12}"
        f"{'RMSE unmasked':>18} │"
    )
    lines.append(header)
    lines.append(
        "│ "
        + f"{'':<16}{'W/m^2':>10}{'W/m^2':>10}{'%':>10}{'vs persist':>12}{'W/m^2':>18}"
        + " │"
    )
    lines.append("├" + "─" * width + "┤")

    for model in models:
        mae = lookup(metrics, model, split, "all", "mae_masked")
        rmse = lookup(metrics, model, split, "all", "rmse_masked")
        mape = lookup(metrics, model, split, "all", "mape_masked")
        skill = lookup(metrics, model, split, "all", "skill_score_masked")
        rmse_unmasked = lookup(metrics, model, split, "all", "rmse_unmasked")
        lines.append(
            f"│ {model:<16}{mae:>10.2f}{rmse:>10.2f}{mape:>10.2f}{skill:>12.3f}"
            f"{rmse_unmasked:>18.2f} │"
        )

    lines.append("├" + "─" * width + "┤")
    lines.append("│" + "  Daylight-masked RMSE by season, W/m^2".ljust(width) + "│")
    lines.append(
        "│ "
        + f"{'Model':<16}{'wet (Jun-Dec)':>18}{'dry (Jan-May)':>18}{'wet skill':>12}"
        f"{'dry skill':>12}"
        + " │"
    )
    for model in models:
        wet = lookup(metrics, model, split, "wet", "rmse_masked")
        dry = lookup(metrics, model, split, "dry", "rmse_masked")
        wet_skill = lookup(metrics, model, split, "wet", "skill_score_masked")
        dry_skill = lookup(metrics, model, split, "dry", "skill_score_masked")
        lines.append(
            f"│ {model:<16}{wet:>18.2f}{dry:>18.2f}{wet_skill:>12.3f}{dry_skill:>12.3f} │"
        )

    lines.append("└" + "─" * width + "┘")

    skill = lookup(metrics, LSTM, split, "all", "skill_score_masked")
    if np.isfinite(skill) and skill > 0:
        closing = (
            f"The LSTM beat persistence by {skill:.1%} of its RMSE. "
            "The sun kept its schedule, and so did the pipeline."
        )
    else:
        closing = (
            "Persistence held its ground this time, which is a real result and worth "
            "reporting plainly. The gates all passed, so the finding is about the sky, "
            "not the code."
        )
    lines.append("")
    lines.append(closing)
    lines.append("")
    lines.append(
        "Note: MAPE is reported for daylight hours only. An unmasked MAPE is undefined "
        "here, because night GHI is exactly zero."
    )
    return "\n".join(lines)


def run(
    config: Config,
    splits: dict[str, SplitWindows] | None = None,
    features: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Evaluate every available model and write metrics and the summary.

    Args:
        config: loaded pipeline configuration.
        splits: optional pre-loaded windows.
        features: optional pre-loaded feature frame.

    Returns:
        The long-format metrics frame that was written to disk.
    """
    windows = load_windows(config) if splits is None else splits
    frame = load_features(config) if features is None else features

    pieces = [
        evaluate_split(config, split, windows[split], frame, load_predictions(config, split))
        for split in ("val", "test")
    ]
    metrics = pd.concat(pieces, ignore_index=True)

    metrics_path = config.path("metrics_csv")
    metrics.to_csv(metrics_path, index=False)
    LOGGER.info("wrote %d metric rows to %s", len(metrics), metrics_path)

    summary = format_summary(config, metrics)
    summary_path = config.path("summary_txt")
    summary_path.write_text(summary + "\n", encoding="utf-8")
    LOGGER.info("wrote summary to %s", summary_path)

    return metrics


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for `python -m src.evaluate`."""
    parser = argparse.ArgumentParser(
        description="Compute daylight-masked metrics for every model on identical windows."
    )
    parser.add_argument(
        "--config", default=None, help="path to config.yaml, defaults to project root"
    )
    args = parser.parse_args(argv)

    configure_logging()
    config = Config.load(args.config)
    metrics = run(config)

    print()
    print(format_summary(config, metrics))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
