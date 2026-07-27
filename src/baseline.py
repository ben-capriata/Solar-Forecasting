"""Twenty-four hour persistence baseline.

The forecast rule is one line: the prediction for hour t+h is the observation at hour
t+h-24. Tomorrow will be like today.

WHY this is the right baseline rather than a strawman. GHI at a fixed site is dominated
by the diurnal cycle, so the 24-hour lag is by far the strongest naive predictor
available. An autocorrelation of the Port of Spain series peaks sharply at lags 24 and 48.
Persistence therefore already captures sunrise, sunset, and the seasonal envelope for
free, and it does so without a single fitted parameter. A model that cannot beat it has
learned nothing worth reporting, which is exactly what makes the skill score informative.

Implementation note. Because the encoder window is hours t-23 through t and the horizon is
h from 1 to 24, the lag-24 lookup t+h-24 lands on precisely those same hours. Persistence
is therefore the encoder window's own GHI channel. This module does not exploit that
shortcut. It looks the values up from the unscaled series by timestamp, as the spec
requires, and then asserts agreement with the encoder window as a cross-check. Two
independent derivations that agree are worth more than one that is merely plausible.
"""

from __future__ import annotations

import argparse
import logging

import numpy as np
import pandas as pd

from src.config import Config, configure_logging
from src.preprocess import load_features
from src.windowing import SplitWindows
from src.windowing import load as load_windows

LOGGER = logging.getLogger(__name__)

MODEL_NAME = "persistence"


def persistence_forecast(
    config: Config,
    features: pd.DataFrame,
    window: SplitWindows,
) -> np.ndarray:
    """Predict each target hour with the observation 24 hours earlier.

    Args:
        config: loaded pipeline configuration.
        features: unscaled feature frame in physical units, hourly index in LST.
        window: the split's windows, supplying the forecast origins to predict from.

    Returns:
        Predictions in W/m^2 of shape (samples, horizon_hours), the identical shape and
        units the LSTM produces, so `src.evaluate` handles both through one code path.

    Raises:
        ValueError: if any looked-up hour is missing from the series, which would mean the
            window builder admitted a window it should have dropped.
    """
    horizon = int(config.windowing["horizon_hours"])
    ghi = features[config.target_column]

    # Offsets t+h-24 for h in 1..24, which is -23..0 hours relative to the origin.
    lag_offsets = pd.to_timedelta(np.arange(1, horizon + 1) - horizon, unit="h")
    lookup_times = window.origins.to_numpy()[:, None] + lag_offsets.to_numpy()[None, :]

    flat = pd.DatetimeIndex(lookup_times.ravel())
    values = ghi.reindex(flat).to_numpy(dtype="float64")
    missing = int(np.isnan(values).sum())
    if missing:
        raise ValueError(
            f"persistence lookup found {missing} missing hours for split {window.name}. "
            "The window builder should have dropped any window touching a gap."
        )

    predictions = values.reshape(len(window.origins), horizon)

    # Cross-check against the independently built encoder window. These are two separate
    # derivations of the same quantity, so disagreement means one of them is wrong.
    encoder_ghi = features[config.target_column].to_numpy(dtype="float64")[
        window.encoder_positions
    ]
    deviation = float(np.abs(predictions - encoder_ghi).max())
    if deviation > 0.0:
        raise ValueError(
            f"persistence forecast disagrees with the encoder GHI channel by up to "
            f"{deviation:.6f} W/m^2 for split {window.name}. The lag-24 lookup and the "
            "encoder window should be identical by construction."
        )

    LOGGER.info(
        "persistence forecast for split %-5s: %s, range %.1f to %.1f W/m^2",
        window.name,
        predictions.shape,
        predictions.min(),
        predictions.max(),
    )
    return predictions


def run(
    config: Config,
    features: pd.DataFrame | None = None,
    splits: dict[str, SplitWindows] | None = None,
) -> dict[str, np.ndarray]:
    """Produce and persist persistence forecasts for the evaluation splits.

    Args:
        config: loaded pipeline configuration.
        features: optional pre-loaded feature frame.
        splits: optional pre-loaded windows.

    Returns:
        Predictions in W/m^2 keyed by split name.

    Only validation and test are forecast. Persistence has nothing to learn, so a
    training-split forecast would serve no purpose.
    """
    frame = load_features(config) if features is None else features
    windows = load_windows(config) if splits is None else splits

    predictions: dict[str, np.ndarray] = {}
    for split in ("val", "test"):
        forecast = persistence_forecast(config, frame, windows[split])
        path = config.prediction_path(MODEL_NAME, split)
        np.save(path, forecast)
        LOGGER.info("wrote %s predictions to %s", split, path)
        predictions[split] = forecast

    return predictions


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for `python -m src.baseline`."""
    parser = argparse.ArgumentParser(
        description="Produce 24-hour persistence forecasts for the evaluation splits."
    )
    parser.add_argument(
        "--config", default=None, help="path to config.yaml, defaults to project root"
    )
    args = parser.parse_args(argv)

    configure_logging()
    config = Config.load(args.config)
    predictions = run(config)

    for split, forecast in predictions.items():
        print(f"Persistence {split}: {forecast.shape[0]} forecasts of {forecast.shape[1]} hours.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
