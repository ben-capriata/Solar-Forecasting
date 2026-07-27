"""Deterministic synthetic fixture and the Gate 1 pipeline validation.

GATE 1 answers a question that must be settled before a single real-data training run is
launched: does the machinery work at all?

The logic is diagnostic separation. If the pipeline is exercised end to end on a series
whose correct answer is known by construction, and it fails, the fault is in the pipeline,
not the model. If it succeeds and the real run then produces poor skill, the poor skill is
a genuine result about Port of Spain irradiance rather than a silent bug in the window
builder. Without this gate those two outcomes are indistinguishable, and a term project can
lose a week to that ambiguity.

The fixture is a clipped sinusoid: a half-cosine bell across daylight hours, a 24 hour
period, and a peak that is modulated annually. Two variants are generated. The noise-free
variant is where persistence must achieve near-zero error, because a smooth periodic series
really is almost identical from one day to the next. The noisy variant is what the model
trains on, so the training loop has something non-trivial to reduce.

Gate 1 runs inside a sandbox directory. Its features, scalers, windows and checkpoints are
written under the configured sandbox prefix so that a gate run can never overwrite the real
experiment's artefacts. Only the validation report is shared, because that is the one file
that should carry all three verdicts for a single run.
"""

from __future__ import annotations

import argparse
import copy
import logging

import numpy as np
import pandas as pd

from src import preprocess, train, windowing
from src.baseline import persistence_forecast
from src.config import Config, configure_logging
from src.gates import check, require

LOGGER = logging.getLogger(__name__)

GATE_NAME = "GATE 1, synthetic pipeline validation"

HOURS_PER_DAY = 24
SOLAR_NOON_HOUR = 12.0
DAYS_PER_YEAR = 365.25


def sandbox_config(config: Config) -> Config:
    """Return a config whose artefact paths point inside the Gate 1 sandbox.

    Args:
        config: the real pipeline configuration.

    Returns:
        A copy with every path redirected under the sandbox prefix, and with the synthetic
        stride and epoch overrides applied.

    WHY every path is redirected, the validation report included: a Gate 1 run that
    overwrote the real features.parquet or the real checkpoint would corrupt the experiment
    it exists to protect. The report is redirected too, because Gate 1 internally reruns
    Gate 0 and Gate 2 on the fixture, and those sandbox verdicts appearing in the real
    report would make it read as though Gate 2 had run three times. The sandbox report stays
    on disk for inspection, and Gate 1's own verdict is written to the real report by the
    caller, which holds the unsandboxed config.
    """
    raw = copy.deepcopy(config.raw)
    prefix = str(raw["synthetic"]["sandbox_prefix"])

    raw["paths"] = {key: f"{prefix}/{value}" for key, value in raw["paths"].items()}
    raw["windowing"]["train_stride"] = int(raw["synthetic"]["train_stride"])
    raw["training"]["max_epochs"] = int(raw["synthetic"]["max_epochs"])

    return Config(raw=raw, source=config.source)


def generate(config: Config, with_noise: bool = True) -> pd.DataFrame:
    """Generate the synthetic fixture with the same index and schema as the real data.

    Args:
        config: loaded pipeline configuration, supplying the date range and shape parameters.
        with_noise: whether to add the small seeded noise term. False gives the smooth
            variant that persistence must nearly solve exactly.

    Returns:
        DataFrame of shape (hours, source_columns) indexed hourly over the configured date
        range, using NASA POWER column names so the fixture travels the identical
        preprocessing code path as real data. GHI is in W/m^2, the other columns carry the
        units their real counterparts do.

    The GHI shape is `peak(day) * cos(pi * (hour - 12) / (2 * half_width))`, clipped at zero.
    At solar noon the cosine is one, and it reaches zero `half_width` hours either side, so
    the daylight window is twice the half width. That gives a smooth bell with a hard night
    floor, which is the essential character of a real tropical GHI day.
    """
    settings = config.synthetic
    index = pd.date_range(
        f"{config.start_date} 00:00", f"{config.end_date} 23:00", freq="h"
    )

    hour = index.hour.to_numpy(dtype="float64")
    day_of_year = index.dayofyear.to_numpy(dtype="float64")
    annual_phase = 2.0 * np.pi * day_of_year / DAYS_PER_YEAR
    diurnal_phase = 2.0 * np.pi * hour / HOURS_PER_DAY

    half_width = float(settings["daylight_half_width_hours"])
    peak = float(settings["peak_base"]) + float(
        settings["peak_annual_amplitude"]
    ) * np.sin(annual_phase)

    bell = np.cos(np.pi * (hour - SOLAR_NOON_HOUR) / (2.0 * half_width))
    ghi = np.clip(peak * bell, 0.0, None)
    # Hours beyond the daylight half width would fold back up through the cosine, so the
    # bell is masked to the daylight span explicitly rather than relying on the clip.
    ghi = np.where(np.abs(hour - SOLAR_NOON_HOUR) <= half_width, ghi, 0.0)

    # Deterministic companion weather. These exist so the fixture matches the real schema
    # and the encoder has the same width. They are plausible in range but make no claim to
    # physical fidelity, which is why the gate never checks them.
    temperature = 27.0 + 2.0 * np.sin(annual_phase) + 3.0 * np.sin(diurnal_phase)
    humidity = 78.0 + 8.0 * np.sin(annual_phase) - 6.0 * np.sin(diurnal_phase)
    wind_speed = 4.0 + 1.5 * np.sin(diurnal_phase)
    cloud_amount = 45.0 + 20.0 * np.sin(annual_phase)
    precipitable_water = 4.0 + 1.0 * np.sin(annual_phase)

    if with_noise:
        # A single seeded generator, so the fixture is byte-identical across runs.
        rng = np.random.default_rng(config.seed)
        noise_std = float(settings["noise_std"])
        ghi = np.clip(ghi + rng.normal(0.0, noise_std, size=ghi.shape), 0.0, None)
        temperature = temperature + rng.normal(0.0, 0.2, size=temperature.shape)
        humidity = humidity + rng.normal(0.0, 1.0, size=humidity.shape)
        wind_speed = np.clip(wind_speed + rng.normal(0.0, 0.2, size=wind_speed.shape), 0.0, None)
        cloud_amount = np.clip(
            cloud_amount + rng.normal(0.0, 2.0, size=cloud_amount.shape), 0.0, 100.0
        )
        precipitable_water = np.clip(
            precipitable_water + rng.normal(0.0, 0.1, size=precipitable_water.shape),
            0.0,
            None,
        )

    frame = pd.DataFrame(
        {
            "ALLSKY_SFC_SW_DWN": ghi,
            "T2M": temperature,
            "RH2M": humidity,
            "WS10M": wind_speed,
            "CLOUD_AMT": cloud_amount,
            "PW": precipitable_water,
        },
        index=index,
    )

    available = [c for c in config.data["source_columns"] if c in frame.columns]
    LOGGER.info(
        "generated synthetic series: %d rows, noise %s, ghi peak %.1f W/m^2",
        len(frame),
        "on" if with_noise else "off",
        float(frame["ALLSKY_SFC_SW_DWN"].max()),
    )
    return frame[available]


def gate1(config: Config) -> None:
    """Run the full pipeline on the synthetic fixture and assert it behaves.

    Args:
        config: the real pipeline configuration. A sandboxed copy is derived internally.

    Raises:
        GateFailure: if any assertion fails, meaning the pipeline is broken.

    The assertions:

        Window shapes. Encoder, decoder and target must have the widths and lengths the
            configuration declares. Catches an offset or transposition error.
        Finiteness. No NaN may reach the model from any split.
        Persistence on the noise-free variant. A smooth periodic series is nearly identical
            day to day, so lag-24 persistence must achieve near-zero daylight error. A
            large error here means the lag alignment is wrong.
        Learning. The model must reduce validation loss below the configured threshold
            within a few epochs on a series it can in principle fit exactly. Failure here
            means the training loop is not learning, whatever the real data later shows.
    """
    report_path = config.path("validation_report")
    sandbox = sandbox_config(config)
    encoder_hours = int(config.windowing["encoder_hours"])
    horizon = int(config.windowing["horizon_hours"])
    threshold = float(config.evaluation["daylight_threshold"])
    settings = config.synthetic

    results: list[tuple[bool, str]] = []

    # Part one, the noise-free variant. Persistence should nearly solve it.
    smooth = generate(config, with_noise=False)
    smooth_features = preprocess.run(sandbox, raw=smooth)
    smooth_windows = windowing.run(sandbox, features=smooth_features, echo=False)
    smooth_test = smooth_windows["test"]

    smooth_persistence = persistence_forecast(sandbox, smooth_features, smooth_test)
    mask = smooth_test.target_actual > threshold
    persistence_rmse = float(
        np.sqrt(
            np.mean((smooth_persistence[mask] - smooth_test.target_actual[mask]) ** 2)
        )
    )
    results.append(
        check(
            f"persistence daylight RMSE on the noise-free fixture below "
            f"{settings['persistence_rmse_max']} W/m^2",
            persistence_rmse < float(settings["persistence_rmse_max"]),
            f"{persistence_rmse:.4f} W/m^2",
        )
    )

    # Part two, the noisy variant. The model must actually learn.
    noisy = generate(config, with_noise=True)
    noisy_features = preprocess.run(sandbox, raw=noisy)
    noisy_windows = windowing.run(sandbox, features=noisy_features, echo=False)
    bundle = preprocess.load_scalers(sandbox)

    n_encoder_features = len(bundle.feature_columns)
    n_calendar = len([c for c in config.calendar_features if c in noisy_features.columns])

    for name, window in noisy_windows.items():
        results.append(
            check(
                f"{name}: encoder shape is (samples, {encoder_hours}, {n_encoder_features})",
                window.encoder.shape[1:] == (encoder_hours, n_encoder_features),
                window.encoder.shape,
            )
        )
        results.append(
            check(
                f"{name}: decoder shape is (samples, {horizon}, {n_calendar})",
                window.decoder.shape[1:] == (horizon, n_calendar),
                window.decoder.shape,
            )
        )
        results.append(
            check(
                f"{name}: target shape is (samples, {horizon})",
                window.target.shape[1:] == (horizon,),
                window.target.shape,
            )
        )
        results.append(
            check(
                f"{name}: no NaN reaches the model",
                bool(
                    np.isfinite(window.encoder).all()
                    and np.isfinite(window.decoder).all()
                    and np.isfinite(window.target).all()
                ),
                "all finite",
            )
        )

    artifacts = train.train_model(
        sandbox,
        noisy_windows,
        bundle,
        max_epochs=int(settings["max_epochs"]),
        # Patience equal to the epoch cap means early stopping cannot fire inside the gate,
        # so the model is always given its full few epochs to demonstrate learning.
        patience=int(settings["max_epochs"]),
        announce=False,
    )
    loss_threshold = float(settings["loss_threshold"])
    results.append(
        check(
            f"validation MSE falls below {loss_threshold} within "
            f"{settings['max_epochs']} epochs",
            artifacts.best_val_loss < loss_threshold,
            f"{artifacts.best_val_loss:.6f} at epoch {artifacts.best_epoch}",
        )
    )
    results.append(
        check(
            "training loss decreased from the first epoch to the best epoch",
            artifacts.history["train_loss"].iloc[-1]
            < artifacts.history["train_loss"].iloc[0],
            f"{artifacts.history['train_loss'].iloc[0]:.6f} to "
            f"{artifacts.history['train_loss'].iloc[-1]:.6f}",
        )
    )

    # Gate 1 reran Gate 0 and Gate 2 against the fixture inside the sandbox. Point a reader
    # at those verdicts rather than duplicating them into this report.
    results.append(
        (
            True,
            f"[ok  ] sandbox artefacts and the fixture's own Gate 0 and Gate 2 verdicts: "
            f"{sandbox.path('validation_report')}",
        )
    )

    require(
        GATE_NAME,
        all(ok for ok, _ in results),
        [line for _, line in results],
        report_path,
    )


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for `python -m src.synthetic`."""
    parser = argparse.ArgumentParser(
        description="Generate the synthetic fixture and run Gate 1 pipeline validation."
    )
    parser.add_argument(
        "--config", default=None, help="path to config.yaml, defaults to project root"
    )
    args = parser.parse_args(argv)

    configure_logging()
    config = Config.load(args.config)
    gate1(config)
    print("Gate 1 complete: the pipeline learns a series whose answer is known.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
