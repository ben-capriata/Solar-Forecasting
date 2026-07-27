"""Supervised window construction, chronological splits, and the Gate 2 leakage audit.

One sample of this forecasting problem is:

    encoder input    hours t-23 through t, past-observed ghi, weather, and calendar
    decoder input    hours t+1 through t+24, calendar features only
    target           ghi at hours t+1 through t+24

The decoder restriction is the whole scientific claim of the project. A model given
tomorrow's observed cloud cover is not forecasting, it is interpolating a weather feed
it does not have. Calendar features are admissible because a clock needs no forecast.

**Split assignment is by whole-window containment, not by row year.** A window whose
origin sits on 2024-01-01 00:00 draws its encoder from 2023, the validation year. Assigning
it to test by its origin's year would make the test set overlap the validation set, and the
spec's own Gate 2 requirement that the split index ranges be disjoint would be false. So a
window joins a split only when its encoder input and its target both lie entirely inside
that split's years. The cost is two windows per year boundary, which is counted and logged.

Persistence needs no separate data source, which is a pleasing consequence of the layout:
the prediction for t+h is the observation at t+h-24, and for h from 1 to 24 that is exactly
hours t-23 through t, the encoder window. See `src.baseline`.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.config import Config, configure_logging
from src.gates import check, require
from src.preprocess import ScalerBundle, load_features, load_scalers

LOGGER = logging.getLogger(__name__)

GATE_NAME = "GATE 2, leakage audit"

# Tolerance for the content alignment checks in Gate 2. Windows are stored as float32
# while the scalers compute in float64, so an exact equality check would fail on
# representation alone rather than on any real misalignment.
ALIGNMENT_TOLERANCE = 1e-5


@dataclass(frozen=True)
class ScaledFrame:
    """The scaled feature matrix and target, computed once and shared by every consumer.

    Attributes:
        matrix: scaled model input features, shape (rows, features).
        ghi: scaled target, shape (rows,).
        actual_ghi: the target in W/m^2, shape (rows,).
        feature_columns: ordered feature names matching `matrix` columns.
        calendar_names: the calendar features present after Gate 0.
        calendar_columns: positions of those calendar features within `matrix`.

    WHY this type exists: building the three splits and then auditing them in Gate 2 needs
    the same scaled matrix over all 52608 rows four separate times. Transforming once and
    passing the result removes three redundant full-matrix transforms per run, and it puts
    the calendar-column derivation in one place instead of repeating the same list
    comprehension at four call sites where they could drift apart.
    """

    matrix: np.ndarray
    ghi: np.ndarray
    actual_ghi: np.ndarray
    feature_columns: list[str]
    calendar_names: list[str]
    calendar_columns: list[int]

    @classmethod
    def build(
        cls, config: Config, features: pd.DataFrame, bundle: ScalerBundle
    ) -> ScaledFrame:
        """Apply the train-fitted scalers to the whole feature frame.

        Args:
            config: loaded pipeline configuration.
            features: unscaled feature frame in physical units.
            bundle: scalers fitted on training rows only.

        Returns:
            A frozen ScaledFrame covering every row of `features`.

        Scaling every row with train-only statistics is correct and is not leakage: the
        statistics flow from train to the other splits, never the reverse. Gate 2 asserts
        that direction by checking the recorded fit index.
        """
        feature_columns = list(bundle.feature_columns)
        calendar_names = [c for c in config.calendar_features if c in features.columns]
        return cls(
            matrix=bundle.feature_scaler.transform(
                features[feature_columns].to_numpy(dtype="float64")
            ),
            ghi=bundle.target_scaler.transform(
                features[[config.target_column]].to_numpy(dtype="float64")
            ).ravel(),
            actual_ghi=features[config.target_column].to_numpy(dtype="float64"),
            feature_columns=feature_columns,
            calendar_names=calendar_names,
            calendar_columns=[feature_columns.index(name) for name in calendar_names],
        )


@dataclass(frozen=True)
class SplitWindows:
    """Windowed tensors for one chronological split.

    Attributes:
        name: split name, one of "train", "val", "test".
        encoder: scaled past-observed inputs, shape (samples, encoder_hours, features).
        decoder: scaled calendar inputs for the target window, shape (samples, horizon, calendar).
        target: scaled ghi to predict, shape (samples, horizon).
        target_actual: the same ghi in W/m^2, shape (samples, horizon).
        origins: forecast-origin timestamp per sample, shape (samples,).
        encoder_positions: row positions of each encoder hour, shape (samples, encoder_hours).
        target_positions: row positions of each target hour, shape (samples, horizon).
    """

    name: str
    encoder: np.ndarray
    decoder: np.ndarray
    target: np.ndarray
    target_actual: np.ndarray
    origins: pd.DatetimeIndex
    encoder_positions: np.ndarray
    target_positions: np.ndarray

    @property
    def n_samples(self) -> int:
        return int(self.encoder.shape[0])


def _split_position_bounds(features: pd.DataFrame, years: list[int]) -> tuple[int, int]:
    """First and last row position belonging to a set of calendar years.

    Args:
        features: the full feature frame, chronologically sorted.
        years: calendar years making up one split.

    Returns:
        Inclusive pair of row positions.

    Raises:
        ValueError: if those years do not form one contiguous block of rows, which would
            mean the split is not a single chronological period and containment logic
            based on bounds would be wrong.
    """
    positions = np.flatnonzero(features.index.year.isin(years))
    if positions.size == 0:
        raise ValueError(f"no rows found for years {years}")
    first, last = int(positions[0]), int(positions[-1])
    if last - first + 1 != positions.size:
        raise ValueError(
            f"years {years} do not form a contiguous block of rows, "
            f"spanning positions {first} to {last} but holding {positions.size} rows"
        )
    return first, last


def build_split(
    config: Config,
    features: pd.DataFrame,
    scaled: ScaledFrame,
    split: str,
) -> SplitWindows:
    """Construct every admissible window for one split.

    Args:
        config: loaded pipeline configuration.
        features: unscaled feature frame from preprocess, hourly and sorted.
        scaled: the shared scaled matrix and target, built once per run.
        split: one of "train", "val", "test".

    Returns:
        The split's windowed tensors.

    Stride is taken from config: dense for training so the model sees every hour as an
    origin, and 24 with a midnight origin for validation and test so that evaluation
    measures the operational task, one day-ahead forecast issued once per day.
    """
    encoder_hours = int(config.windowing["encoder_hours"])
    horizon = int(config.windowing["horizon_hours"])
    origin_hour = int(config.windowing["origin_hour"])
    is_training = split == "train"

    first_pos, last_pos = _split_position_bounds(features, config.split_years(split))
    candidates = np.arange(first_pos, last_pos + 1)

    if is_training:
        # Every hour is a candidate origin, thinned by the configured stride.
        stride = int(config.windowing["train_stride"])
        candidates = candidates[::stride]
    else:
        # Evaluation origins are pinned to the configured origin hour, which by itself
        # spaces them exactly 24 hours apart. Applying eval_stride on top would keep only
        # every 24th day, so the configured stride is asserted rather than reapplied.
        stride = int(config.windowing["eval_stride"])
        candidates = candidates[
            features.index.hour.to_numpy()[candidates] == origin_hour
        ]
        spacing = np.unique(np.diff(candidates)) if candidates.size > 1 else np.array([])
        if spacing.size and not np.array_equal(spacing, np.array([stride])):
            raise ValueError(
                f"split {split} origins are spaced {spacing} hours apart, expected {stride}"
            )

    encoder_offsets = np.arange(-(encoder_hours - 1), 1)
    target_offsets = np.arange(1, horizon + 1)

    encoder_positions = candidates[:, None] + encoder_offsets[None, :]
    target_positions = candidates[:, None] + target_offsets[None, :]

    # Containment: both the encoder input and the target must lie inside this split.
    contained = (encoder_positions[:, 0] >= first_pos) & (
        target_positions[:, -1] <= last_pos
    )
    n_boundary_dropped = int((~contained).sum())

    candidates = candidates[contained]
    encoder_positions = encoder_positions[contained]
    target_positions = target_positions[contained]

    encoder = scaled.matrix[encoder_positions]
    decoder = scaled.matrix[target_positions][:, :, scaled.calendar_columns]
    target = scaled.ghi[target_positions]
    target_actual = scaled.actual_ghi[target_positions]

    # Any window touching an unfilled gap is dropped whole. A window with an imputed
    # hole is a window whose loss is partly measured against invented data.
    finite = (
        np.isfinite(encoder).all(axis=(1, 2))
        & np.isfinite(decoder).all(axis=(1, 2))
        & np.isfinite(target).all(axis=1)
    )
    n_nan_dropped = int((~finite).sum())

    LOGGER.info(
        "split %-5s stride %-2d: %d windows kept, %d dropped for crossing a split boundary, "
        "%d dropped for touching a NaN",
        split,
        stride,
        int(finite.sum()),
        n_boundary_dropped,
        n_nan_dropped,
    )

    return SplitWindows(
        name=split,
        encoder=encoder[finite].astype("float32"),
        decoder=decoder[finite].astype("float32"),
        target=target[finite].astype("float32"),
        target_actual=target_actual[finite].astype("float64"),
        origins=features.index[candidates[finite]],
        encoder_positions=encoder_positions[finite],
        target_positions=target_positions[finite],
    )


def gate2(
    config: Config,
    features: pd.DataFrame,
    bundle: ScalerBundle,
    scaled: ScaledFrame,
    splits: dict[str, SplitWindows],
    echo: bool = True,
) -> None:
    """Run the Gate 2 leakage audit over the constructed windows.

    Args:
        config: loaded pipeline configuration.
        features: the unscaled feature frame the windows were cut from.
        bundle: the fitted scalers, carrying their fit provenance.
        scaled: the shared scaled matrix the windows were cut from.
        splits: constructed windows keyed by split name.
        echo: whether to print the verdict to stdout. Gate 1 sets this False for the two
            sandbox reruns so the console reports each gate once.

    Raises:
        GateFailure: on any leakage or alignment violation.

    The checks, and what each would catch:

        Ordering. Every sample's latest input hour precedes its earliest target hour.
            Catches an off-by-one that lets a sample read its own first target hour.
        Encoder content. Encoder values equal the scaled observations at the encoder
            timestamps. This is the strongest available check, because it proves the
            encoder holds the actual observed past rather than merely being shaped right.
            An index shuffle or a sign error in the offsets fails here.
        Target content. Same proof for the target.
        Decoder content and names. The decoder equals the calendar features at the target
            timestamps, and its width equals the calendar feature count. Catches any
            weather channel leaking into the known-future input.
        Disjointness. No hour is touched by two splits. This is what containment buys.
        Chronology. Splits appear in time order, train then validation then test.
        Scaler provenance. The recorded fit index ends before the validation year starts.
        Finiteness. No NaN reaches the model.
    """
    report_path = config.path("validation_report")
    calendar = scaled.calendar_names

    results: list[tuple[bool, str]] = []
    touched: dict[str, tuple[pd.Timestamp, pd.Timestamp]] = {}
    touched_sets: dict[str, set[int]] = {}

    for name in config.split_names:
        window = splits[name]
        encoder_times = features.index.to_numpy()[window.encoder_positions]
        target_times = features.index.to_numpy()[window.target_positions]

        results.append(
            check(
                f"{name}: every sample's last input hour precedes its first target hour",
                bool((encoder_times.max(axis=1) < target_times.min(axis=1)).all()),
                f"{window.n_samples} samples verified",
            )
        )

        expected_encoder = scaled.matrix[window.encoder_positions]
        encoder_error = float(
            np.abs(expected_encoder - window.encoder.astype("float64")).max()
        )
        results.append(
            check(
                f"{name}: encoder values equal scaled observations at the encoder hours",
                encoder_error < ALIGNMENT_TOLERANCE,
                f"max absolute difference {encoder_error:.3e}",
            )
        )

        expected_target = scaled.ghi[window.target_positions]
        target_error = float(
            np.abs(expected_target - window.target.astype("float64")).max()
        )
        results.append(
            check(
                f"{name}: target values equal scaled ghi at the target hours",
                target_error < ALIGNMENT_TOLERANCE,
                f"max absolute difference {target_error:.3e}",
            )
        )

        expected_decoder = scaled.matrix[window.target_positions][:, :, scaled.calendar_columns]
        decoder_error = float(
            np.abs(expected_decoder - window.decoder.astype("float64")).max()
        )
        results.append(
            check(
                f"{name}: decoder values equal calendar features at the target hours",
                decoder_error < ALIGNMENT_TOLERANCE,
                f"max absolute difference {decoder_error:.3e}",
            )
        )
        results.append(
            check(
                f"{name}: decoder carries only the {len(calendar)} calendar features",
                window.decoder.shape[2] == len(calendar),
                f"width {window.decoder.shape[2]}, calendar features {calendar}",
            )
        )

        finite = (
            np.isfinite(window.encoder).all()
            and np.isfinite(window.decoder).all()
            and np.isfinite(window.target).all()
        )
        results.append(
            check(f"{name}: no NaN reaches the model", bool(finite), "all finite")
        )

        positions = set(window.encoder_positions.ravel().tolist()) | set(
            window.target_positions.ravel().tolist()
        )
        touched_sets[name] = positions
        span = features.index[sorted(positions)]
        touched[name] = (span.min(), span.max())

    # Disjointness across every pair of splits.
    names = list(config.split_names)
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            overlap = touched_sets[left] & touched_sets[right]
            results.append(
                check(
                    f"{left} and {right} touch no common hour",
                    not overlap,
                    f"{len(overlap)} shared hours",
                )
            )

    # Chronology across consecutive splits.
    for left, right in zip(names, names[1:]):
        results.append(
            check(
                f"{left} ends before {right} begins",
                touched[left][1] < touched[right][0],
                f"{left} ends {touched[left][1]}, {right} begins {touched[right][0]}",
            )
        )

    first_val_year = min(config.val_years)
    validation_start = pd.Timestamp(f"{first_val_year}-01-01")
    results.append(
        check(
            "scaler fit index ends before the validation year begins",
            bundle.fit_index_max < validation_start,
            f"fit spanned {bundle.fit_index_min} to {bundle.fit_index_max} "
            f"over {bundle.n_fit_rows} rows, validation starts {validation_start}",
        )
    )
    results.append(
        check(
            "scaler fit index begins in the first training year",
            bundle.fit_index_min.year == min(config.train_years),
            bundle.fit_index_min.year,
        )
    )

    require(
        GATE_NAME,
        all(ok for ok, _ in results),
        [line for _, line in results],
        report_path,
        echo=echo,
    )


def build_upperbound_decoders(
    config: Config,
    features: pd.DataFrame,
    bundle: ScalerBundle,
    splits: dict[str, SplitWindows],
) -> dict[str, np.ndarray]:
    """Decoder inputs that also carry tomorrow's OBSERVED weather. Section 13 only.

    Args:
        config: loaded pipeline configuration.
        features: unscaled feature frame the windows were cut from.
        bundle: fitted scalers.
        splits: constructed windows keyed by split name.

    Returns:
        Replacement decoder arrays keyed by split name, shape
        (samples, horizon, weather + calendar features).

    WHY this exists and why it is quarantined. Giving the model tomorrow's observed weather
    is not forecasting, it is measuring how much a perfect numerical weather prediction feed
    would be worth. That upper bound is genuinely useful to quantify for a grid operator
    deciding whether to buy an NWP subscription. It is also exactly the leakage the primary
    model's Gate 2 forbids, which is why it lives in its own function, produces its own
    `lstm_upperbound` label, and must never be presented as the headline result.

    GHI itself is deliberately excluded. Handing the model tomorrow's irradiance would be
    handing it the answer, not a weather feed.
    """
    scaled = ScaledFrame.build(config, features, bundle)
    calendar = scaled.calendar_names
    weather = [
        c
        for c in scaled.feature_columns
        if c not in calendar and c != config.target_column
    ]
    selected = [scaled.feature_columns.index(name) for name in weather + calendar]

    LOGGER.warning(
        "building UPPER BOUND decoders with observed future weather %s, "
        "this variant is not a forecast and must not be reported as the headline model",
        weather,
    )
    return {
        name: scaled.matrix[window.target_positions][:, :, selected].astype("float32")
        for name, window in splits.items()
    }


def save(config: Config, splits: dict[str, SplitWindows]) -> None:
    """Persist the windows and an origin index.

    Args:
        config: loaded pipeline configuration.
        splits: constructed windows keyed by split name.

    The origin index is a separate CSV rather than a column inside the npz because the
    seasonal and hour-of-day analyses need to join metrics back to calendar dates, and a
    CSV is the format a human can open to check that join.
    """
    arrays: dict[str, np.ndarray] = {}
    index_rows: list[dict[str, object]] = []

    for name, window in splits.items():
        arrays[f"{name}_encoder"] = window.encoder
        arrays[f"{name}_decoder"] = window.decoder
        arrays[f"{name}_target"] = window.target
        arrays[f"{name}_target_actual"] = window.target_actual
        arrays[f"{name}_origins"] = window.origins.to_numpy().astype("datetime64[ns]")
        arrays[f"{name}_encoder_positions"] = window.encoder_positions
        arrays[f"{name}_target_positions"] = window.target_positions
        index_rows.extend(
            {"split": name, "sample": position, "origin": origin.isoformat()}
            for position, origin in enumerate(window.origins)
        )

    windows_path = config.path("windows")
    np.savez_compressed(windows_path, **arrays)
    LOGGER.info("wrote %d arrays to %s", len(arrays), windows_path)

    index_path = config.path("window_index")
    pd.DataFrame(index_rows).to_csv(index_path, index=False)
    LOGGER.info("wrote %d window origins to %s", len(index_rows), index_path)


def load(config: Config) -> dict[str, SplitWindows]:
    """Read persisted windows back into SplitWindows objects.

    Args:
        config: loaded pipeline configuration.

    Returns:
        Windows keyed by split name.
    """
    with np.load(config.path("windows"), allow_pickle=False) as data:
        return {
            name: SplitWindows(
                name=name,
                encoder=data[f"{name}_encoder"],
                decoder=data[f"{name}_decoder"],
                target=data[f"{name}_target"],
                target_actual=data[f"{name}_target_actual"],
                origins=pd.DatetimeIndex(data[f"{name}_origins"]),
                encoder_positions=data[f"{name}_encoder_positions"],
                target_positions=data[f"{name}_target_positions"],
            )
            for name in config.split_names
        }


def run(
    config: Config,
    features: pd.DataFrame | None = None,
    echo: bool = True,
) -> dict[str, SplitWindows]:
    """Build windows for all splits, run Gate 2, and persist the result.

    Args:
        config: loaded pipeline configuration.
        features: optional pre-loaded feature frame. Gate 1 passes its synthetic
            features so the fixture travels the identical code path.
        echo: whether Gate 2 prints its verdict to stdout.

    Returns:
        Windows keyed by split name.
    """
    frame = load_features(config) if features is None else features
    bundle = load_scalers(config)
    scaled = ScaledFrame.build(config, frame, bundle)

    splits = {
        name: build_split(config, frame, scaled, name) for name in config.split_names
    }
    gate2(config, frame, bundle, scaled, splits, echo=echo)
    save(config, splits)
    return splits


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for `python -m src.windowing`."""
    parser = argparse.ArgumentParser(
        description="Build supervised windows and run the Gate 2 leakage audit."
    )
    parser.add_argument(
        "--config", default=None, help="path to config.yaml, defaults to project root"
    )
    args = parser.parse_args(argv)

    configure_logging()
    config = Config.load(args.config)
    splits = run(config)

    for name in config.split_names:
        window = splits[name]
        print(
            f"{name:<5} {window.n_samples:>6} windows, "
            f"encoder {window.encoder.shape[1:]}, decoder {window.decoder.shape[1:]}, "
            f"origins {window.origins.min()} to {window.origins.max()}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
