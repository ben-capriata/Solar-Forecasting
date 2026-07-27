"""Gap handling, calendar features, and train-only scaling.

Input is the Gate 0 approved hourly frame from `src.ingest`. Output is
`data/processed/features.parquet` in physical units, plus a pickled scaler bundle.

Two design points are worth reading before the code, because both are leakage
defences that the handoff spec's stated gates would not have caught.

**Interpolation runs inside split segments, never across them.** Filling a gap that
straddles 2022-12-31 23:00 to 2023-01-01 01:00 would compute a training row from a
validation value and a validation row from a training value. That is leakage in both
directions, it is invisible to a timestamp-ordering assertion, and it is entirely
avoided by interpolating each split's rows independently.

**features.parquet holds unscaled physical units.** Scaling is a separate persisted
artefact applied downstream in `src.windowing`. Keeping the two apart means the
baseline and the evaluator can both read true W/m^2 without an inverse transform,
and it means the scaler's fitted statistics exist as an auditable file rather than
as transient state inside one function.
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.config import Config, configure_logging
from src.ingest import load_raw

LOGGER = logging.getLogger(__name__)

HOURS_PER_DAY = 24
DAYS_PER_YEAR = 365.25   # includes the leap-year quarter day, so the annual cycle does not drift


@dataclass(frozen=True)
class ScalerBundle:
    """Fitted scalers plus the provenance Gate 2 needs to audit them.

    Attributes:
        feature_scaler: StandardScaler over every model input feature, fit on train rows.
        target_scaler: StandardScaler over ghi alone, fit on the same train rows.
        feature_columns: ordered input feature names the feature_scaler expects.
        fit_index_min: earliest timestamp the fit actually saw.
        fit_index_max: latest timestamp the fit actually saw.
        n_fit_rows: how many rows the fit actually saw.

    WHY the fit index bounds are stored: the spec asks Gate 2 to assert that the
    scaler fit call received only train rows. An assertion cannot inspect a call that
    already returned, so the call records what it was given, and the gate checks the
    record. Without this the "no leakage" claim would rest on reading the source, which
    is exactly the kind of unfalsifiable check the gates exist to replace.
    """

    feature_scaler: StandardScaler
    target_scaler: StandardScaler
    feature_columns: list[str]
    fit_index_min: pd.Timestamp
    fit_index_max: pd.Timestamp
    n_fit_rows: int


def assert_no_fill_values(frame: pd.DataFrame, fill_value: float) -> None:
    """Assert the upstream sentinel was already converted to NaN at ingest.

    Args:
        frame: raw hourly frame.
        fill_value: the POWER sentinel, -999.0.

    Raises:
        ValueError: if any cell still holds the sentinel.

    WHY this is an assertion and not a conversion: ingest owns that conversion. A
    second conversion here would mask an ingest regression rather than surface it.
    """
    offenders = (frame == fill_value).sum()
    total = int(offenders.sum())
    if total:
        columns = offenders[offenders > 0].to_dict()
        raise ValueError(
            f"{total} cells still hold the fill value {fill_value}: {columns}. "
            "src.ingest.to_frame should have replaced these."
        )
    LOGGER.info("confirmed no %s sentinel values survive from ingest", fill_value)


def nan_run_lengths(mask: pd.Series) -> pd.Series:
    """Length of the consecutive NaN run each missing cell belongs to.

    Args:
        mask: boolean series, True where the value is missing.

    Returns:
        Integer series of the same length. Missing cells carry their run length,
        present cells carry 0.
    """
    run_id = mask.ne(mask.shift()).cumsum()
    lengths = mask.groupby(run_id).transform("size")
    return lengths.where(mask, other=0).astype(int)


def interpolate_segment(
    segment: pd.DataFrame,
    columns: Iterable[str],
    max_gap_hours: int,
) -> tuple[pd.DataFrame, int, int]:
    """Linearly fill short gaps inside one split's rows, leaving long gaps as NaN.

    Args:
        segment: contiguous hourly rows belonging to a single split.
        columns: numeric columns to interpolate.
        max_gap_hours: longest run of consecutive NaN that may be filled.

    Returns:
        Triple of the filled segment, the number of gap runs that were filled, and
        the number of gap runs deliberately left as NaN.

    WHY pandas `limit=` is not used: `interpolate(limit=3)` fills the first three
    hours of a five hour gap, leaving a partly synthetic run that no longer looks
    like a gap to anything downstream. The spec wants a long gap left entirely alone
    so the windows touching it can be dropped. So this interpolates without a limit,
    then restores NaN across every run longer than the allowance. `limit_area="inside"`
    prevents extrapolation past the first and last observation of the segment, which
    matters because a segment edge is a split boundary.
    """
    filled = segment.copy()
    runs_filled = 0
    runs_left = 0

    for column in columns:
        original = segment[column]
        missing = original.isna()
        if not missing.any():
            continue

        run_lengths = nan_run_lengths(missing)
        interpolated = original.interpolate(method="linear", limit_area="inside")

        # Restore NaN wherever the run was longer than the allowance, so a long gap
        # stays a recognisable gap rather than becoming partly synthetic data.
        too_long = missing & (run_lengths > max_gap_hours)
        interpolated = interpolated.mask(too_long)

        # Classify each run empirically by whether interpolation actually resolved it.
        # A leading or trailing run has no left or right anchor, so it stays NaN and
        # is correctly counted as left rather than filled.
        run_id = missing.ne(missing.shift()).cumsum()
        for _, run_index in missing[missing].groupby(run_id[missing]).groups.items():
            if interpolated.loc[run_index].isna().any():
                runs_left += 1
            else:
                runs_filled += 1

        filled[column] = interpolated

    return filled, runs_filled, runs_left


def interpolate_by_split(config: Config, frame: pd.DataFrame) -> pd.DataFrame:
    """Interpolate short gaps within each split independently.

    Args:
        config: loaded pipeline configuration.
        frame: raw hourly frame with the sentinel already NaN.

    Returns:
        Frame of identical shape and index with short gaps filled.

    Raises:
        ValueError: if any row does not belong to exactly one configured split.
    """
    max_gap = int(config.data["max_interpolation_gap_hours"])
    numeric_columns = [column for column in frame.columns]

    years = frame.index.year
    assigned = np.zeros(len(frame), dtype=int)
    pieces: list[pd.DataFrame] = []
    total_filled = 0
    total_left = 0

    for split in config.split_names:
        split_years = config.split_years(split)
        mask = years.isin(split_years)
        assigned += mask.astype(int)
        segment = frame.loc[mask]
        if segment.empty:
            continue

        filled, runs_filled, runs_left = interpolate_segment(
            segment, numeric_columns, max_gap
        )
        pieces.append(filled)
        total_filled += runs_filled
        total_left += runs_left
        LOGGER.info(
            "split %-5s years %s: %d rows, %d short gaps filled, %d long gaps left",
            split,
            split_years,
            len(segment),
            runs_filled,
            runs_left,
        )

    unassigned = int((assigned == 0).sum())
    duplicated = int((assigned > 1).sum())
    if unassigned or duplicated:
        raise ValueError(
            f"split coverage is not a partition: {unassigned} rows belong to no split, "
            f"{duplicated} rows belong to more than one. Check config splits."
        )

    LOGGER.info(
        "interpolation totals: %d gap runs filled at or below %dh, %d gap runs left as NaN",
        total_filled,
        max_gap,
        total_left,
    )
    return pd.concat(pieces).sort_index()


def add_calendar_features(config: Config, frame: pd.DataFrame) -> pd.DataFrame:
    """Append the deterministic clock features.

    Args:
        config: loaded pipeline configuration.
        frame: hourly frame indexed in local solar time.

    Returns:
        The frame with hour_sin, hour_cos, doy_sin, doy_cos and wet_season appended.

    WHY sine and cosine pairs rather than a raw hour integer: hour 23 and hour 0 are
    one hour apart in reality but 23 apart as integers. Projecting onto a circle makes
    that adjacency true for the model. Both components are needed, since either one
    alone is ambiguous between two times of day.

    WHY 365.25 for the annual period: the six year span contains two leap years, and a
    365.0 period would let the seasonal phase drift by a day and a half across it.
    """
    result = frame.copy()
    hour = frame.index.hour.to_numpy(dtype="float64")
    day_of_year = frame.index.dayofyear.to_numpy(dtype="float64")
    month = frame.index.month

    result["hour_sin"] = np.sin(2.0 * np.pi * hour / HOURS_PER_DAY)
    result["hour_cos"] = np.cos(2.0 * np.pi * hour / HOURS_PER_DAY)
    result["doy_sin"] = np.sin(2.0 * np.pi * day_of_year / DAYS_PER_YEAR)
    result["doy_cos"] = np.cos(2.0 * np.pi * day_of_year / DAYS_PER_YEAR)

    wet_months = set(int(m) for m in config.evaluation["wet_season_months"])
    result["wet_season"] = month.isin(wet_months).astype("float64")

    LOGGER.info(
        "added calendar features %s, wet season months %s",
        config.calendar_features,
        sorted(wet_months),
    )
    return result


def build_features(config: Config, raw: pd.DataFrame) -> pd.DataFrame:
    """Run the full preprocessing sequence in the order the spec prescribes.

    Args:
        config: loaded pipeline configuration.
        raw: Gate 0 approved hourly frame with POWER column names.

    Returns:
        Feature frame in physical units, indexed hourly in local solar time, with
        ghi first, then the surviving weather columns, then the calendar features.
    """
    fill_value = float(config.api["fill_value"])
    ghi_source = config.data["ghi_source_column"]
    target = config.target_column

    assert_no_fill_values(raw, fill_value)
    interpolated = interpolate_by_split(config, raw)
    renamed = interpolated.rename(columns={ghi_source: target})
    featured = add_calendar_features(config, renamed)

    # Physical floor. Irradiance cannot be negative, and linear interpolation across a
    # dawn gap can undershoot below zero. The same floor is later applied to model
    # predictions, so the model is held to the physics its inputs obey.
    below_zero = int((featured[target] < 0).sum())
    featured[target] = featured[target].clip(lower=0.0)
    if below_zero:
        LOGGER.info("clipped %d negative ghi values to the zero physical floor", below_zero)

    weather = [c for c in config.data["weather_features"] if c in featured.columns]
    dropped = [c for c in config.data["weather_features"] if c not in featured.columns]
    if dropped:
        LOGGER.warning("weather columns absent after Gate 0, excluded here: %s", dropped)

    ordered = [target, *weather, *config.calendar_features]
    return featured[ordered]


def fit_scalers(config: Config, features: pd.DataFrame) -> ScalerBundle:
    """Fit the feature and target scalers on training rows only.

    Args:
        config: loaded pipeline configuration.
        features: unscaled feature frame from `build_features`.

    Returns:
        A ScalerBundle carrying both fitted scalers and the provenance of the fit.

    Raises:
        ValueError: if the training rows contain NaN, which would poison the statistics.

    WHY the target gets its own scaler: predictions come out of the model in scaled
    space and must return to W/m^2 exactly. A single scaler over the whole feature
    matrix would require slicing out the ghi column's mean and variance by position,
    which silently breaks the moment the feature order changes. A dedicated
    single-column scaler makes the inverse transform unambiguous.
    """
    train_mask = features.index.year.isin(config.train_years)
    train_rows = features.loc[train_mask]
    if train_rows.empty:
        raise ValueError(f"no rows found for train years {config.train_years}")

    # Statistics must come from complete rows. A NaN would propagate into mean and
    # variance and from there into every scaled value in every split.
    complete = train_rows.dropna()
    if complete.empty:
        raise ValueError("every training row contains at least one NaN")
    if len(complete) < len(train_rows):
        LOGGER.info(
            "fitting scalers on %d complete training rows of %d (%d hold NaN)",
            len(complete),
            len(train_rows),
            len(train_rows) - len(complete),
        )

    feature_columns = list(features.columns)
    feature_scaler = StandardScaler().fit(complete[feature_columns].to_numpy())
    target_scaler = StandardScaler().fit(
        complete[[config.target_column]].to_numpy()
    )

    bundle = ScalerBundle(
        feature_scaler=feature_scaler,
        target_scaler=target_scaler,
        feature_columns=feature_columns,
        fit_index_min=complete.index.min(),
        fit_index_max=complete.index.max(),
        n_fit_rows=len(complete),
    )
    LOGGER.info(
        "scalers fit on %d rows spanning %s to %s",
        bundle.n_fit_rows,
        bundle.fit_index_min,
        bundle.fit_index_max,
    )
    return bundle


def write_feature_spec(config: Config, features: pd.DataFrame) -> dict[str, object]:
    """Record the feature layout that actually materialised, and persist it.

    Args:
        config: loaded pipeline configuration.
        features: the unscaled feature frame.

    Returns:
        The spec mapping that was written to disk.

    WHY this file exists rather than deriving the lists from config at each use site:
    Gate 0 may drop a weather column, so the realised feature layout is a runtime fact,
    not a configuration intent. Downstream modules read the realised layout, which keeps
    the encoder width, the model's input size, and the leakage audit in agreement.
    """
    calendar = [c for c in config.calendar_features if c in features.columns]
    encoder = list(features.columns)
    spec = {
        "target": config.target_column,
        "encoder_features": encoder,
        "decoder_features": calendar,
        "calendar_features": calendar,
        "weather_features": [
            c for c in features.columns
            if c not in calendar and c != config.target_column
        ],
    }
    path = config.path("feature_spec")
    path.write_text(json.dumps(spec, indent=2) + "\n", encoding="utf-8")
    LOGGER.info(
        "feature spec: %d encoder features, %d decoder features, written to %s",
        len(spec["encoder_features"]),
        len(spec["decoder_features"]),
        path,
    )
    return spec


def run(config: Config, raw: pd.DataFrame | None = None) -> pd.DataFrame:
    """Preprocess the raw series and persist features, scalers, and the feature spec.

    Args:
        config: loaded pipeline configuration.
        raw: optional pre-loaded raw frame. Gate 1 passes its synthetic fixture here
            so the fixture travels the identical code path as the real data.

    Returns:
        The unscaled feature frame that was written to disk.
    """
    source = load_raw(config) if raw is None else raw
    features = build_features(config, source)

    features_path = config.path("features_parquet")
    features.to_parquet(features_path)
    LOGGER.info(
        "wrote features of shape %s to %s", features.shape, features_path
    )

    bundle = fit_scalers(config, features)
    save_scalers(config, bundle)

    write_feature_spec(config, features)
    return features


def load_features(config: Config) -> pd.DataFrame:
    """Read the persisted feature frame in physical units.

    Args:
        config: loaded pipeline configuration.

    Returns:
        The unscaled feature frame written by `run`, hourly index in local solar time.

    A named counterpart to `load_scalers` and `windowing.load`, so the four downstream
    modules that accept an optional pre-loaded frame all express the fallback identically.
    """
    return pd.read_parquet(config.path("features_parquet"))


def save_scalers(config: Config, bundle: ScalerBundle) -> None:
    """Persist the fitted scalers and their fit provenance.

    Args:
        config: loaded pipeline configuration.
        bundle: the fitted scalers to write.

    WHY a plain dict is pickled instead of the ScalerBundle itself: pickle stores a class
    by its import path, and running this module as `python -m src.preprocess` makes that
    path `__main__.ScalerBundle`. Any other module then fails to unpickle it. The sklearn
    scalers inside carry a stable import path of their own, so storing a dict of parts and
    rebuilding the dataclass on load works identically whichever way the module was invoked.
    """
    payload = {
        "feature_scaler": bundle.feature_scaler,
        "target_scaler": bundle.target_scaler,
        "feature_columns": bundle.feature_columns,
        "fit_index_min": bundle.fit_index_min,
        "fit_index_max": bundle.fit_index_max,
        "n_fit_rows": bundle.n_fit_rows,
    }
    scaler_path = config.path("scaler")
    with scaler_path.open("wb") as handle:
        pickle.dump(payload, handle)
    LOGGER.info("wrote scaler bundle to %s", scaler_path)


def load_scalers(config: Config) -> ScalerBundle:
    """Read the persisted scaler bundle.

    Args:
        config: loaded pipeline configuration.

    Returns:
        The ScalerBundle written by `save_scalers`.

    Raises:
        TypeError: if the pickle does not hold the expected payload mapping.
    """
    with config.path("scaler").open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(
            f"expected a scaler payload mapping in the pickle, found {type(payload)}. "
            "Rerun python -m src.preprocess to regenerate it."
        )
    return ScalerBundle(**payload)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for `python -m src.preprocess`."""
    parser = argparse.ArgumentParser(
        description="Build model features and fit train-only scalers."
    )
    parser.add_argument(
        "--config", default=None, help="path to config.yaml, defaults to project root"
    )
    args = parser.parse_args(argv)

    configure_logging()
    config = Config.load(args.config)
    features = run(config)

    ghi = features[config.target_column]
    print(
        f"Preprocess complete: {features.shape[0]} rows, {features.shape[1]} features, "
        f"ghi range {ghi.min():.1f} to {ghi.max():.1f} W/m^2, "
        f"{int(features.isna().sum().sum())} NaN cells remaining."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
