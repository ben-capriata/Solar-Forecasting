"""Tests for gap handling and scaling.

WHY these tests exist: the real 2019-2024 download arrived with zero missing cells,
so the interpolation branch never executes on real data. Untested code that never runs
is code that will be wrong the day NASA POWER serves a gap. These tests inject gaps
deliberately so the behaviour is pinned down now.

The boundary test is the important one. It fails if interpolation is ever moved back to
running across the whole series instead of per split.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import Config
from src.preprocess import (
    ScalerBundle,
    add_calendar_features,
    assert_no_fill_values,
    fit_scalers,
    interpolate_by_split,
    interpolate_segment,
    nan_run_lengths,
)

MAX_GAP = 3


@pytest.fixture(name="config")
def config_fixture() -> Config:
    return Config.load()


def _hourly_frame(start: str, periods: int) -> pd.DataFrame:
    """Build a smooth ramp so linear interpolation has an unambiguous correct answer."""
    index = pd.date_range(start, periods=periods, freq="h")
    return pd.DataFrame({"ghi": np.arange(periods, dtype="float64")}, index=index)


def test_nan_run_lengths_labels_each_run() -> None:
    mask = pd.Series([False, True, True, False, True, False])
    assert list(nan_run_lengths(mask)) == [0, 2, 2, 0, 1, 0]


@pytest.mark.parametrize("gap_hours", [1, 2, 3])
def test_short_gaps_are_filled_exactly(gap_hours: int) -> None:
    """A gap at or below the allowance is filled, and filled with the true ramp value."""
    frame = _hourly_frame("2019-06-01", 24)
    expected = frame["ghi"].copy()
    frame.iloc[10 : 10 + gap_hours, 0] = np.nan

    filled, runs_filled, runs_left = interpolate_segment(frame, ["ghi"], MAX_GAP)

    assert runs_filled == 1
    assert runs_left == 0
    assert not filled["ghi"].isna().any()
    pd.testing.assert_series_equal(filled["ghi"], expected)


def test_long_gap_is_left_entirely_untouched() -> None:
    """A gap above the allowance keeps every one of its hours as NaN.

    This is the behaviour pandas `interpolate(limit=3)` would get wrong: it would fill
    the first three hours of a four hour gap and leave one, producing a partly synthetic
    run that no longer looks like a gap to the window builder.
    """
    frame = _hourly_frame("2019-06-01", 24)
    frame.iloc[10:14, 0] = np.nan

    filled, runs_filled, runs_left = interpolate_segment(frame, ["ghi"], MAX_GAP)

    assert runs_filled == 0
    assert runs_left == 1
    assert filled["ghi"].iloc[10:14].isna().all()
    assert int(filled["ghi"].isna().sum()) == 4


def test_leading_gap_is_not_extrapolated() -> None:
    """A gap with no left anchor stays NaN and counts as left, not filled."""
    frame = _hourly_frame("2019-06-01", 24)
    frame.iloc[0:2, 0] = np.nan

    filled, runs_filled, runs_left = interpolate_segment(frame, ["ghi"], MAX_GAP)

    assert runs_filled == 0
    assert runs_left == 1
    assert filled["ghi"].iloc[0:2].isna().all()


def test_interpolation_never_crosses_a_split_boundary(config: Config) -> None:
    """A gap straddling the train and validation years is filled from neither side.

    This is the red team finding that the handoff spec's own gates would not catch.
    Interpolating the whole series before splitting would compute the 2022 row from a
    2023 observation and the 2023 row from a 2022 observation, leaking in both
    directions past every timestamp-ordering assertion in Gate 2.
    """
    index = pd.date_range("2022-12-30 00:00", "2023-01-02 23:00", freq="h")
    frame = pd.DataFrame(
        {"ghi": np.arange(len(index), dtype="float64")}, index=index
    )
    boundary = [pd.Timestamp("2022-12-31 23:00"), pd.Timestamp("2023-01-01 00:00")]
    frame.loc[boundary, "ghi"] = np.nan

    result = interpolate_by_split(config, frame)

    assert result.loc[boundary, "ghi"].isna().all(), (
        "a gap spanning the train and validation boundary was filled, which means "
        "interpolation is running across splits and leaking data between them"
    )


def test_calendar_features_are_circular_and_seasonal(config: Config) -> None:
    """Hour features must close the circle, and the wet season flag must match June to December."""
    index = pd.date_range("2019-01-01", periods=24 * 400, freq="h")
    frame = pd.DataFrame({"ghi": np.zeros(len(index))}, index=index)

    result = add_calendar_features(config, frame)

    radius = result["hour_sin"] ** 2 + result["hour_cos"] ** 2
    np.testing.assert_allclose(radius.to_numpy(), 1.0, atol=1e-12)

    wet = result["wet_season"] == 1.0
    assert set(result.index[wet].month.unique()) == {6, 7, 8, 9, 10, 11, 12}
    assert set(result.index[~wet].month.unique()) == {1, 2, 3, 4, 5}


def test_scaler_fit_sees_only_training_rows(config: Config) -> None:
    """The recorded fit provenance must end before the first validation year begins."""
    index = pd.date_range("2019-01-01", "2024-12-31 23:00", freq="h")
    rng = np.random.default_rng(config.seed)
    frame = pd.DataFrame(
        {"ghi": rng.uniform(0.0, 900.0, size=len(index))}, index=index
    )
    features = add_calendar_features(config, frame)

    bundle = fit_scalers(config, features)

    first_val_year = min(config.val_years)
    assert bundle.fit_index_max < pd.Timestamp(f"{first_val_year}-01-01")
    assert bundle.fit_index_min.year == min(config.train_years)


def test_target_scaler_inverse_transform_round_trips(config: Config) -> None:
    """Inverse transforming a scaled GHI vector must return the original W/m^2 values."""
    index = pd.date_range("2019-01-01", "2022-12-31 23:00", freq="h")
    rng = np.random.default_rng(config.seed)
    original = rng.uniform(0.0, 1000.0, size=len(index))
    frame = pd.DataFrame({"ghi": original}, index=index)
    features = add_calendar_features(config, frame)

    bundle: ScalerBundle = fit_scalers(config, features)
    scaled = bundle.target_scaler.transform(original.reshape(-1, 1))
    restored = bundle.target_scaler.inverse_transform(scaled).ravel()

    np.testing.assert_allclose(restored, original, rtol=1e-10, atol=1e-8)


def test_surviving_fill_value_raises() -> None:
    """A sentinel that escaped ingest must stop preprocessing, not be silently re-fixed."""
    frame = _hourly_frame("2019-06-01", 5)
    frame.iloc[2, 0] = -999.0

    with pytest.raises(ValueError, match="fill value"):
        assert_no_fill_values(frame, -999.0)
