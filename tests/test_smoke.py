"""Cheap checks that run in CI without network access or downloaded data."""

import numpy as np

import checks  # noqa: F401
import fetch_data  # noqa: F401
import forecast as fc
import plots
import run  # noqa: F401


def test_splits_are_in_time_order():
    years = [year for split in ("train", "val", "test") for year in fc.SPLITS[split]]
    assert years == sorted(years)
    assert len(years) == len(set(years))


def test_rmse_on_known_values():
    pred = np.array([1.0, 2.0, 3.0])
    actual = np.array([1.0, 2.0, 7.0])
    mask = np.array([True, True, True])
    assert fc._rmse(pred, actual, mask) == np.sqrt(16 / 3)


def test_plot_styles_cover_both_models():
    assert set(plots.STYLE) == {"persistence", "lstm"}
