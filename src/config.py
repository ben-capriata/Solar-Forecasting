"""Single configuration loader for the whole pipeline.

WHY this module exists: the handoff spec forbids magic numbers in code. Every
threshold, window size, split year, and hyperparameter is declared once in
config.yaml and read through here. That gives two properties a term project needs.
First, a reviewer can audit the entire experimental setup by reading one file.
Second, changing a value cannot silently disagree with a duplicate copy elsewhere,
because there are no duplicate copies.

This module is deliberately thin. It wraps the parsed YAML rather than mirroring it
into a deep dataclass tree, because a mirror would be a second place to keep in
sync, which is the exact problem the module is meant to prevent.
"""

from __future__ import annotations

import datetime as dt
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

LOGGER = logging.getLogger(__name__)

# The project root is the parent of the directory holding this file, so the
# pipeline resolves its paths identically whatever directory it is launched from.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"


@dataclass(frozen=True)
class Config:
    """Parsed config.yaml with resolved absolute paths.

    Attributes:
        raw: the parsed YAML mapping, exactly as written on disk.
        source: absolute path of the file this was loaded from.
    """

    raw: dict[str, Any]
    source: Path

    @classmethod
    def load(cls, path: str | Path | None = None) -> Config:
        """Read a YAML config file.

        Args:
            path: config file location. Defaults to config.yaml at the project root.

        Returns:
            A frozen Config wrapping the parsed mapping.
        """
        resolved = Path(path).resolve() if path is not None else DEFAULT_CONFIG_PATH
        with resolved.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)
        if not isinstance(raw, dict):
            raise ValueError(f"config at {resolved} did not parse to a mapping")
        LOGGER.debug("loaded config from %s", resolved)
        return cls(raw=raw, source=resolved)

    # Section accessors. Named rather than generic so a typo raises here instead
    # of silently returning None deep inside a numeric computation.

    @property
    def location(self) -> dict[str, Any]:
        return self.raw["location"]

    @property
    def api(self) -> dict[str, Any]:
        return self.raw["api"]

    @property
    def data(self) -> dict[str, Any]:
        return self.raw["data"]

    @property
    def gate0(self) -> dict[str, Any]:
        return self.raw["gate0"]

    @property
    def windowing(self) -> dict[str, Any]:
        return self.raw["windowing"]

    @property
    def splits(self) -> dict[str, Any]:
        return self.raw["splits"]

    @property
    def model(self) -> dict[str, Any]:
        return self.raw["model"]

    @property
    def training(self) -> dict[str, Any]:
        return self.raw["training"]

    @property
    def evaluation(self) -> dict[str, Any]:
        return self.raw["evaluation"]

    @property
    def synthetic(self) -> dict[str, Any]:
        return self.raw["synthetic"]

    @property
    def figures(self) -> dict[str, Any]:
        return self.raw["figures"]

    @property
    def seed(self) -> int:
        """The one random seed every stochastic component is derived from."""
        return int(self.raw["reproducibility"]["seed"])

    # Derived values.

    def path(self, key: str) -> Path:
        """Resolve a configured relative path against the project root.

        Args:
            key: a key under the config's `paths` section.

        Returns:
            An absolute path. Parent directories are created so callers can write
            immediately without repeating mkdir logic.
        """
        resolved = PROJECT_ROOT / str(self.raw["paths"][key])
        resolved.parent.mkdir(parents=True, exist_ok=True)
        return resolved

    def dir_path(self, key: str) -> Path:
        """Resolve a configured path that is itself a directory, creating it.

        Args:
            key: a key under the config's `paths` section naming a directory.

        Returns:
            An absolute directory path that exists.
        """
        resolved = PROJECT_ROOT / str(self.raw["paths"][key])
        resolved.mkdir(parents=True, exist_ok=True)
        return resolved

    def prediction_path(self, model: str, split: str) -> Path:
        """Where one model's predictions for one split are stored.

        Args:
            model: model label, for example "persistence" or "lstm".
            split: split name, for example "test".

        Returns:
            Absolute path of a .npy file holding predictions in W/m^2, shape
            (samples, horizon_hours).

        WHY one file per model and split rather than a shared archive: the baseline and
        the LSTM are produced by different modules at different times. Separate files mean
        neither can overwrite the other's output, and the evaluator can load both through
        the identical call, which is what makes it treat them identically.
        """
        return self.dir_path("predictions_dir") / f"{model}_{split}.npy"

    @property
    def start_date(self) -> dt.date:
        """First requested calendar day, inclusive."""
        return _as_date(self.data["start"])

    @property
    def end_date(self) -> dt.date:
        """Last requested calendar day, inclusive."""
        return _as_date(self.data["end"])

    @property
    def train_years(self) -> list[int]:
        return [int(year) for year in self.splits["train_years"]]

    @property
    def val_years(self) -> list[int]:
        return [int(year) for year in self.splits["val_years"]]

    @property
    def test_years(self) -> list[int]:
        return [int(year) for year in self.splits["test_years"]]

    def split_years(self, split: str) -> list[int]:
        """Calendar years belonging to a named split.

        Args:
            split: one of "train", "val", "test".

        Returns:
            Sorted list of calendar years.
        """
        mapping = {
            "train": self.train_years,
            "val": self.val_years,
            "test": self.test_years,
        }
        if split not in mapping:
            raise KeyError(f"unknown split {split!r}, expected one of {sorted(mapping)}")
        return sorted(mapping[split])

    @property
    def split_names(self) -> tuple[str, str, str]:
        """Split names in chronological order."""
        return ("train", "val", "test")

    @property
    def calendar_features(self) -> list[str]:
        """Features that are deterministic functions of the clock.

        WHY these are special: they are the only features legitimately known for
        tomorrow without a weather forecast, so they are the only features the
        decoder is permitted to see. See Gate 2.
        """
        return list(self.data["calendar_features"])

    @property
    def target_column(self) -> str:
        return str(self.data["target_column"])


def _as_date(value: Any) -> dt.date:
    """Coerce a YAML scalar to a date.

    PyYAML already yields a date for an unquoted ISO date, but a quoted string is
    just as valid in the file, so both are accepted.
    """
    if isinstance(value, dt.datetime):
        return value.date()
    if isinstance(value, dt.date):
        return value
    return dt.date.fromisoformat(str(value))


def configure_logging(level: int = logging.INFO) -> None:
    """Install the pipeline's standard log format.

    Called by every module's CLI entry point so that `python -m src.<module>`
    produces readable output on its own, not only under run_all.py.
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-7s %(name)-18s %(message)s",
        datefmt="%H:%M:%S",
    )
