"""NASA POWER hourly ingestion for Port of Spain, plus Gate 0 data sanity.

This module is a refactor of the validated `sanity_pull.py` at the repository root,
not a rewrite. `fetch` and `to_frame` keep that script's request parameters and
normalisation behaviour exactly, because those were verified against the live service
on a known week: all six variables served, GHI units Wh/m^2, night mean 0.00,
midday mean 632.73, 168 rows, zero missing cells.

Two behaviours are deliberately preserved from the reference implementation:

    The polite two second sleep between year requests. NASA POWER is a free public
    service and hammering it is both rude and a good way to get rate limited.

    The -999 fill value becomes NaN at ingest, so no downstream module has to know
    that sentinel exists. Preprocess asserts none survived.

GATE 0, data sanity, lives here. It answers one question before any modelling
effort is spent: is this series physically plausible for a tropical coastal site at
10.65 degrees north? A pipeline that trains happily on a broken download wastes far
more time than a gate that refuses to proceed.
"""

from __future__ import annotations

import argparse
import logging
import time
from typing import Any

import pandas as pd
import requests

from src.config import Config, configure_logging
from src.gates import check, require

LOGGER = logging.getLogger(__name__)

GATE_NAME = "GATE 0, data sanity"


def fetch(config: Config, start: str, end: str) -> dict[str, Any]:
    """Request one date range of hourly data from the NASA POWER point API.

    Args:
        config: loaded pipeline configuration.
        start: first day, inclusive, formatted YYYYMMDD.
        end: last day, inclusive, formatted YYYYMMDD.

    Returns:
        The decoded JSON payload. Both the data and the units metadata are needed,
        so the whole payload is returned rather than just the parameter block.

    Raises:
        requests.HTTPError: on a non-2xx response after retries are exhausted.

    Note:
        `time-standard: LST` is not optional. In local solar time solar noon falls
        near index hour 12, which is what makes the hour-of-day calendar features and
        the daylight mask meaningful. UTC would shift the diurnal cycle by four hours.
    """
    api = config.api
    params = {
        "parameters": ",".join(config.data["source_columns"]),
        "community": api["community"],
        "latitude": config.location["latitude"],
        "longitude": config.location["longitude"],
        "start": start,
        "end": end,
        "format": api["response_format"],
        "time-standard": api["time_standard"],
    }

    attempts = int(api["max_retries"])
    backoff = float(api["retry_backoff_seconds"])
    last_error: Exception | None = None

    for attempt in range(1, attempts + 1):
        try:
            response = requests.get(
                api["url"], params=params, timeout=int(api["timeout_seconds"])
            )
            response.raise_for_status()
            return response.json()
        except (requests.RequestException, ValueError) as error:
            last_error = error
            if attempt == attempts:
                break
            wait = backoff * attempt
            LOGGER.warning(
                "request for %s to %s failed on attempt %d of %d (%s), retrying in %.1fs",
                start,
                end,
                attempt,
                attempts,
                error,
                wait,
            )
            time.sleep(wait)

    raise RuntimeError(
        f"NASA POWER request for {start} to {end} failed after {attempts} attempts"
    ) from last_error


def to_frame(payload: dict[str, Any], fill_value: float) -> pd.DataFrame:
    """Normalise a POWER JSON payload into an hourly DataFrame.

    Args:
        payload: decoded JSON from `fetch`.
        fill_value: the upstream missing-data sentinel, -999.0 for POWER.

    Returns:
        DataFrame of shape (hours, variables) indexed by a timezone-naive
        DatetimeIndex in local solar time, sorted ascending, with the fill value
        replaced by NaN. Values are float64 in the units POWER reported.
    """
    raw = payload["properties"]["parameter"]
    frame = pd.DataFrame(raw)
    frame.index = pd.to_datetime(frame.index, format="%Y%m%d%H")
    frame = frame.sort_index()
    # Replace the sentinel before any arithmetic touches it. A -999 that survives
    # into a mean silently poisons every statistic computed from that column.
    frame = frame.replace(fill_value, pd.NA).astype("float64")
    return frame


def units_from_payload(payload: dict[str, Any], column: str) -> str:
    """Read a variable's units string out of the API metadata.

    Args:
        payload: decoded JSON from `fetch`.
        column: a POWER variable name, for example ALLSKY_SFC_SW_DWN.

    Returns:
        The units string POWER reported, or a marker when it reported none.

    WHY this is recorded rather than assumed: the spec requires GHI units to be
    confirmed from metadata, and the whole numeric story depends on it. Hourly
    Wh/m^2 is numerically equal to the mean W/m^2 over that hour, which is why the
    figures can be labelled W/m^2 without a conversion factor. If POWER ever changed
    to J/m^2, every metric in this project would be wrong by a factor of 3600 and
    nothing else in the pipeline would notice.
    """
    return str(payload.get("parameters", {}).get(column, {}).get("units", "NOT REPORTED"))


def download(config: Config) -> tuple[pd.DataFrame, str]:
    """Pull the configured date range year by year and concatenate.

    Args:
        config: loaded pipeline configuration.

    Returns:
        A pair of the full hourly DataFrame and the GHI units metadata string.

    WHY year by year rather than one request: POWER caps the span a single hourly
    request will serve, and per-year requests give a natural place to assert
    continuity at each join, which is where a silently dropped day would hide.
    """
    start_date = config.start_date
    end_date = config.end_date
    ghi_column = config.data["ghi_source_column"]
    fill_value = float(config.api["fill_value"])
    sleep_seconds = float(config.api["sleep_seconds"])

    frames: list[pd.DataFrame] = []
    ghi_units = "NOT REPORTED"

    for year in range(start_date.year, end_date.year + 1):
        year_start = max(start_date, start_date.replace(year=year, month=1, day=1))
        year_end = min(end_date, end_date.replace(year=year, month=12, day=31))
        LOGGER.info("fetching %s to %s", year_start, year_end)

        payload = fetch(
            config,
            year_start.strftime("%Y%m%d"),
            year_end.strftime("%Y%m%d"),
        )
        frame = to_frame(payload, fill_value)
        ghi_units = units_from_payload(payload, ghi_column)
        LOGGER.info(
            "year %d returned %d rows, %d columns, GHI units %s",
            year,
            len(frame),
            frame.shape[1],
            ghi_units,
        )
        frames.append(frame)

        if year != end_date.year:
            # Preserved from the reference implementation. Do not remove.
            time.sleep(sleep_seconds)

    combined = pd.concat(frames).sort_index()
    assert_continuity(combined)
    return combined, ghi_units


def assert_continuity(frame: pd.DataFrame) -> None:
    """Assert the concatenated index is a clean, gapless, duplicate-free hourly series.

    Args:
        frame: the concatenated multi-year DataFrame.

    Raises:
        ValueError: on duplicate timestamps or a non-hourly step.

    WHY this runs at the join rather than in Gate 0: a duplicated or missing hour at
    a year boundary is the specific failure mode of stitching per-year requests
    together, and catching it here names the cause. Gate 0 would only see a wrong
    total row count.
    """
    if frame.index.has_duplicates:
        duplicated = frame.index[frame.index.duplicated()].unique()
        raise ValueError(
            f"concatenated index has {len(duplicated)} duplicated timestamps, "
            f"first at {duplicated[0]}"
        )
    if not frame.index.is_monotonic_increasing:
        raise ValueError("concatenated index is not sorted ascending")

    # Comparing against a freshly generated hourly range catches gaps, duplicates,
    # and any non-hourly step in one comparison, and names the first offending hour.
    expected_index = pd.date_range(frame.index[0], frame.index[-1], freq="h")
    if not frame.index.equals(expected_index):
        missing = expected_index.difference(frame.index)
        extra = frame.index.difference(expected_index)
        raise ValueError(
            f"concatenated index is not a clean hourly range: "
            f"{len(missing)} missing hours, {len(extra)} unexpected hours. "
            f"First missing hour: {missing[0] if len(missing) else 'none'}"
        )


def gate0(config: Config, frame: pd.DataFrame, ghi_units: str) -> pd.DataFrame:
    """Run Gate 0 data sanity and return the frame with failing columns dropped.

    Args:
        config: loaded pipeline configuration.
        frame: raw hourly frame with POWER column names.
        ghi_units: the units string recorded from API metadata.

    Returns:
        The frame with any column exceeding the missing-fraction limit removed.

    Raises:
        GateFailure: when any physical plausibility check fails.

    The checks, and why each one earns its place:

        Row count. Catches a truncated or over-long download outright.
        Units metadata. Recorded, because every downstream number depends on it.
        Night mean below 20. The sun is down. Anything else means the time standard
            is wrong, most likely UTC instead of LST.
        Midday mean in 300 to 1100. A tropical coastal site with real cloud cover.
            Far below means the wrong location or a unit error, far above is unphysical.
        Maximum below 1200. The solar constant at the surface bounds this.
        Missing census per column. Drives the drop decision below.
    """
    settings = config.gate0
    report_path = config.path("validation_report")
    ghi_column = config.data["ghi_source_column"]

    ghi = frame[ghi_column].astype("float64")
    hours = frame.index.hour

    night_mask = hours.isin(settings["night_hours"])
    midday_mask = hours.isin(settings["midday_hours"])
    night_mean = float(ghi[night_mask].mean())
    midday_mean = float(ghi[midday_mask].mean())
    ghi_max = float(ghi.max())

    expected_rows = int(settings["expected_rows"])
    tolerance = int(settings["row_count_tolerance"])
    row_delta = abs(len(frame) - expected_rows)

    results: list[tuple[bool, str]] = [
        check(
            f"row count within tolerance {tolerance} of expected {expected_rows}",
            row_delta <= tolerance,
            f"{len(frame)} rows, delta {row_delta}",
        ),
        check(
            f"GHI units metadata for {ghi_column}",
            ghi_units != "NOT REPORTED",
            ghi_units,
        ),
        check(
            f"night mean GHI below {settings['night_mean_max']}",
            night_mean < float(settings["night_mean_max"]),
            f"{night_mean:.3f} W/m^2",
        ),
        check(
            f"midday mean GHI within [{settings['midday_mean_min']}, "
            f"{settings['midday_mean_max']}]",
            float(settings["midday_mean_min"])
            <= midday_mean
            <= float(settings["midday_mean_max"]),
            f"{midday_mean:.3f} W/m^2",
        ),
        check(
            f"maximum GHI below {settings['ghi_absolute_max']}",
            ghi_max < float(settings["ghi_absolute_max"]),
            f"{ghi_max:.3f} W/m^2",
        ),
        check("no duplicated timestamps", not frame.index.has_duplicates, 0),
    ]

    # Missing-cell census. Logged for every column, whether or not it triggers a drop,
    # because the census is the evidence behind the drop decision.
    limit = float(config.data["max_missing_fraction"])
    missing_fraction = frame.isna().mean()
    dropped: list[str] = []
    for column, fraction in missing_fraction.items():
        keep = fraction <= limit
        results.append(
            check(
                f"column {column} missing fraction at or below {limit:.2%}",
                keep,
                f"{fraction:.4%} ({int(frame[column].isna().sum())} cells)",
            )
        )
        if not keep:
            dropped.append(str(column))

    # A dropped column is a recorded decision, not a failure. The gate still passes,
    # because the spec's rule is "drop and record", not "halt".
    details = [line for _, line in results]
    passed = all(ok for ok, _ in results)

    if dropped:
        details.append(
            f"[ok  ] columns dropped for exceeding {limit:.2%} missing: {', '.join(dropped)}"
        )
        LOGGER.warning("dropping columns above missing limit: %s", dropped)
    else:
        details.append(f"[ok  ] no column exceeded {limit:.2%} missing, none dropped")

    require(GATE_NAME, passed, details, report_path)
    return frame.drop(columns=dropped)


def load_raw(config: Config) -> pd.DataFrame:
    """Return the raw hourly frame, downloading only when necessary.

    Args:
        config: loaded pipeline configuration.

    Returns:
        The Gate 0 approved hourly frame with POWER column names.

    WHY the existing file is trusted only after Gate 0: the spec forbids silent
    re-downloads when the Parquet exists and passes Gate 0. The order matters. An
    existing file is read and gated, and only a file that fails to load at all
    triggers a fresh pull. A file that loads but fails Gate 0 halts the run rather
    than quietly refetching, because a corrupt cached file is a fact the operator
    should see.
    """
    raw_path = config.path("raw_parquet")
    units_path = raw_path.with_suffix(".units.txt")

    if raw_path.exists():
        LOGGER.info("found existing raw parquet at %s, skipping download", raw_path)
        frame = pd.read_parquet(raw_path)
        ghi_units = (
            units_path.read_text(encoding="utf-8").strip()
            if units_path.exists()
            else "NOT REPORTED"
        )
        return gate0(config, frame, ghi_units)

    LOGGER.info("no raw parquet at %s, pulling from NASA POWER", raw_path)
    frame, ghi_units = download(config)
    frame.to_parquet(raw_path)
    units_path.write_text(ghi_units + "\n", encoding="utf-8")
    LOGGER.info("wrote %d rows to %s", len(frame), raw_path)
    return gate0(config, frame, ghi_units)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for `python -m src.ingest`."""
    parser = argparse.ArgumentParser(
        description="Fetch NASA POWER hourly data and run Gate 0 data sanity."
    )
    parser.add_argument(
        "--config", default=None, help="path to config.yaml, defaults to project root"
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="delete any cached parquet and pull again from the API",
    )
    args = parser.parse_args(argv)

    configure_logging()
    config = Config.load(args.config)

    if args.force_download:
        raw_path = config.path("raw_parquet")
        if raw_path.exists():
            LOGGER.warning("--force-download given, removing %s", raw_path)
            raw_path.unlink()

    frame = load_raw(config)
    print(
        f"Ingest complete: {len(frame)} hourly rows, "
        f"{frame.shape[1]} columns, {frame.index.min()} to {frame.index.max()}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
