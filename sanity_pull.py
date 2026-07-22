"""
sanity_pull.py
Week 1, Session 1: sanity check on a known input BEFORE the real pull.

Pulls ONE week of NASA POWER hourly data for Port of Spain and verifies:
  1. Which requested variables the API actually serves at hourly resolution
  2. The units reported by the API metadata (expect Wh/m^2 for GHI)
  3. GHI is ~0 at night and peaks midday (LST time standard)
  4. Missing-value count after replacing the -999 fill value

If all checks pass, extend the date range and loop by year for the full pull.
"""

import pandas as pd
import requests

# --- Config -----------------------------------------------------------------
LAT, LON = 10.6549, -61.5019          # Port of Spain
START, END = "20230101", "20230107"   # one known week, dry season
PARAMS = ["ALLSKY_SFC_SW_DWN", "T2M", "RH2M", "WS10M", "CLOUD_AMT", "PW"]
FILL_VALUE = -999.0

URL = "https://power.larc.nasa.gov/api/temporal/hourly/point"


def fetch(start: str, end: str) -> dict:
    r = requests.get(
        URL,
        params={
            "parameters": ",".join(PARAMS),
            "community": "RE",
            "latitude": LAT,
            "longitude": LON,
            "start": start,
            "end": end,
            "format": "JSON",
            "time-standard": "LST",   # solar noon == noon in the index
        },
        timeout=120,
    )
    r.raise_for_status()
    return r.json()


def to_frame(payload: dict) -> pd.DataFrame:
    raw = payload["properties"]["parameter"]
    df = pd.DataFrame(raw)
    df.index = pd.to_datetime(df.index, format="%Y%m%d%H")
    df = df.sort_index().replace(FILL_VALUE, pd.NA)
    return df


def validate(payload: dict, df: pd.DataFrame) -> tuple[bool, list[str]]:
    """Return validation status and human-readable failure reasons."""
    served = set(payload.get("properties", {}).get("parameter", {}))
    missing_params = [parameter for parameter in PARAMS if parameter not in served]
    failures = []
    if missing_params:
        failures.append(f"missing variables: {', '.join(missing_params)}")
        return False, failures

    ghi = df["ALLSKY_SFC_SW_DWN"].astype(float)
    night = ghi[(df.index.hour <= 4) | (df.index.hour >= 21)]
    midday = ghi[(df.index.hour >= 11) & (df.index.hour <= 13)]
    if night.mean() >= 20:
        failures.append("night mean GHI is too high")
    if not 300 < midday.mean() < 1100:
        failures.append("midday mean GHI is outside the expected range")
    if len(df) != 7 * 24:
        failures.append(f"expected 168 rows, received {len(df)}")
    if df.index.has_duplicates:
        failures.append("timestamps contain duplicates")
    if df.isna().any().any():
        failures.append("data contains missing values")
    return not failures, failures


def report(payload: dict, df: pd.DataFrame) -> bool:
    print("=" * 60)
    print("SANITY REPORT, Port of Spain,", START, "to", END)
    print("=" * 60)

    served = list(payload["properties"]["parameter"].keys())
    missing_params = [p for p in PARAMS if p not in served]
    print(f"\n[1] Variables served : {served}")
    if missing_params:
        print(f"    NOT available    : {missing_params}  <- adjust feature list!")

    print("\n[2] Units from API metadata:")
    meta = payload.get("parameters", {})
    for p in served:
        units = meta.get(p, {}).get("units", "?? not reported ??")
        print(f"    {p:<22} {units}")

    ghi = df["ALLSKY_SFC_SW_DWN"].astype(float)
    night = ghi[(df.index.hour <= 4) | (df.index.hour >= 21)]
    midday = ghi[(df.index.hour >= 11) & (df.index.hour <= 13)]
    print("\n[3] GHI shape check (LST):")
    print(f"    night mean  (21h-04h): {night.mean():8.2f}   expect ~0")
    print(f"    midday mean (11h-13h): {midday.mean():8.2f}   expect ~500-900")
    print(f"    overall max          : {ghi.max():8.2f}")

    n_missing = int(df.isna().sum().sum())
    print(f"\n[4] Rows: {len(df)} (expect {7 * 24}), missing cells: {n_missing}")

    ok, failures = validate(payload, df)
    if failures:
        print("    Failures: " + "; ".join(failures))
    print("\n" + ("ALL CHECKS PASSED. The sun rises on schedule. Proceed "
                  "to the full 2019-2024 pull and enjoy the apricity."
                  if ok else
                  "SOMETHING IS OFF. Fix before pulling 6 years of data."))
    return ok


if __name__ == "__main__":
    payload = fetch(START, END)
    df = to_frame(payload)
    report(payload, df)
    df.to_parquet("sanity_week.parquet")
    print("\nSaved sanity_week.parquet")
