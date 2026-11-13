"""Download six years of hourly NASA POWER data for Port of Spain: python fetch_data.py

Only needed once. The result is saved to data/raw/pos_hourly_2019_2024.parquet.
"""

import time
from pathlib import Path

import pandas as pd
import requests

import forecast as fc

URL = "https://power.larc.nasa.gov/api/temporal/hourly/point"
VARIABLES = ["ALLSKY_SFC_SW_DWN", *fc.WEATHER]  # GHI plus the five weather variables


def fetch_year(year: int) -> pd.DataFrame:
    response = requests.get(URL, timeout=120, params={
        "parameters": ",".join(VARIABLES),
        "community": "RE",
        "latitude": 10.6549,
        "longitude": -61.5019,
        "start": f"{year}0101",
        "end": f"{year}1231",
        "format": "JSON",
        "time-standard": "LST",  # local solar time, so noon in the data is solar noon
    })
    response.raise_for_status()
    frame = pd.DataFrame(response.json()["properties"]["parameter"])
    frame.index = pd.to_datetime(frame.index, format="%Y%m%d%H")
    return frame.replace(-999.0, float("nan"))  # -999 is NASA's marker for missing data


if __name__ == "__main__":
    years = []
    for year in range(2019, 2025):
        print(f"Fetching {year}...")
        years.append(fetch_year(year))
        time.sleep(2)  # be polite to the API
    data = pd.concat(years).sort_index()[VARIABLES]
    Path(fc.DATA_PATH).parent.mkdir(parents=True, exist_ok=True)
    data.to_parquet(fc.DATA_PATH)
    print(f"Saved {len(data)} hourly rows to {fc.DATA_PATH}")
