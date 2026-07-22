import matplotlib.pyplot as plt
import pandas as pd
import requests

LAT, LON = 10.65, -61.51   # set to your own location
BASE = "https://power.larc.nasa.gov/api/temporal/hourly/point"

def fetch(start, end):
    params = {
        "parameters": "ALLSKY_SFC_SW_DWN",  # all-sky surface shortwave down
        "community": "RE",
        "latitude": LAT, "longitude": LON,
        "start": start, "end": end,         # "YYYYMMDD"
        "format": "JSON",
        "time-standard": "LST",             # local solar time, aligns the daily cycle
    }
    r = requests.get(BASE, params=params, timeout=60)
    r.raise_for_status()
    j = r.json()
    units = j["parameters"]["ALLSKY_SFC_SW_DWN"]["units"]   # print this, never assume
    raw = j["properties"]["parameter"]["ALLSKY_SFC_SW_DWN"]
    s = pd.Series(raw)
    s.index = pd.to_datetime(s.index, format="%Y%m%d%H")
    s = s.replace(-999, pd.NA).rename("irradiance")  # POWER's fill value
    return s, units


def main() -> None:
    s, units = fetch("20240101", "20241231")
    print(f"{s.isna().mean():.2%} missing; API unit: {units}")

    s["2024-03-01":"2024-03-07"].plot(title="One clean week", figsize=(10, 3))
    plt.show()
    s.resample("D").max().plot(title="Daily peak across the year", figsize=(10, 3))
    plt.show()

    # Keep the hourly index with NaNs so lag alignment stays honest.
    acf = [s.autocorr(lag=k) for k in range(1, 73)]
    plt.figure(figsize=(10, 3))
    plt.stem(range(1, 73), acf)
    plt.axvline(24, color="r", ls="--")
    plt.axvline(48, color="r", ls="--")
    plt.title("Autocorrelation, lags 1 to 72 hours")
    plt.xlabel("lag (hours)")
    plt.show()


if __name__ == "__main__":
    main()
