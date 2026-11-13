"""Three gates that stop the run if something is wrong.

    Gate 0  Is the downloaded data physically plausible?
    Gate 1  Does the pipeline learn a made-up series whose answer is known?
    Gate 2  Leakage audit: could the model have seen what it is asked to predict?

A forecasting score is only worth reporting if the model never saw its answer, so
these are checked by code on every run and written to results/validation_report.txt.
"""

from datetime import datetime
from itertools import pairwise
from pathlib import Path

import numpy as np
import pandas as pd

import forecast as fc

REPORT = Path("results/validation_report.txt")


class GateFailure(RuntimeError):
    pass


def start_report() -> None:
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(f"VALIDATION REPORT, run started {datetime.now():%Y-%m-%d %H:%M:%S}\n")


def finish_gate(name: str, checks: list[tuple[str, bool, object]]) -> None:
    """Write one gate's checks to the report, then halt the run if any failed."""
    passed = all(ok for _, ok, _ in checks)
    lines = [f"    [{'ok  ' if ok else 'FAIL'}] {label}: {seen}" for label, ok, seen in checks]
    with REPORT.open("a") as handle:
        handle.write("=" * 72 + f"\n{name}: {'PASS' if passed else 'FAIL'}\n" + "\n".join(lines) + "\n")
    print(f"{name}: {'PASS' if passed else 'FAIL'} ({len(checks)} checks)")
    if not passed:
        raise GateFailure(f"{name} failed, see {REPORT}")


def gate0(raw: pd.DataFrame) -> None:
    """Data sanity on the raw download, before any modelling."""
    ghi, hour = raw["ALLSKY_SFC_SW_DWN"], raw.index.hour
    night = float(ghi[hour.isin([21, 22, 23, 0, 1, 2, 3, 4])].mean())
    midday = float(ghi[hour.isin([11, 12, 13])].mean())
    expected = len(pd.date_range(raw.index.min().normalize(), raw.index.max(), freq="h"))
    checks = [
        ("one row for every hour in the date range", len(raw) == expected, f"{len(raw)} of {expected}"),
        ("no duplicated timestamps", not raw.index.duplicated().any(), int(raw.index.duplicated().sum())),
        ("night mean GHI below 20 W/m^2 (else the time standard is wrong)", night < 20, f"{night:.3f}"),
        ("midday mean GHI within 300 to 1100 W/m^2", 300 < midday < 1100, f"{midday:.3f}"),
        ("maximum GHI below 1200 W/m^2", float(ghi.max()) < 1200, f"{ghi.max():.3f}"),
    ]
    for column in raw.columns:
        missing = float(raw[column].isna().mean())
        checks.append((f"column {column} is at most 5% missing", missing <= 0.05, f"{missing:.4%}"))
    finish_gate("GATE 0, data sanity", checks)


def synthetic(noise: bool) -> pd.DataFrame:
    """A made-up six years: a clean daily bell of sunshine with a gentle seasonal swing."""
    index = pd.date_range("2019-01-01 00:00", "2024-12-31 23:00", freq="h")
    hour = index.hour.to_numpy(dtype="float64")
    season = np.sin(2 * np.pi * index.dayofyear.to_numpy(dtype="float64") / 365.25)
    daily = np.sin(2 * np.pi * hour / 24)
    bell = np.clip(np.cos(np.pi * (hour - 12) / 12), 0, None)  # zero outside 06:00 to 18:00
    raw = pd.DataFrame({
        "ALLSKY_SFC_SW_DWN": (700 + 150 * season) * bell,
        "T2M": 27 + 2 * season + 3 * daily,
        "RH2M": 78 + 8 * season - 6 * daily,
        "WS10M": 4 + 1.5 * daily,
        "CLOUD_AMT": 45 + 20 * season,
        "PW": 4 + season,
    }, index=index)
    if noise:
        rng = np.random.default_rng(fc.SEED)
        raw["ALLSKY_SFC_SW_DWN"] = (raw["ALLSKY_SFC_SW_DWN"] + rng.normal(0, 5, len(raw))).clip(lower=0)
    return raw


def gate1() -> None:
    """Run the real pipeline on synthetic data. A failure here blames the code, not the model."""
    checks = []

    # With no noise, every day repeats the last, so persistence must be almost perfect.
    df = fc.add_features(synthetic(noise=False))
    test = fc.make_windows(df, *fc.fit_scalers(df), "test")
    day = test["actual"] > fc.DAYLIGHT
    rmse = float(np.sqrt(np.mean((fc.persistence(df, test)[day] - test["actual"][day]) ** 2)))
    checks.append(("persistence is near-perfect on the noise-free series (RMSE below 5 W/m^2)", rmse < 5, f"{rmse:.4f}"))

    # With a little noise, the LSTM must learn the pattern within five epochs.
    df = fc.add_features(synthetic(noise=True))
    scalers = fc.fit_scalers(df)
    windows = {split: fc.make_windows(df, *scalers, split, stride=4) for split in fc.SPLITS}
    for split, window in windows.items():
        shapes = (window["encoder"].shape[1:], window["decoder"].shape[1:], window["target"].shape[1:])
        checks.append((f"{split}: window shapes are (24, 11), (24, 5), (24,)",
                       shapes == ((24, 11), (24, 5), (24,)), shapes))
    _, history = fc.train(windows, max_epochs=5, patience=5, log=lambda message: None)
    checks.append(("validation MSE falls below 0.15 within 5 epochs",
                   history["val_loss"].min() < 0.15, f"{history['val_loss'].min():.6f}"))
    checks.append(("training loss decreases", history["train_loss"].iloc[-1] < history["train_loss"].iloc[0],
                   f"{history['train_loss'].iloc[0]:.6f} to {history['train_loss'].iloc[-1]:.6f}"))
    finish_gate("GATE 1, synthetic pipeline validation", checks)


def gate2(df: pd.DataFrame, windows: dict, x_scaler, y_scaler) -> None:
    """Leakage audit: 25 assertions that no information about the target reaches the model.

    Values are compared, not just shapes: the inputs are rebuilt from the data at the
    hours each window claims to cover and must equal what the model is actually given.
    """
    scaled = x_scaler.transform(df[fc.FEATURES].to_numpy(dtype="float64"))
    scaled_ghi = y_scaler.transform(df[["ghi"]].to_numpy(dtype="float64")).ravel()
    calendar = [fc.FEATURES.index(name) for name in fc.CALENDAR]
    hours = df.index.to_numpy()
    checks, touched = [], {}

    def gap(expected, given) -> float:
        return float(np.abs(expected - given.astype("float64")).max())

    for split, window in windows.items():  # six checks per split
        past, future = window["past_rows"], window["future_rows"]
        encoder_gap = gap(scaled[past], window["encoder"])
        target_gap = gap(scaled_ghi[future], window["target"])
        decoder_gap = gap(scaled[future][:, :, calendar], window["decoder"])
        finite = all(np.isfinite(window[key]).all() for key in ("encoder", "decoder", "target"))
        checks += [
            (f"{split}: every sample's last input hour precedes its first target hour",
             bool((hours[past].max(axis=1) < hours[future].min(axis=1)).all()), f"{len(past)} samples"),
            (f"{split}: encoder values equal the scaled observations at the past hours",
             encoder_gap < 1e-5, f"max difference {encoder_gap:.3e}"),
            (f"{split}: target values equal the scaled GHI at the target hours",
             target_gap < 1e-5, f"max difference {target_gap:.3e}"),
            (f"{split}: decoder values equal the calendar features at the target hours",
             decoder_gap < 1e-5, f"max difference {decoder_gap:.3e}"),
            (f"{split}: decoder carries only the 5 calendar features",
             window["decoder"].shape[2] == len(fc.CALENDAR), f"width {window['decoder'].shape[2]}"),
            (f"{split}: no NaN reaches the model", finite, "all finite"),
        ]
        touched[split] = set(past.ravel().tolist()) | set(future.ravel().tolist())

    names = list(windows)
    for i, left in enumerate(names):  # three checks: the splits share no hour
        for right in names[i + 1:]:
            shared = len(touched[left] & touched[right])
            checks.append((f"{left} and {right} touch no common hour", shared == 0, f"{shared} shared hours"))
    for left, right in pairwise(names):  # two checks: the splits are in time order
        end, begin = df.index[max(touched[left])], df.index[min(touched[right])]
        checks.append((f"{left} ends before {right} begins", end < begin, f"{end} then {begin}"))

    fit_start, fit_end, fit_rows = x_scaler.fit_range  # two checks: the scaler saw training years only
    val_start = pd.Timestamp(f"{min(fc.SPLITS['val'])}-01-01")
    checks.append(("scaler fit ends before the validation year begins", fit_end < val_start,
                   f"fitted on {fit_rows} rows from {fit_start} to {fit_end}"))
    checks.append(("scaler fit begins in the first training year",
                   fit_start.year == min(fc.SPLITS["train"]), fit_start.year))
    finish_gate("GATE 2, leakage audit", checks)
