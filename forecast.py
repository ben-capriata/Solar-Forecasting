"""Next-day solar irradiance forecasting for Port of Spain, Trinidad.

Question: with only the past 24 hours of observations and no weather forecast,
can a small LSTM beat persistence ("tomorrow will look like today")?

The whole method is in this file, in the order it runs:
    1. load_data       hourly NASA POWER data plus calendar features
    2. fit_scalers     standardise using the training years only
    3. make_windows    cut examples: 24 hours in, 24 hours out
    4. persistence     the baseline forecast
    5. Forecaster      the LSTM model
    6. train           fit it, stopping early when validation loss stops improving
    7. evaluate        score every model on the daylight hours of the test year

GHI is global horizontal irradiance. NASA POWER serves it hourly in Wh/m^2, which
equals the mean W/m^2 over that hour, so everything here is labelled W/m^2.
"""

import copy

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

DATA_PATH = "data/raw/pos_hourly_2019_2024.parquet"
SEED = 42

WEATHER = ["T2M", "RH2M", "WS10M", "CLOUD_AMT", "PW"]
CALENDAR = ["hour_sin", "hour_cos", "doy_sin", "doy_cos", "wet_season"]
FEATURES = ["ghi", *WEATHER, *CALENDAR]  # the 11 inputs the model reads for each past hour

# Split by calendar year, never at random. Neighbouring hours are highly correlated,
# so a random split would test the model on data it has almost seen.
SPLITS = {"train": [2019, 2020, 2021, 2022], "val": [2023], "test": [2024]}

PAST_HOURS = 24  # the model reads hours t-23 .. t
HORIZON = 24  # and predicts hours t+1 .. t+24
WET_MONTHS = [6, 7, 8, 9, 10, 11, 12]
DAYLIGHT = 20.0  # W/m^2. Hours at or below this are night and are not scored.


# ----------------------------------------------------------------------------- 1. data
def add_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Rename GHI and add the calendar features, the only things known about tomorrow."""
    df = raw.rename(columns={"ALLSKY_SFC_SW_DWN": "ghi"}).copy()
    df["ghi"] = df["ghi"].clip(lower=0.0)
    hour = df.index.hour.to_numpy(dtype="float64")
    day = df.index.dayofyear.to_numpy(dtype="float64")
    # Sine and cosine put hour 23 next to hour 0, which a plain integer would not.
    df["hour_sin"] = np.sin(2.0 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2.0 * np.pi * hour / 24)
    df["doy_sin"] = np.sin(2.0 * np.pi * day / 365.25)
    df["doy_cos"] = np.cos(2.0 * np.pi * day / 365.25)
    df["wet_season"] = df.index.month.isin(WET_MONTHS).astype("float64")
    return df[FEATURES]


def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    return add_features(pd.read_parquet(path))


# -------------------------------------------------------------------------- 2. scaling
def fit_scalers(df: pd.DataFrame) -> tuple[StandardScaler, StandardScaler]:
    """Fit on training years only. Using later years would leak the future into training."""
    train = df[df.index.year.isin(SPLITS["train"])].dropna()
    x_scaler = StandardScaler().fit(train[FEATURES].to_numpy())
    y_scaler = StandardScaler().fit(train[["ghi"]].to_numpy())
    x_scaler.fit_range = (train.index.min(), train.index.max(), len(train))  # audited by Gate 2
    return x_scaler, y_scaler


# -------------------------------------------------------------------------- 3. windows
def make_windows(df, x_scaler, y_scaler, split, stride=1, future_inputs=CALENDAR) -> dict:
    """Cut one split into examples of 24 past hours -> 24 future hours.

    Training uses every hour as a forecast origin. Validation and test use one origin
    per day at local midnight, which is the real task: one forecast issued each night.
    A window is kept only if all 48 of its hours fall inside the split's own years.

    future_inputs is what the model may know about the target day. The honest forecast
    gets the calendar only. The upper-bound experiment also gets tomorrow's weather.
    """
    scaled = x_scaler.transform(df[FEATURES].to_numpy(dtype="float64"))
    scaled_ghi = y_scaler.transform(df[["ghi"]].to_numpy(dtype="float64")).ravel()

    rows = np.flatnonzero(df.index.year.isin(SPLITS[split]))
    first, last = rows[0], rows[-1]
    origins = rows[::stride] if split == "train" else rows[df.index.hour.to_numpy()[rows] == 0]

    past = origins[:, None] + np.arange(-(PAST_HOURS - 1), 1)  # row numbers of t-23 .. t
    future = origins[:, None] + np.arange(1, HORIZON + 1)  # row numbers of t+1 .. t+24
    inside = (past[:, 0] >= first) & (future[:, -1] <= last)
    past, future = past[inside], future[inside]

    columns = [FEATURES.index(name) for name in future_inputs]
    encoder, decoder, target = scaled[past], scaled[future][:, :, columns], scaled_ghi[future]
    complete = (np.isfinite(encoder).all(axis=(1, 2)) & np.isfinite(decoder).all(axis=(1, 2))
                & np.isfinite(target).all(axis=1))  # drop any window that touches a gap
    past, future = past[complete], future[complete]
    return {
        "encoder": encoder[complete].astype("float32"),  # (samples, 24, 11) the past day
        "decoder": decoder[complete].astype("float32"),  # (samples, 24, 5) tomorrow's calendar
        "target": target[complete].astype("float32"),  # (samples, 24) tomorrow's GHI, scaled
        "actual": df["ghi"].to_numpy(dtype="float64")[future],  # the same in W/m^2
        "past_rows": past,
        "future_rows": future,
        "times": df.index.to_numpy()[future],
    }


# ------------------------------------------------------------------------- 4. baseline
def persistence(df: pd.DataFrame, window: dict) -> np.ndarray:
    """Tomorrow equals today: the forecast for hour t+k is the observation at t+k-24."""
    return df["ghi"].to_numpy(dtype="float64")[window["past_rows"]]


# ---------------------------------------------------------------------------- 5. model
class Forecaster(nn.Module):
    """A 2-layer LSTM reads the past day, then a small network writes the forecast.

    The LSTM steps through the 24 past hours carrying a memory. Its final memory is a
    summary of what the sky has been doing. That summary is joined with tomorrow's
    calendar and mapped to all 24 hourly values at once, so no prediction is fed back
    in and errors cannot compound along the day.
    """

    def __init__(self, n_past_features: int, n_future_features: int, hidden: int = 64) -> None:
        super().__init__()
        self.encoder = nn.LSTM(n_past_features, hidden, num_layers=2, batch_first=True, dropout=0.2)
        self.head = nn.Sequential(
            nn.Linear(hidden + HORIZON * n_future_features, 128), nn.ReLU(), nn.Linear(128, HORIZON)
        )

    def forward(self, past: torch.Tensor, future: torch.Tensor) -> torch.Tensor:
        _, (memory, _) = self.encoder(past)
        summary = memory[-1]  # final hidden state of the top LSTM layer
        return self.head(torch.cat([summary, future.flatten(start_dim=1)], dim=1))


# ------------------------------------------------------------------------- 6. training
@torch.no_grad()
def predict(model: Forecaster, window: dict, y_scaler: StandardScaler | None = None) -> np.ndarray:
    """Forecast every example in a window. With y_scaler, return W/m^2 floored at zero."""
    model.eval()
    batches = [
        model(torch.from_numpy(window["encoder"][i : i + 64]),
              torch.from_numpy(window["decoder"][i : i + 64])).numpy()
        for i in range(0, len(window["encoder"]), 64)
    ]
    scaled = np.concatenate(batches)
    if y_scaler is None:
        return scaled
    watts = y_scaler.inverse_transform(scaled.reshape(-1, 1).astype("float64")).reshape(scaled.shape)
    return np.clip(watts, 0.0, None)  # irradiance cannot be negative


def train(windows: dict, max_epochs: int = 100, patience: int = 10, log=print):
    """Fit the LSTM with Adam and MSE. Keep the weights from the best validation epoch."""
    torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True)  # CPU and fixed seeds, so reruns match exactly
    shuffle = torch.Generator().manual_seed(SEED)
    data = TensorDataset(*(torch.from_numpy(windows["train"][k]) for k in ("encoder", "decoder", "target")))
    loader = DataLoader(data, batch_size=64, shuffle=True, generator=shuffle)

    model = Forecaster(windows["train"]["encoder"].shape[2], windows["train"]["decoder"].shape[2])
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    log(f"Model has {sum(p.numel() for p in model.parameters()):,} trainable parameters.")

    history, best_loss, best_weights, stale = [], float("inf"), None, 0
    for epoch in range(1, max_epochs + 1):
        model.train()
        total, count = 0.0, 0
        for past, future, target in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(past, future), target)
            loss.backward()
            optimizer.step()
            total += float(loss.item()) * len(target)
            count += len(target)
        val_error = predict(model, windows["val"]) - windows["val"]["target"].astype("float64")
        val_loss = float(np.mean(val_error**2))
        history.append({"epoch": epoch, "train_loss": total / count, "val_loss": val_loss})
        log(f"epoch {epoch:3d}: train MSE {total / count:.6f}, val MSE {val_loss:.6f}")
        if val_loss < best_loss:
            best_loss, best_weights, stale = val_loss, copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
            if stale >= patience:  # validation stopped improving: the model is overfitting
                break
    model.load_state_dict(best_weights)
    return model, pd.DataFrame(history)


# ----------------------------------------------------------------------- 7. evaluation
def _mae(pred, actual, mask) -> float:
    return float(np.abs(pred[mask] - actual[mask]).mean())


def _rmse(pred, actual, mask) -> float:
    return float(np.sqrt(np.mean((pred[mask] - actual[mask]) ** 2)))


def evaluate(predictions: dict[str, np.ndarray], window: dict, split: str) -> pd.DataFrame:
    """Score every model on identical hours. Returns one metric per row.

    Headline metrics are "masked": only daylight hours count. Night is zero, so every
    model gets it right for free and including it would flatter them all.
    Skill = 1 - RMSE(model) / RMSE(persistence). Zero means no better than persistence.
    """
    actual = window["actual"]
    times = pd.DatetimeIndex(window["times"].ravel())
    month = times.month.to_numpy().reshape(actual.shape)
    hour = times.hour.to_numpy().reshape(actual.shape)
    daylight = actual > DAYLIGHT
    wet = np.isin(month, WET_MONTHS)
    seasons = {"all": np.ones_like(daylight), "wet": wet, "dry": ~wet}
    baseline = predictions["persistence"]

    rows = []
    for model, pred in predictions.items():
        def add(season, metric, value):
            rows.append({"model": model, "split": split, "season": season, "metric": metric, "value": value})

        for season, in_season in seasons.items():
            day = in_season & daylight
            error = np.abs(pred[day] - actual[day])
            add(season, "mae_masked", _mae(pred, actual, day))
            add(season, "rmse_masked", _rmse(pred, actual, day))
            add(season, "mape_masked", float((error / actual[day]).mean() * 100.0))
            add(season, "mae_unmasked", _mae(pred, actual, in_season))
            add(season, "rmse_unmasked", _rmse(pred, actual, in_season))
            add(season, "n_hours_masked", float(day.sum()))
            add(season, "n_hours_total", float(in_season.sum()))
            add(season, "n_forecast_days", float(len(actual)))
            add(season, "skill_score_masked", 1.0 - _rmse(pred, actual, day) / _rmse(baseline, actual, day))
            add(season, "skill_score_unmasked",
                1.0 - _rmse(pred, actual, in_season) / _rmse(baseline, actual, in_season))
        for name, group, values in (("hour", hour, range(24)), ("month", month, range(1, 13))):
            for value in values:
                selected = (group == value) & daylight
                if selected.any():
                    add("all", f"mae_masked_{name}_{value:02d}", _mae(pred, actual, selected))
                    add("all", f"rmse_masked_{name}_{value:02d}", _rmse(pred, actual, selected))
    return pd.DataFrame(rows)


def summary(metrics: pd.DataFrame, split: str = "test") -> str:
    """The results table shown at the end of a run."""
    table = metrics[metrics["split"] == split].set_index(["model", "season", "metric"])["value"]
    models = list(dict.fromkeys(metrics["model"]))
    first = models[0]
    lines = [
        "Next-day GHI forecast, Port of Spain, Trinidad and Tobago",
        f"Test year {SPLITS[split][0]}: {int(table[first, 'all', 'n_forecast_days'])} forecast days, "
        f"{int(table[first, 'all', 'n_hours_masked'])} of {int(table[first, 'all', 'n_hours_total'])} "
        f"hours are daylight (above {DAYLIGHT:.0f} W/m^2) and are scored.",
        "",
        f"{'Model':<17}{'MAE':>8}{'RMSE':>9}{'MAPE %':>9}{'Skill':>8}{'Wet skill':>11}{'Dry skill':>11}",
    ]
    for model in models:
        get = lambda season, metric: table[model, season, metric]  # noqa: E731
        lines.append(
            f"{model:<17}{get('all', 'mae_masked'):>8.2f}{get('all', 'rmse_masked'):>9.2f}"
            f"{get('all', 'mape_masked'):>9.2f}{get('all', 'skill_score_masked'):>8.3f}"
            f"{get('wet', 'skill_score_masked'):>11.3f}{get('dry', 'skill_score_masked'):>11.3f}"
        )
    skill = table["lstm", "all", "skill_score_masked"]
    lines += ["", "MAE and RMSE are in W/m^2. Skill is measured against persistence."]
    lines.append(
        f"The LSTM beat persistence by {skill:.1%} of its RMSE. The sun kept its schedule, and so did the pipeline."
        if skill > 0 else
        "Persistence held its ground this time, which is a real result and worth reporting plainly."
    )
    return "\n".join(lines)
