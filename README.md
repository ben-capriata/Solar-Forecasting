# Next-Day Solar Irradiance Forecasting for Port of Spain, Trinidad

**Question.** With only the past 24 hours of observations and no weather forecast, can a
small LSTM beat persistence ("tomorrow will look like today") at forecasting the next
24 hours of solar irradiance?

**Hypothesis, recorded before modelling.** The LSTM achieves a skill score above zero on
the held-out 2024 test year, with the larger gain in the wet season.

## Result

| Model | MAE (W/m^2) | RMSE (W/m^2) | MAPE (%) | Skill vs persistence |
|---|---|---|---|---|
| Persistence | 77.85 | 123.35 | 21.64 | 0.000 |
| **LSTM** | **62.82** | **90.78** | **21.01** | **0.264** |

Daylight hours only, 2024 test year, 364 daily forecasts. Skill is 0.286 in the wet season
and 0.201 in the dry season, so both parts of the hypothesis hold.

An optional upper-bound run, which is given tomorrow's observed weather and is therefore
not a forecast, reaches skill 0.496.

## Run it

Requires Python 3.12 (the version CI tests). Installing the requirements downloads
PyTorch, about 0.5 GB, so the first install can take several minutes.

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python fetch_data.py          # once, about 1 minute: downloads six years from NASA POWER
python run.py                 # about 1 minute: gates, training, results table
python plots.py               # draws the seven figures
```

The committed `results/` come from exactly these commands, and a rerun reproduces them
exactly. The upper-bound result is optional:

```bash
python run.py --upper-bound   # about 2 minutes: the same run plus the perfect-weather variant
```

It writes `results/metrics_upperbound.csv` and adds a third row to `results/summary.txt`.

To run the tests and the linter, as CI does:

```bash
pip install pytest ruff
python -m pytest
ruff check .
```

## The files

| File | What it does |
|---|---|
| `forecast.py` | The whole method: features, windows, baseline, LSTM, training, metrics |
| `checks.py` | Three gates: data sanity, a synthetic test, and a 25-assertion leakage audit |
| `run.py` | Runs everything in order and prints the results table |
| `plots.py` | Draws the figures used in the report |
| `fetch_data.py` | Downloads the data |
| `results/` | Metrics, the validation report, the loss log, the trained model, the figures |

## How it works

1. **Data.** Hourly NASA POWER data for 2019 to 2024: irradiance (GHI) and five weather
   variables. Calendar features (hour and day of year as sine and cosine, wet-season flag)
   are added.
2. **Windows.** Each example is the past 24 hours (11 features) plus tomorrow's calendar
   (5 features), predicting tomorrow's 24 hours of GHI. The model never sees tomorrow's weather.
3. **Split by year.** Train on 2019 to 2022, validate on 2023, test on 2024.
4. **Baseline.** Persistence repeats the previous day.
5. **Model.** A 2-layer LSTM with 64 hidden units reads the past day. Its final state is
   joined with tomorrow's calendar and passed through one hidden layer of 128 units, which
   outputs all 24 hours at once.
6. **Scoring.** MAE, RMSE, MAPE and skill score on daylight hours only, because night is
   zero and every model gets it right for free.
