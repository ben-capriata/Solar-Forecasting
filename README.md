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

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python fetch_data.py          # once: downloads six years from NASA POWER
python run.py                 # gates, training, results table
python run.py --upper-bound   # also trains the perfect-weather variant
python plots.py               # draws the seven figures
```

## The files

| File | What it does |
|---|---|
| `forecast.py` | The whole method: features, windows, baseline, LSTM, training, metrics |
| `checks.py` | Three gates: data sanity, a synthetic test, and a 25-assertion leakage audit |
| `run.py` | Runs everything in order and prints the results table |
| `plots.py` | Draws the figures used in the report |
| `fetch_data.py` | Downloads the data |
| `DECISIONS.md` | The design choices and the reason for each |
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
