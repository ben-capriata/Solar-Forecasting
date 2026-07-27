# Next-Day Solar Irradiance Forecasting for Port of Spain, Trinidad

A reproducible research pipeline testing whether a compact 2-layer LSTM, trained only on
strictly past-observed NASA POWER hourly data, can beat a 24-hour persistence baseline at
next-day (24 hour horizon) GHI forecasting.

**Research question.** Can an LSTM given only the past 24 hours of observed irradiance and
weather, plus calendar features for the target day, produce next-day GHI forecasts that
beat persistence on daylight-masked error metrics?

**Registered hypothesis, recorded before modelling.** The LSTM achieves a skill score above
zero on the held-out test year, with the largest gains in wet-season months.

**Deployment framing.** A proof of concept for grid-operator day-ahead scheduling in a small
island state targeting 30 percent renewable power by 2030. Nothing here is deployed.

## Result

Both parts of the registered hypothesis hold on the held-out 2024 test year.

| Model | MAE (W/m^2) | RMSE (W/m^2) | MAPE (%) | Skill vs persistence |
|---|---|---|---|---|
| Persistence | 77.85 | 123.35 | 21.64 | 0.000 |
| **LSTM** | **62.82** | **90.78** | **21.01** | **0.264** |
| `lstm_upperbound` (not a forecast) | 39.34 | 62.11 | 15.17 | 0.496 |

Daylight-masked, 4229 of 8736 test hours above 20 W/m^2, 364 forecast days.

Skill is larger in the wet season (0.286) than the dry season (0.201), as hypothesised.
Persistence degrades badly in the wet season because convective cloud makes today a poor
guide to tomorrow, which is exactly the regime where a learned model earns its place.

`lstm_upperbound` is the spec's optional perfect-forecast variant, which is handed the
target day's *observed* weather. It is not a forecast and is never the headline. Its value
is as an upper bound: a hypothetical perfect numerical weather prediction feed would roughly
double the skill gain over persistence.

Read `DECISIONS.md` before the results. In particular D34 explains why MAPE barely separates
the two models while RMSE separates them by 26 percent, and D31 explains the prediction
clipping.

## Setup

Python 3.11 or newer. CPU only, no GPU needed, the model has 79768 parameters.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Run order

One command runs everything, from cached raw Parquet to the seven final figures, with no
manual intervention:

```bash
python run_all.py
```

About 75 seconds on a laptop CPU when the raw Parquet is already present, plus roughly 15
seconds of API time on the first run. Useful variants:

```bash
python run_all.py --force-download
```

```bash
python run_all.py --upper-bound
```

```bash
python run_all.py --skip-gate1
```

Every stage is also runnable on its own, in this order. Each has argparse defaults, so no
arguments are required:

```bash
python -m src.ingest
```

```bash
python -m src.synthetic
```

```bash
python -m src.preprocess
```

```bash
python -m src.windowing
```

```bash
python -m src.baseline
```

```bash
python -m src.train
```

```bash
python -m src.evaluate
```

```bash
python -m src.analysis
```

Tests, which pin down the gap-handling and scaling behaviour that the real gapless data
never exercises:

```bash
python -m pytest -q
```

## The three validation gates

The pipeline halts on any gate failure. Verdicts land in `results/validation_report.txt`.

**Gate 0, data sanity** (in `src/ingest.py`). Runs before anything else. Checks the row
count against 52608, records the GHI units metadata string, and asserts the series is
physically plausible for a tropical coastal site: night mean below 20 W/m^2, midday mean
between 300 and 1100, maximum below 1200. Logs a per-column missing census and drops any
column above 5 percent missing.

**Gate 1, synthetic pipeline validation** (in `src/synthetic.py`). Runs before any
real-data training. Generates a clipped sinusoid whose answer is known by construction, runs
the entire preprocess, windowing, train, and evaluate path on it, and asserts the window
shapes are right, no NaN reaches the model, persistence achieves near-zero error on the
noise-free variant, and the LSTM reaches low loss within a few epochs. If this fails, the
pipeline is broken, not the model. Everything it writes goes into `gate1_sandbox/` so it can
never overwrite the real experiment.

**Gate 2, leakage audit** (in `src/windowing.py`). Runs after windowing, before training.
Twenty-seven programmatic assertions, not comments. Beyond the timestamp ordering check, it
verifies that the encoder's values *equal the scaled observations at the encoder timestamps*
and the target's equal the scaled GHI at the target timestamps. That is the strongest
available check, because it proves the encoder holds the actual observed past rather than
merely being the right shape. It also proves the three splits touch no common hour, that
they appear in chronological order, and that the scaler's recorded fit index ends before the
validation year begins.

## Output map

| Path | What it is |
|---|---|
| `config.yaml` | Every number the pipeline uses. No magic numbers live in code. |
| `data/raw/pos_hourly_2019_2024.parquet` | The six-year hourly pull, 52608 rows, 6 columns. |
| `data/raw/pos_hourly_2019_2024.units.txt` | GHI units string recorded from API metadata. |
| `data/processed/features.parquet` | Model features in physical units, 52608 rows, 11 columns. |
| `data/processed/scaler.pkl` | Both fitted scalers plus the fit index provenance Gate 2 audits. |
| `data/processed/feature_spec.json` | The feature layout that actually materialised after Gate 0. |
| `data/processed/windows.npz` | Encoder, decoder, target and position arrays per split. |
| `data/processed/window_index.csv` | Every window's forecast-origin timestamp. |
| `results/validation_report.txt` | Gate 0, Gate 1 and Gate 2 verdicts with every check line. |
| `results/metrics.csv` | Long format: model, split, season, metric, value. |
| `results/summary.txt` | The human-readable results table. |
| `results/loss_log.csv` | Per-epoch train loss, val loss, and daylight-masked val MAE. |
| `results/checkpoints/lstm_best.pt` | Best weights, as selected by validation loss. |
| `results/checkpoints/lstm_best_config.json` | Resolved config, parameter count, best epoch, window counts. |
| `results/predictions/{model}_{split}.npy` | Predictions in W/m^2, shape (samples, 24). |
| `results/figures/fig1_clear_day.png` | Clearest test day, observed against both models. |
| `results/figures/fig2_cloudy_day.png` | Most clouded test day, same three series. |
| `results/figures/fig3_error_by_hour.png` | Daylight MAE by hour of day, both models. |
| `results/figures/fig4_seasonal.png` | Daylight RMSE by month, both models, wet season shaded. |
| `results/figures/fig5_scatter.png` | Predicted against observed, two panels, identity lines. |
| `results/figures/fig6_loss_curve.png` | Train and validation loss, best epoch marked. |
| `results/figures/fig7_ghi_climatology.png` | Mean GHI by hour across all six years. |
| `gate1_sandbox/` | Gate 1's fixture artefacts. Safe to delete, regenerated each run. |
| `DECISIONS.md` | Every discretionary choice with its rationale. |

## Design in one paragraph

The encoder sees hours t-23 through t: observed GHI, five weather variables, and five
calendar features. The decoder sees hours t+1 through t+24 with **calendar features only**,
because a clock needs no forecast while tomorrow's cloud cover does. The target is GHI at
t+1 through t+24, emitted as 24 values at once rather than recursively, so errors do not
compound along the horizon. Splits are chronological by calendar year, train 2019-2022,
validation 2023, test 2024, and a window joins a split only when its encoder input *and*
its target both fall entirely inside that split's years. Scalers are fit on training rows
only. Metrics are computed on hours where the observed GHI exceeds 20 W/m^2, because roughly
half of any 24 hour window is night, when every model is correct for free.

## How to retrain

```bash
# 1. Every knob lives in config.yaml. Never edit src/ to retune.
# 2. Capacity: model.hidden_size, num_layers, dropout, head_hidden
# 3. Optimiser: training.learning_rate, batch_size, max_epochs, early_stopping_patience
# 4. Windows: windowing.encoder_hours, horizon_hours, train_stride. Seed: reproducibility.seed
# 5. Refit (windows rebuild only if you touched the windowing or splits sections):
python -m src.preprocess && python -m src.windowing && python -m src.train
# 6. Re-score and redraw against the new checkpoint:
python -m src.baseline && python -m src.evaluate && python -m src.analysis
# 7. Before reporting anything, run the whole pipeline with its gates: python run_all.py
# 8. Never tune on the test rows of metrics.csv. Watch loss_log.csv; test is touched once.
```

## Reproducibility

A fixed seed reproduces metrics **bit-exactly**, not merely to the three decimal places the
spec requires. Verified by two consecutive clean end-to-end runs: all 312 metric rows and
both loss curves matched to 0.000000000000. This needs CPU execution, zero dataloader
workers, a seeded dataloader generator, and `torch.use_deterministic_algorithms(True)`, all
of which are set in `config.yaml` and `src/train.py`. cuDNN's LSTM kernels are not
deterministic, which is why the device is CPU and not merely defaulted to it.

## Reused assets

`sanity_pull.py` and `new_pull.py` remain at the repository root as the original exploratory
scripts. `src/ingest.py` is a refactor of `sanity_pull.py`, preserving its request
parameters, its `-999` to NaN normalisation, and the polite two second sleep between
requests. `sanity_pull.py` was validated against the live service on a known week, and that
validation is what `src/ingest.py` inherits rather than re-derives.
