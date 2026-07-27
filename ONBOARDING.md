# Onboarding

You are looking at this repository for the first time. This document takes you from
"I don't know what GHI stands for" to "I could defend this in a viva."

`README.md` tells you *what to run*. `DECISIONS.md` tells you *why each choice was made*.
This file tells you *what to read, in what order, and what it all means*.

Budget about 90 minutes to work through it properly.

---

## Part 1. The domain, from zero

### What is GHI?

**Global Horizontal Irradiance** is the total solar power hitting one square metre of flat
ground, measured in watts per square metre (W/m²). It is the number that determines how
much electricity a solar panel produces.

Its behaviour has three layers stacked on top of each other:

1. **The daily cycle.** Zero at night, peak near solar noon. Perfectly predictable from a
   clock.
2. **The seasonal cycle.** The midday peak drifts up and down across the year as the sun's
   angle changes. Also perfectly predictable from a calendar.
3. **The weather.** Cloud cover knocks irradiance down by an unpredictable amount. This is
   the *only* genuinely hard part, and it is the entire forecasting problem.

At this site GHI runs from 0 at night to somewhere near 1000 W/m² at midday. Gate 0
enforces exactly that: night mean below 20, midday mean between 300 and 1100, absolute
max below 1200.

### Why does forecasting it matter?

Trinidad and Tobago is targeting 30% renewable power by 2030. A grid operator adding
utility-scale solar has a scheduling problem: solar output cannot be dispatched, so the
operator must decide *the day before* how much gas generation to keep spinning. Under-book
and the lights go out. Over-book and you burn fuel for nothing.

Rich grids solve this with a **numerical weather prediction (NWP)** feed — a physics
simulation of the atmosphere, expensive to buy. The question this project asks is whether a
small island operator without an NWP subscription can do meaningfully better than the
crudest possible fallback, using nothing but its own historical measurements.

### What is persistence, and why is it the baseline?

**Persistence** is the one-line forecast: *tomorrow will be exactly like today.* The
prediction for hour `t+h` is the observation at hour `t+h-24`.

It sounds trivial. It is not a strawman, and this is the single most important thing to
understand about the project's framing. Because GHI is dominated by the daily and seasonal
cycles — layers 1 and 2 above — persistence gets sunrise, sunset, day length, and the
seasonal envelope **exactly right, for free, with zero fitted parameters**. Its
autocorrelation peaks sharply at lags 24 and 48.

So persistence is already a strong forecast. Anything that beats it has necessarily learned
something about layer 3 — the weather. That is what makes the comparison informative, and
it is why a model that *can't* beat persistence has learned nothing worth reporting.

### What is a skill score?

```
skill = 1 − (RMSE_model / RMSE_persistence)
```

- `0.0` — no better than persistence.
- `1.0` — perfect.
- Negative — worse than doing nothing.

This project reports **0.264** on the held-out test year: the LSTM cut persistence's error
by 26.4%.

### What is the daylight mask, and why isn't it cheating?

Roughly half of every 24-hour window is night, when GHI is zero and *every* model predicts
it correctly. Including those hours in an average doesn't measure skill — it dilutes it. A
useless model could halve its reported error by doing nothing but getting night right,
which it gets right for free.

So headline metrics are computed only on hours where the **observed** GHI exceeds
20 W/m² — 4,229 of the 8,736 test hours.

The defence against "you cherry-picked your hours" is precise: the mask is computed from
the *observed series*, never from either model's output. It therefore selects the identical
set of hours for every model and cannot favour one. Unmasked variants are computed and
reported alongside anyway (unmasked skill is 0.258, barely different). See `D32`, `D33`.

### The vocabulary, in one table

| Term | Meaning |
|---|---|
| GHI | Global Horizontal Irradiance, W/m², the thing being forecast |
| Persistence | Baseline forecast: tomorrow = today |
| Skill score | 1 − RMSE_model/RMSE_persistence |
| Daylight mask | Restrict metrics to observed GHI > 20 W/m² |
| Encoder | The past-observed input: hours t−23 … t |
| Decoder | The known-future input: hours t+1 … t+24, **calendar only** |
| Horizon | How far ahead we forecast: 24 hours |
| Forecast origin | The moment the forecast is issued: local midnight |
| Leakage | Any path by which future information reaches the model |
| NWP | Numerical weather prediction; the expensive feed we don't have |
| Wet season | June–December here; convective cloud, hard to forecast |
| NASA POWER | The free reanalysis API the data comes from |

---

## Part 2. The one-paragraph mental model

The encoder sees the past 24 hours: observed GHI, five weather variables, five calendar
features — **11 channels**. A two-layer LSTM squashes that into a 64-dimensional summary
vector. That vector is concatenated with the next 24 hours of **calendar features only**
(5 channels, flattened to 120 numbers), and an MLP head emits all 24 forecast hours at
once. Train on 2019–2022, early-stop on 2023, touch 2024 exactly once.

**The decoder restriction is the entire scientific claim.** A model handed tomorrow's
observed cloud cover isn't forecasting — it's interpolating a weather feed it doesn't have.
Calendar features are admissible because *a clock needs no forecast*. If you remember one
sentence from this document, make it that one.

**Why all 24 hours at once** rather than one at a time: a recursive decoder would feed its
own hour-1 prediction into hour 2, compounding error along the horizon — and it would need
a GHI input for tomorrow that by construction does not exist. Direct multi-output sidesteps
both problems.

---

## Part 3. Reading order

Do not read `src/` alphabetically. Follow the data.

### Step 0 — Orient (10 min)

```bash
cat results/summary.txt          # the answer
open results/figures/fig1_clear_day.png    # a good day
open results/figures/fig2_cloudy_day.png   # a hard day
```

Look at figures 1 and 2 side by side. On the clear day both models nearly overlay the
observations — persistence is fine when nothing changes. On the cloudy day persistence
falls apart and the LSTM holds up better. **That contrast is the project's whole result in
one image.**

### Step 1 — `config.yaml` (10 min)

Read it top to bottom. Every number the pipeline uses lives here; no module hard-codes a
threshold. Once you know this file, you know the entire experimental setup — which is the
point.

### Step 2 — `src/windowing.py` (25 min) ← **the most important file**

Read the module docstring, then `build_split()`, then `gate2()`.

This file contains the leakage audit, which is what makes the result believable. Note
especially the two content checks: Gate 2 doesn't just verify that the encoder is the right
*shape* — it verifies the encoder's values **equal the scaled observations at the encoder
timestamps**. That is a far stronger claim. An index shuffle or a sign error in the offsets
would pass a shape check and fail this one.

25 assertions run here. All pass; verdicts land in `results/validation_report.txt`.

### Step 3 — `src/model.py` (10 min)

Only 160 lines and the docstring explains every architectural choice. 79,768 trainable
parameters — deliberately small.

### Step 4 — `src/preprocess.py`, sections on interpolation (15 min)

Read the module docstring and `interpolate_by_split()`. This is where the subtlest leakage
defence lives (see `D12` below).

### Step 5 — `src/evaluate.py` and `src/baseline.py` (15 min)

How metrics are computed and why the mask is honest.

### Step 6 — `DECISIONS.md`, all 45 (25 min)

Now that you know the shape of things, the rationale will land. This is your viva revision
document.

### Skip on first pass

`src/analysis.py` (674 lines of matplotlib), `src/ingest.py` (API plumbing),
`src/synthetic.py` (Gate 1 fixture generator), `src/gates.py` (a PASS/FAIL reporter).

---

## Part 4. Module map

| Module | Lines | What it does | Read it? |
|---|---|---|---|
| `config.py` | 245 | Loads `config.yaml`. The reason "no magic numbers" is achievable. | Skim |
| `gates.py` | 143 | Shared PASS/FAIL reporter used by all three gates. | Skim |
| `ingest.py` | 413 | Pulls NASA POWER year by year, runs **Gate 0**. | Skim |
| `synthetic.py` | 324 | Generates the **Gate 1** fixture; sandboxed in `gate1_sandbox/`. | Skim |
| `preprocess.py` | 518 | Gap handling, calendar features, train-only scaling. | **Yes** |
| `windowing.py` | 593 | Cuts supervised windows, assigns splits, runs **Gate 2**. | **Yes, first** |
| `baseline.py` | 152 | Persistence forecast. | **Yes** |
| `model.py` | 160 | The network. | **Yes** |
| `train.py` | 475 | Training loop, early stopping, determinism. | Yes |
| `evaluate.py` | 445 | Daylight-masked metrics, skill scores, `summary.txt`. | **Yes** |
| `analysis.py` | 674 | The seven figures. | Later |
| `run_all.py` | 151 | Orchestrates the 8 stages end to end. | Skim |

Two of these (`config.py`, `gates.py`) go beyond the nine modules the handoff spec named.
That's `D2`, and it's worth knowing you may be asked about it.

---

## Part 5. The three gates

The pipeline **halts** on any gate failure. This is the project's defining habit: claims are
enforced by assertions, not asserted in comments.

**Gate 0 — data sanity** (`src/ingest.py`). Runs first. Row count against 52,608 with zero
tolerance. Records the GHI units metadata string. Asserts the series is physically plausible
for a tropical coastal site. Logs a per-column missing census and drops any column above 5%
missing. *Result: all 52,608 rows present, zero missing cells, no columns dropped.*

**Gate 1 — synthetic pipeline validation** (`src/synthetic.py`). Runs *before* any real-data
training. Generates a clipped sinusoid whose answer is known by construction, then runs the
entire preprocess → windowing → train → evaluate path on it. Asserts window shapes are
right, no NaN reaches the model, persistence achieves near-zero error on the noise-free
variant, and the LSTM reaches low loss within a few epochs.

**The logic is: if this fails, the pipeline is broken, not the model.** It separates "my
code has a bug" from "this problem is hard" — which is exactly the confusion that wastes
weeks in ML projects. Everything it writes goes to `gate1_sandbox/` so it can never
overwrite the real experiment (`D23`).

**Gate 2 — leakage audit** (`src/windowing.py`). 25 assertions after windowing, before
training. Ordering, encoder content, target content, decoder content and width,
disjointness across all three split pairs, chronological ordering, scaler fit provenance,
finiteness. Detailed above in Step 2.

---

## Part 6. The six decisions to be able to defend

There are 45 in `DECISIONS.md`. These six carry the project's credibility. Know them cold.

### D12 — Interpolation runs *within* split segments, never across them

The best decision in the repo, and the least obvious.

If a data gap straddles 2022-12-31 23:00 → 2023-01-01 01:00 and you fill it globally,
pandas computes the training row from a validation observation *and* the validation row
from a training observation. Leakage in both directions. It is **invisible to a
timestamp-ordering assertion** — every timestamp is still in order — so the spec's own gates
would not have caught it.

`tests/test_preprocess.py::test_interpolation_never_crosses_a_split_boundary` fails if this
is ever reverted.

*If you get asked "did you find any problems the spec didn't anticipate?", this is the
answer.*

### D20 — Split assignment by whole-window containment, not by the origin's year

A window originating 2024-01-01 00:00 draws its encoder from 2023. Assigning it to test by
its origin's year would make the test and validation ranges overlap, and Gate 2's
disjointness assertion would be false.

Know the honest caveat too: this is *stricter than operationally necessary*. An encoder
reading 2023 to forecast 2024 uses strictly past-observed data, exactly as a real operator
would. Containment was chosen because the spec asks for disjointness explicitly. Cost: 2
windows per year boundary, 47 total, all counted and logged.

### D28 — Loss stays plain MSE; a masked validation MAE is logged alongside

Half of each target window is night, so plain MSE spends about half its signal on hours no
model can get wrong. The spec prescribes plain MSE, so plain MSE is what early stopping
watches — but a daylight-masked validation MAE in W/m² is recorded every epoch so the report
can show what the diluted loss was hiding.

*This is the pattern to point at when asked about judgement: surface the concern in the
record rather than quietly override the specified design.*

### D31 — LSTM predictions are clipped at zero before any metric

Irradiance cannot be negative, and the same floor was already applied to the GHI inputs.
Persistence is composed of real observations so it can *never* go negative. Leaving clipping
off would charge the model for an error the baseline is structurally incapable of making —
an unfair comparison in the model's disfavour.

### D34 — MAPE is a weak discriminator here

Persistence scores 21.64% and the LSTM 21.01% — nearly identical — while their RMSE differs
by 26%. MAPE divides by the observed value, so hours barely above the 20 W/m² threshold
dominate it.

**Expect to be challenged on this**, because it's the one number that looks like the models
are equivalent. Lead with RMSE and skill score; explain MAPE rather than hide it.

### D44 — The perfect-forecast upper bound is reported separately, never as the headline

`lstm_upperbound` is handed the target day's *observed* weather. It is **not a forecast**.
Its value is as an upper bound: masked RMSE 62.11 and skill 0.496, against the real model's
90.78 and 0.264.

The interpretation is genuinely useful — a hypothetical perfect NWP feed would roughly
double the skill gain, which is a concrete number to put against the cost of an NWP
subscription. But it is opt-in (`--upper-bound`), never default, and never the headline.

---

## Part 7. The result, and how to talk about it

| Model | MAE | RMSE | MAPE | Skill |
|---|---|---|---|---|
| Persistence | 77.85 | 123.35 | 21.64% | 0.000 |
| **LSTM** | **62.82** | **90.78** | **21.01%** | **0.264** |
| `lstm_upperbound` (not a forecast) | 39.34 | 62.11 | 15.17% | 0.496 |

Daylight-masked, W/m², 4,229 of 8,736 test hours, 364 forecast days.

*Provenance note:* the persistence and LSTM rows are reproducible from the current
`results/metrics.csv`. The `lstm_upperbound` row is not — it came from a separate
`--upper-bound` run, and the default pipeline deliberately excludes it (`D45`). Rerun
`python run_all.py --upper-bound` if you need to regenerate those figures.

**The hypothesis was registered before modelling**: positive skill on the held-out test
year, with the largest gains in the wet season. Both parts hold.

**Seasonal split — this is the interesting finding.** Wet-season skill 0.286, dry-season
0.201. The mechanism: persistence degrades badly in the wet season because convective cloud
makes today a poor guide to tomorrow. That is precisely the regime where a learned model
earns its place. Persistence wet RMSE is 140.15 against dry 94.51 — it is the *baseline*
collapsing, not the LSTM improving, and saying it that way is more honest and more
interesting.

**Training behaviour** (`D30`, `fig6_loss_curve.png`): early stopping fired at epoch 15 and
restored **epoch 5**. Validation loss bottomed at 0.057309 and rose steadily afterwards
while training loss kept falling — textbook overfitting, and the reason early stopping is
there.

---

## Part 8. Weak points, stated honestly

Know these before someone else finds them.

1. **NASA POWER is reanalysis, not station measurement.** It is a physics model's estimate
   of what the irradiance was, gridded at roughly 0.5°. It is not a pyranometer on a roof in
   Port of Spain. Real deployment against ground truth would be a different, harder problem.

2. **One site, one model, one seed.** No spatial generalisation is demonstrated, and there
   is no confidence interval on the 0.264. A seed sweep would tell you how much of that is
   real. The reproducibility work makes the number *stable*, which is not the same as making
   it *robust*.

3. **The validation year is a single year.** Early stopping selected epoch 5 on the basis of
   2023 alone. A different year might have chosen differently.

4. **79,768 parameters exceeds the spec's "low tens of thousands" guide** (`D27`). Driven by
   the second LSTM layer at 33,280 on its own. Every architectural value is exactly as
   prescribed, so the count is reported rather than tuned toward the guide — but be ready to
   say that plainly.

5. **Hour t+24 lands at local midnight and is always masked out** (`D37`). The effective
   evaluated horizon is the daylight subset of the remaining 23 hours. A consequence of the
   t+1…t+24 definition with a midnight origin.

6. **The interpolation path is exercised only by tests** (`D14`). Real data had zero gaps.
   Eleven tests inject gaps of one to four hours, leading gaps, and a boundary-straddling
   gap — because untested code that never runs is code that will be wrong the first time
   NASA POWER serves a gap.

---

## Part 9. Quick reference

```bash
python run_all.py                  # everything, ~75 s on laptop CPU
python run_all.py --force-download # refetch from NASA POWER
python run_all.py --upper-bound    # include the perfect-forecast variant
python run_all.py --skip-gate1     # skip the synthetic validation
python -m pytest -q                # 13 tests
```

**Never edit `src/` to retune.** Every knob is in `config.yaml`. Retrain with:

```bash
python -m src.preprocess && python -m src.windowing && python -m src.train
python -m src.baseline && python -m src.evaluate && python -m src.analysis
```

**Never tune on the test rows of `metrics.csv`.** Watch `loss_log.csv`. Test is touched once.

### Numbers worth memorising

| | |
|---|---|
| Site | Port of Spain, Trinidad (10.65 °N, −61.50 °E) |
| Data | NASA POWER hourly, 2019–2024, 52,608 rows, 6 variables |
| Splits | train 2019–22 / val 2023 / test 2024 |
| Windows | 35,017 train / 363 val / 364 test |
| Encoder | (24, 11) — GHI + 5 weather + 5 calendar |
| Decoder | (24, 5) — calendar only |
| Parameters | 79,768 |
| Best epoch | 5 of 15 run |
| Headline skill | 0.264 (wet 0.286, dry 0.201) |
| Gate 2 checks | 25, all passing |
| Metric rows | 312 |

---

## Part 10. What is not finished

Two things future-Ben left open. Neither is a code problem.

1. **`report/main.tex` has unfilled placeholders**: `[YOUR NAME]`, `[ROLL NO]`,
   `[7/8/9]`, `[your-email]@op.iitg.ac.in`.

2. **Almost nothing is committed to git.** `git log` shows one commit — "Establish
   repository foundation." All of `src/`, `DECISIONS.md`, `config.yaml`, `run_all.py`,
   `results/`, and `report/` are untracked, and there is an unmerged
   `agent/repository-foundation` branch. Until that is fixed, a single careless command
   destroys everything above.

---

## Known erratum

`README.md` describes Gate 2 as "twenty-seven programmatic assertions." The actual count is
**25**: six per split × three splits = 18, plus three disjointness pairs, two chronology
checks, and two scaler-provenance checks. `results/validation_report.txt` confirms 25.
Worth correcting before the report is submitted, since the number appears in a document a
reviewer may check.
