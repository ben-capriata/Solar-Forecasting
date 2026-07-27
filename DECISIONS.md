# Decisions

Every choice the handoff spec left open, with a one-line rationale. Organised by the
stage it affects. Numbered so the report and the viva can cite them.

## Repository and environment

**D1. Built in the existing repository root, not a nested `solar-forecasting-tt/`.**
The spec names that directory, but this repo already is the git repository and already
holds the legacy assets the spec says to reuse. A nested duplicate would be worse for a
reader.

**D2. Two support modules beyond section 5's nine: `src/config.py` and `src/gates.py`.**
Section 5's nine named modules all exist. `config.py` is the single YAML loader that makes
"no magic numbers in code" achievable, and `gates.py` is the shared PASS or FAIL reporter
used by all three gates. The alternative, reaching for config and reporting helpers out of
`ingest.py`, would invert the dependency direction and read worse.

**D3. Python 3.12.8 rather than exactly 3.11.** The spec requires 3.11 or newer, and the
project's existing virtual environment is 3.12.8.

**D4. Four dependencies added beyond what the repo already had: `torch`, `scikit-learn`,
`matplotlib`, `PyYAML`.** All four are named in spec section 3, so this is a record rather
than an expansion. PyYAML is the one the spec implies rather than names, and it is required
because section 3 mandates a `config.yaml`. Versions pinned in `requirements.txt`.

**D5. `.gitignore` amended to stop excluding the deliverables.** It ignored `*.png`
wholesale, which would have swallowed the seven report figures that section 15 lists as
deliverables. `results/figures/*.png` is now exempted. Parquet, npz, pickle, and the
prediction directory stay ignored, since all are reproducible from `run_all.py`.

## Ingest and Gate 0

**D6. `full_pull.py` did not exist and was written from scratch inside `src/ingest.py`.**
Spec section 4 lists it as an existing asset. The repo held `sanity_pull.py`, which is
validated and was refactored as instructed, and `new_pull.py`, which pulls a single
variable for one year and plots interactively. The year-by-year loop with continuity
checks is therefore new code, built on `sanity_pull.py`'s verified `fetch` and `to_frame`.

**D7. Gate 0 expected row count is 52608 with a tolerance of zero.** 2019-01-01 through
2024-12-31 inclusive is 2192 days, including two leap years, times 24 hours. The real
download matched exactly, so no tolerance was needed. The tolerance is a config value, so
a future documented gap can raise it without touching code.

**D8. No columns were dropped by Gate 0.** All six POWER variables returned with zero
missing cells across all 52608 hours, so the five percent missing rule never triggered.
`CLOUD_AMT` surviving is why figures 1 and 2 can rank days by observed cloud amount.

**D9. Retry with backoff added to the API call, three attempts.** Six sequential year
requests with no retry policy is a single transient timeout away from a failed run. The
polite two second sleep between requests is preserved exactly as the spec requires.

**D10. A cached Parquet that loads but fails Gate 0 halts the run rather than refetching.**
The spec forbids silent re-downloads when the file passes Gate 0. Halting on a file that
fails is the stricter reading: a corrupt cache is a fact the operator should see, not
something to paper over with a fresh download.

**D11. The GHI units metadata string is persisted beside the Parquet.** Recorded as
`Wh/m^2`. Hourly Wh/m^2 is numerically equal to mean W/m^2 over the hour, which is why
every figure and metric is labelled W/m^2 with no conversion factor.

## Preprocessing

**D12. Interpolation runs within each split segment independently, never across the whole
series.** This is a red team finding that the spec's own three gates would not catch. A gap
straddling 2022-12-31 23:00 to 2023-01-01 01:00, filled globally, computes a training row
from a validation observation and a validation row from a training observation. That is
leakage in both directions and it is invisible to a timestamp-ordering assertion.
`tests/test_preprocess.py::test_interpolation_never_crosses_a_split_boundary` fails if this
is ever reverted.

**D13. Long gaps are left entirely NaN rather than partially filled.** Pandas
`interpolate(limit=3)` fills the first three hours of a five hour gap, leaving a partly
synthetic run that no longer looks like a gap to the window builder. The implementation
interpolates without a limit, then restores NaN across every run longer than the allowance.

**D14. The real data contained zero gaps, so the interpolation path is exercised only by
tests.** Eleven tests in `tests/test_preprocess.py` inject gaps of one to four hours,
leading gaps, and a boundary-straddling gap, because untested code that never runs is code
that will be wrong the first time NASA POWER serves a gap.

**D15. `features.parquet` holds unscaled physical units, and scaling is a separate
artefact.** The baseline and evaluator can then read true W/m^2 with no inverse transform,
and the fitted statistics exist as an auditable file rather than transient state.

**D16. All model input features are scaled by the feature scaler, calendar features
included.** The spec says "StandardScaler per feature" without enumerating which. Scaling
the sine and cosine pairs is harmless, because both components of a pair have equal
variance over a full cycle and therefore scale by the same factor, preserving the circle.

**D17. The scaler fit index bounds are persisted.** Spec Gate 2 asks for an assertion that
the fit received only train rows. An assertion cannot inspect a call that already returned,
so the call records what it was given and the gate checks the record. Without this the
claim would rest on reading the source, which is the unfalsifiable kind of check the gates
exist to replace.

**D18. Scalers are pickled as a plain dict, not as the `ScalerBundle` dataclass.** Pickle
stores a class by import path, and `python -m src.preprocess` makes that path
`__main__.ScalerBundle`, which no other module can then load. Found by an actual failure,
not by inspection.

## Windowing

**D19. Training stride 1, evaluation stride 24 at local midnight origins.** Spec section 8
permits stride 1 for training only. Dense training origins give roughly 35000 windows
instead of 1460, which matters for a model with 79768 parameters. Validation and test use
stride 24 so both early stopping and final evaluation measure the operational task: one
day-ahead forecast issued once per day.

**D20. Split assignment is by whole-window containment, not by the origin's year.** A
window originating 2024-01-01 00:00 draws its encoder from 2023. Assigning it to test by
origin year would make the test and validation index ranges overlap, and the spec's own
Gate 2 requirement that they be disjoint would be false. Containment is what makes that
assertion true. Cost: 2 windows per year boundary, 47 at the four internal train
boundaries, all counted and logged. Worth stating in the viva that this is stricter than
operationally necessary, since an encoder reading 2023 to forecast 2024 is using strictly
past-observed data exactly as a real operator would. Disjointness was chosen over that
extra window because the spec asks for disjointness explicitly.

**D21. Windows saved as a single compressed npz plus a separate CSV origin index.** The
seasonal and hour-of-day analyses join metrics back to calendar dates, and a CSV is the
format a human can open to check that join.

**D22. Realised window counts: 35017 train, 363 validation, 364 test.** Zero windows were
dropped for touching a NaN, because the source data has no gaps.

## Gate 1

**D23. Gate 1 writes every artefact into `gate1_sandbox/`, including its own validation
report.** A gate run that overwrote the real `features.parquet` or the real checkpoint
would corrupt the experiment it exists to protect. The report is redirected too, because
Gate 1 internally reruns Gate 0 and Gate 2 on the fixture, and those sandbox verdicts in
the real report would make it read as though Gate 2 ran three times. Gate 1's own verdict
is written to the real report by the caller.

**D24. Two fixture variants: noise-free for the persistence assertion, noisy for the
learning assertion.** Persistence must achieve near-zero error on a smooth periodic
series, and the model needs something non-trivial to reduce.

**D25. Gate 1 thresholds set with wide margins to avoid flakiness.** Persistence limit
5.0 W/m^2 against an achieved 1.35, and validation MSE limit 0.15 against an achieved
0.00029 reached by epoch 1. A margin of three orders of magnitude means a green pipeline
will not report a spurious FAIL.

**D26. Gate 1 uses training stride 4 and a 5 epoch cap.** "Within a few epochs" per the
spec, and the coarser stride keeps the mandatory gate to a few seconds. Patience equals
the epoch cap, so early stopping cannot cut the demonstration short.

## Model and training

**D27. Realised parameter count is 79768, above the spec's "low tens of thousands"
guide.** Driven by the second LSTM layer, which is 33280 parameters on its own, and by the
184-wide fusion input to the head. Every architectural value is exactly as the spec
prescribes, so the count is reported rather than tuned toward the guide.

**D28. Loss stays plain MSE over all 24 hours, and a daylight-masked validation MAE is
logged alongside it.** Roughly half of each target window is night, where GHI is near zero
and trivially predictable, so plain MSE spends about half its signal on hours no model can
get wrong. The spec prescribes plain MSE and plain MSE is what early stopping watches. The
extra logged metric surfaces the concern for the report without quietly overriding the
specified design.

**D29. CPU-only execution, `num_workers=0`, deterministic algorithms enabled.** cuDNN LSTM
kernels are not deterministic, so CPU is a correctness choice. This combination reproduces
bit-exactly, verified across two clean end-to-end runs.

**D30. Early stopping fired at epoch 15, restoring epoch 5.** Validation loss bottomed at
0.057309 and rose steadily afterwards while training loss kept falling, which is textbook
overfitting and visible in `fig6_loss_curve.png`.

## Evaluation

**D31. LSTM predictions are clipped at zero before any metric is computed.** Irradiance
cannot be negative, the same floor was already applied to the GHI inputs, and persistence
is composed of real observations so it can never go negative. Leaving clipping off would
charge the model for an error the baseline is structurally incapable of making.

**D32. Masked and unmasked variants are distinguished by metric name, as `rmse_masked` and
`rmse_unmasked`.** The prescribed `metrics.csv` schema has no mask column, and the spec's
hard rules forbid adding columns to it.

**D33. MAPE is reported for daylight hours only and has no unmasked counterpart.** Unmasked
hours include exact zeros at night, so a percentage error against them is undefined. Stated
in `summary.txt` rather than silently omitted.

**D34. MAPE is a weak discriminator here and should be read with caution.** Persistence
scores 21.64 percent and the LSTM 21.01 percent, nearly identical, while their RMSE differs
by 26 percent. MAPE divides by the observed value, so hours barely above the 20 W/m^2
threshold dominate it. The report should lead with RMSE and skill score.

**D35. Hour-of-day and per-month breakdowns are encoded in the metric name.** Same reason
as D32: the schema is fixed at five columns.

**D36. Skill score per group uses that group's own persistence RMSE as denominator.** A
seasonal skill score computed against the overall persistence RMSE would be meaningless.

**D37. Target hour t+24 lands at local midnight and is therefore always masked out.** The
effective evaluated horizon is the daylight subset of the 23 remaining hours. A consequence
of the spec's t+1 to t+24 definition with a midnight origin, noted for completeness.

**D38. Validation is evaluated alongside test in `metrics.csv`.** The prescribed schema has
a `split` column, so populating it costs nothing and lets a reader confirm that the
validation and test pictures agree.

## Figures

**D39. Two categorical hues, blue for the LSTM and orange for persistence, with observed
GHI in neutral ink.** The pair was checked with a colour-vision-deficiency validator, not
by eye: worst-pair deltaE 24.7 under protanopia against a target of 8, both above 3:1
contrast on the figure surface. Observed data is not a competing model, so it does not take
a categorical hue.

**D40. Line style and marker shape duplicate the colour encoding.** IEEE reports are
frequently printed in greyscale, where a hue-only distinction disappears. Solid with round
markers for the LSTM, dashed with square markers for persistence.

**D41. `fig5_scatter.png` is two panels rather than one superimposed scatter.** 4229 points
per model overplot into an unreadable mass. The panels share identical axes and identical
identity lines, so the comparison stays honest.

**D42. Day selection for figures 1 and 2 ranks by observed daylight-mean `CLOUD_AMT`, with
a clearness-index fallback.** The fallback exists because Gate 0 can drop that column; it
ranks each day's daylight GHI against the climatological mean for the same day of year, so
the seasonal envelope cancels. Selectable with `--day-selection clearness`. Chosen days:
clearest 2024-09-05, most clouded 2024-08-24.

**D43. Figure 3 uses lines, figure 4 uses grouped bars.** Hour of day is an ordered
continuum where the shape of the curve is the message; twelve months are discrete
categories that read better as bars.

## Optional stage

**D44. The section 13 perfect-forecast upper bound was implemented, run, and is reported
separately from the headline.** Enabled with `python run_all.py --upper-bound`, or
`python -m src.train --upper-bound` on its own. It receives the observed weather for the
target window as decoder input, in addition to the calendar features. Observed GHI is
deliberately excluded, since that would be handing over the answer rather than a weather
feed. Result on the test year: masked RMSE 62.11 W/m^2 and skill 0.496, against the primary
model's 90.78 and 0.264. The interpretation for the report is that a hypothetical perfect
NWP feed would roughly double the skill gain over persistence, which is a concrete number
to put against the cost of an NWP subscription. It carries 95128 parameters rather than
79768, because the wider decoder widens the fusion layer.

**D45. `run_all.py` treats the upper bound as opt-in, not default.** The spec calls it an
optional stage conditional on everything else being green, and leaving it out of the default
run keeps the headline pipeline's output free of a variant that is not a forecast.
