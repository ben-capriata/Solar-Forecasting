# Decisions

Every discretionary choice in the project, with the reason for it.

## Data

1. **NASA POWER hourly point data, 2019 to 2024, local solar time.** Free and global.
   Local solar time puts solar noon at hour 12, so the hour-of-day features line up with the sun.
2. **No unit conversion.** POWER serves hourly GHI in Wh/m^2, which equals the mean W/m^2
   over that hour.
3. **The data had no gaps.** All 52,608 hours are present for all six variables. Any window
   that touched a gap would be dropped, not filled.

## Features and windows

4. **Past-observed inputs only.** The model sees the previous 24 hours of GHI and weather.
   It never sees tomorrow's weather, because that would assume the forecast feed the problem
   says is missing.
5. **Tomorrow's calendar is allowed.** Hour of day and day of year are known in advance and
   need no forecast.
6. **Sine and cosine encoding.** Hour 23 and hour 0 are neighbours. Integers would put them
   far apart.
7. **Split by calendar year, never at random.** Train 2019 to 2022, validate 2023, test 2024.
   A window belongs to a split only if all 48 of its hours fall inside that split's years.
8. **Training uses every hour as an origin. Evaluation uses one origin per day at midnight.**
   This gives 35,017 training windows, and 363 validation and 364 test forecasts that match
   the real task of one forecast issued per night.
9. **Scalers are fitted on the training years only.** Fitting on all years would leak
   information about the future into training.

## Model and training

10. **A 2-layer LSTM with 64 hidden units, dropout 0.2, and a 128-unit head.** Small on
    purpose: one site and four training years do not support a large model. 79,768 parameters.
11. **All 24 hours are predicted at once.** A step-by-step decoder would feed its own
    predictions back in and compound errors.
12. **Adam, learning rate 0.001, batch size 64, plain MSE over all 24 hours.**
13. **Early stopping on validation loss with patience 10.** Training stopped at epoch 15 and
    the weights from epoch 5 were restored.
14. **CPU only with fixed seeds.** Reruns reproduce the same numbers.

## Evaluation

15. **Persistence is the baseline.** It is what an operator without a weather feed would use,
    and skill score is defined against it.
16. **Daylight mask at 20 W/m^2.** Night hours are zero and would flatter every model.
17. **Predictions are floored at zero.** Irradiance cannot be negative.
18. **RMSE and skill lead, MAPE is secondary.** MAPE divides by the observed value, so the
    dim hours near dawn and dusk dominate it, and both models are similar there.
19. **Target hour t+24 is local midnight and is always masked out.**
20. **The test year is touched once**, after all choices were fixed on the validation year.

## Upper bound

21. **An optional run is given tomorrow's observed weather.** It is not a forecast and is
    never the headline. It shows what a perfect weather feed would be worth: skill 0.496
    against 0.264.
