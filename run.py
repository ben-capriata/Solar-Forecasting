"""Run the whole experiment: python run.py

Add --upper-bound to also train a variant that is given tomorrow's observed weather.
That variant is not a forecast. It shows what a perfect weather feed would be worth.
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import checks
import forecast as fc


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upper-bound", action="store_true", help="also train the perfect-weather variant")
    parser.add_argument("--skip-gate1", action="store_true", help="skip the synthetic check for a faster rerun")
    args = parser.parse_args()
    started = time.time()
    results = Path("results")
    (results / "predictions").mkdir(parents=True, exist_ok=True)
    (results / "checkpoints").mkdir(exist_ok=True)

    # 1. Load the data and check it is physically plausible.
    checks.start_report()
    raw = pd.read_parquet(fc.DATA_PATH)
    checks.gate0(raw)
    df = fc.add_features(raw)

    # 2. Prove the pipeline works on a series whose answer is known.
    if not args.skip_gate1:
        checks.gate1()

    # 3. Cut the windows and audit them for leakage before any training.
    x_scaler, y_scaler = fc.fit_scalers(df)
    windows = {split: fc.make_windows(df, x_scaler, y_scaler, split) for split in fc.SPLITS}
    checks.gate2(df, windows, x_scaler, y_scaler)
    for split, window in windows.items():
        print(f"{split:<5} {len(window['target']):>6} windows")

    # 4. Train the LSTM. The test year is not touched here.
    model, history = fc.train(windows)
    history.to_csv(results / "loss_log.csv", index=False)
    torch.save(model.state_dict(), results / "checkpoints" / "lstm_best.pt")

    # 5. Score persistence and the LSTM on identical windows.
    predictions = {
        split: {"persistence": fc.persistence(df, windows[split]),
                "lstm": fc.predict(model, windows[split], y_scaler)}
        for split in ("val", "test")
    }
    metrics = pd.concat([fc.evaluate(predictions[split], windows[split], split) for split in ("val", "test")],
                        ignore_index=True)
    metrics.to_csv(results / "metrics.csv", index=False)

    # 6. Optional: what would tomorrow's weather be worth? Not a forecast.
    if args.upper_bound:
        print("\nTraining the upper bound, which is given tomorrow's observed weather...")
        with_weather = {split: fc.make_windows(df, x_scaler, y_scaler, split, future_inputs=fc.WEATHER + fc.CALENDAR)
                        for split in fc.SPLITS}
        upper_model, _ = fc.train(with_weather)
        for split in ("val", "test"):
            predictions[split]["lstm_upperbound"] = fc.predict(upper_model, with_weather[split], y_scaler)
        metrics = pd.concat([fc.evaluate(predictions[split], windows[split], split) for split in ("val", "test")],
                            ignore_index=True)
        metrics.to_csv(results / "metrics_upperbound.csv", index=False)

    np.savez(results / "predictions" / "test.npz", actual=windows["test"]["actual"],
             times=windows["test"]["times"], **predictions["test"])
    report = fc.summary(metrics)
    (results / "summary.txt").write_text(report + "\n")
    print("\n" + report)
    print(f"\nFinished in {time.time() - started:.0f} seconds.")


if __name__ == "__main__":
    main()
