"""Run the entire pipeline from raw Parquet to final figures.

Stage order is not arbitrary. It encodes the project's build protocol:

    1. Ingest and GATE 0        Is the real data physically plausible? Sanity check a
                                known input before anything else happens.
    2. GATE 1, synthetic        Does the machinery learn a series whose answer is already
                                known? This runs BEFORE any real-data training, so that a
                                later disappointing result cannot be confused with a bug.
    3. Preprocess               Gap handling and train-only scaling on the real data.
    4. Windowing and GATE 2     Cut the supervised windows, then audit them for leakage
                                before a single gradient step is taken on them.
    5. Baseline                 The persistence forecast, which needs no training.
    6. Train                    Fit the LSTM. The test year is not touched here.
    7. Evaluate                 Daylight-masked metrics. The test year is touched once.
    8. Analysis                 The seven report figures.

Any gate raising GateFailure propagates out of this script uncaught, because halting is
the entire point of a gate. A pipeline that carries on past a failed gate produces numbers
that look finished and are not trustworthy.

Usage:
    python run_all.py                  full pipeline, reusing a cached raw Parquet
    python run_all.py --force-download refetch the six years from NASA POWER
    python run_all.py --skip-gate1     skip only the synthetic gate, for a faster rerun
"""

from __future__ import annotations

import argparse
import logging
import time

from src import analysis, baseline, evaluate, ingest, preprocess, synthetic, train, windowing
from src.config import Config, configure_logging
from src.gates import reset_report

LOGGER = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    """Execute every stage in order.

    Args:
        argv: command line arguments, defaults to sys.argv.

    Returns:
        Process exit code, 0 on success. A gate failure raises rather than returning.
    """
    parser = argparse.ArgumentParser(
        description="Run the full next-day GHI forecasting pipeline end to end."
    )
    parser.add_argument(
        "--config", default=None, help="path to config.yaml, defaults to project root"
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="delete any cached raw Parquet and refetch from NASA POWER",
    )
    parser.add_argument(
        "--skip-gate1",
        action="store_true",
        help="skip the synthetic pipeline validation, for a faster rerun of a green pipeline",
    )
    parser.add_argument(
        "--day-selection",
        choices=("cloud", "clearness"),
        default="cloud",
        help="how figures 1 and 2 rank test days",
    )
    parser.add_argument(
        "--upper-bound",
        action="store_true",
        help=(
            "additionally train the spec section 13 perfect-forecast variant, which is "
            "given tomorrow's observed weather. Quantifies what an NWP feed would be "
            "worth. Never the headline model."
        ),
    )
    args = parser.parse_args(argv)

    configure_logging()
    config = Config.load(args.config)
    started = time.monotonic()

    # A fresh report per run. The file is evidence for this run, not a running log.
    reset_report(config.path("validation_report"))

    LOGGER.info("stage 1 of 8: ingest and Gate 0 data sanity")
    if args.force_download:
        raw_path = config.path("raw_parquet")
        if raw_path.exists():
            LOGGER.warning("--force-download given, removing %s", raw_path)
            raw_path.unlink()
    raw = ingest.load_raw(config)

    if args.skip_gate1:
        LOGGER.warning(
            "stage 2 of 8: GATE 1 SKIPPED at the caller's request, the pipeline is "
            "unvalidated on synthetic data for this run"
        )
    else:
        LOGGER.info("stage 2 of 8: Gate 1 synthetic pipeline validation")
        synthetic.gate1(config)

    LOGGER.info("stage 3 of 8: preprocess")
    features = preprocess.run(config, raw=raw)

    LOGGER.info("stage 4 of 8: windowing and Gate 2 leakage audit")
    splits = windowing.run(config, features=features)

    LOGGER.info("stage 5 of 8: persistence baseline")
    baseline.run(config, features=features, splits=splits)

    LOGGER.info("stage 6 of 8: train the LSTM")
    train.run(config, splits=splits)

    if args.upper_bound:
        LOGGER.info(
            "optional stage: train the perfect-forecast upper bound, spec section 13"
        )
        train.run(config, splits=splits, upper_bound=True)

    LOGGER.info("stage 7 of 8: evaluate")
    metrics = evaluate.run(config, splits=splits, features=features)

    LOGGER.info("stage 8 of 8: figures")
    figures = analysis.run(
        config,
        day_selection=args.day_selection,
        splits=splits,
        features=features,
        metrics=metrics,
    )

    elapsed = time.monotonic() - started
    print()
    print(evaluate.format_summary(config, metrics))
    print()
    print(f"Pipeline complete in {elapsed:.1f}s. Outputs:")
    print(f"  {config.path('validation_report')}")
    print(f"  {config.path('metrics_csv')}")
    print(f"  {config.path('summary_txt')}")
    print(f"  {config.path('checkpoint')}")
    print(f"  {len(figures)} figures in {config.dir_path('figures_dir')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
