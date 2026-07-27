"""Next-day GHI forecasting pipeline for Port of Spain, Trinidad and Tobago.

Module order matches the data flow:

    ingest      fetch NASA POWER hourly data, Gate 0 data sanity
    synthetic   deterministic fixture series, Gate 1 pipeline validation
    preprocess  gap handling, calendar features, train-only scaling
    windowing   encoder and decoder windows, chronological splits, Gate 2 leakage audit
    baseline    24-hour persistence forecast
    model       LSTM encoder plus MLP head
    train       training loop with early stopping
    evaluate    daylight-masked metrics for both models
    analysis    the seven report figures

Support modules, additive to the handoff spec's section 5 list:

    config      single YAML loader, so no module carries a magic number
    gates       shared PASS or FAIL reporter for the three validation gates
"""

__all__ = [
    "analysis",
    "baseline",
    "config",
    "evaluate",
    "gates",
    "ingest",
    "model",
    "preprocess",
    "synthetic",
    "train",
    "windowing",
]
