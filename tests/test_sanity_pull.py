import pandas as pd

from sanity_pull import FILL_VALUE, PARAMS, to_frame, validate


def make_payload() -> dict:
    index = pd.date_range("2023-01-01", periods=168, freq="h")
    keys = index.strftime("%Y%m%d%H")
    values = {}
    for parameter in PARAMS:
        if parameter == "ALLSKY_SFC_SW_DWN":
            values[parameter] = {
                key: 600.0 if 11 <= timestamp.hour <= 13 else 0.0
                for key, timestamp in zip(keys, index, strict=True)
            }
        else:
            values[parameter] = dict.fromkeys(keys, 1.0)
    return {"properties": {"parameter": values}}


def test_to_frame_builds_expected_hourly_table() -> None:
    frame = to_frame(make_payload())

    assert frame.shape == (168, 6)
    assert frame.index.is_monotonic_increasing
    assert not frame.isna().any().any()


def test_to_frame_replaces_fill_values() -> None:
    payload = make_payload()
    payload["properties"]["parameter"][PARAMS[1]]["2023010100"] = FILL_VALUE

    assert pd.isna(to_frame(payload).iloc[0][PARAMS[1]])


def test_validate_accepts_complete_week() -> None:
    payload = make_payload()
    ok, failures = validate(payload, to_frame(payload))

    assert ok
    assert failures == []


def test_validate_rejects_missing_parameter() -> None:
    payload = make_payload()
    del payload["properties"]["parameter"][PARAMS[-1]]
    ok, failures = validate(payload, to_frame(payload))

    assert not ok
    assert failures == [f"missing variables: {PARAMS[-1]}"]
