"""Shared PASS or FAIL reporting for the three mandatory validation gates.

WHY the gates exist, and why they are code rather than comments: the project's
build protocol is sanity check a known input, validate the pipeline on synthetic
data, then touch real experimental data. A comment claiming "no leakage here"
cannot fail. An assertion can. Every gate in this pipeline is therefore a function
that computes a verdict from the data in front of it and halts the run when the
verdict is FAIL.

The three gates:

    Gate 0  data sanity, in ingest. Is the downloaded series physically plausible?
    Gate 1  synthetic pipeline validation, in synthetic. Does the machinery learn
            anything at all on a series whose answer we already know?
    Gate 2  leakage audit, in windowing. Can any sample see its own future?

A FAIL raises GateFailure, which run_all.py does not catch. Halting is the point:
a pipeline that continues past a failed gate produces numbers nobody should trust.
"""

from __future__ import annotations

import datetime as dt
import logging
from pathlib import Path
from typing import Iterable

LOGGER = logging.getLogger(__name__)

_SEPARATOR = "=" * 72


class GateFailure(RuntimeError):
    """Raised when a validation gate fails, to halt the pipeline."""


def reset_report(report_path: Path) -> None:
    """Start a fresh validation report for a new pipeline run.

    Args:
        report_path: file that the three gates append their verdicts to.

    WHY reset rather than append forever: the report is evidence for one run. A file
    accumulating verdicts across runs cannot answer "did this run pass".
    """
    report_path.parent.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().isoformat(timespec="seconds")
    header = (
        f"{_SEPARATOR}\n"
        f"VALIDATION REPORT, run started {stamp}\n"
        f"{_SEPARATOR}\n"
    )
    report_path.write_text(header, encoding="utf-8")
    LOGGER.info("validation report reset at %s", report_path)


def report(
    name: str,
    passed: bool,
    details: Iterable[str],
    report_path: Path,
    echo: bool = True,
) -> bool:
    """Record one gate's verdict to stdout and to the validation report.

    Args:
        name: gate label, for example "GATE 0, data sanity".
        passed: the computed verdict. Never pass a hard-coded True here.
        details: human-readable check lines, one per assertion, already formatted.
        report_path: file to append to.
        echo: whether to print the verdict line to stdout. Gate 1 sets this to False for
            the gates it reruns inside its sandbox, so the console shows one verdict per
            gate rather than Gate 2 appearing three times in a single run. The verdict is
            still written to the sandbox's own report either way.

    Returns:
        The verdict, unchanged, so callers can branch on the return value.
    """
    verdict = "PASS" if passed else "FAIL"
    detail_lines = list(details)

    block = [_SEPARATOR, f"{name}: {verdict}"]
    block.extend(f"    {line}" for line in detail_lines)
    block.append("")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("a", encoding="utf-8") as handle:
        handle.write("\n".join(block) + "\n")

    # The single PASS or FAIL line is a CLI summary, one of the two places the spec
    # permits print over logging. The detail lines go through logging.
    if echo:
        print(f"{name}: {verdict}")
    for line in detail_lines:
        LOGGER.info("%s | %s", name, line)

    return passed


def require(
    name: str,
    passed: bool,
    details: Iterable[str],
    report_path: Path,
    echo: bool = True,
) -> None:
    """Record a gate's verdict and halt the pipeline when it failed.

    Args:
        name: gate label.
        passed: the computed verdict.
        details: human-readable check lines.
        report_path: file to append to.
        echo: whether to print the verdict line to stdout. A FAIL always prints, because a
            halting failure must be visible on the console whatever the caller asked for.

    Raises:
        GateFailure: when `passed` is False.
    """
    if not report(name, passed, details, report_path, echo=echo or not passed):
        raise GateFailure(
            f"{name} failed. See the validation report at {report_path}. "
            "Fix the cause before running any further stage."
        )


def check(label: str, passed: bool, observed: object) -> tuple[bool, str]:
    """Format one assertion into a verdict and a report line.

    Args:
        label: what was checked, in words.
        passed: whether it held.
        observed: the value actually seen, included so a FAIL is diagnosable.

    Returns:
        A pair of the verdict and its formatted report line.

    WHY the observed value is mandatory: a report line reading "row count: FAIL"
    sends a reader back to the debugger. One reading "row count: FAIL, observed
    52584" tells them 24 hours are missing.
    """
    mark = "ok  " if passed else "FAIL"
    return passed, f"[{mark}] {label}: {observed}"
