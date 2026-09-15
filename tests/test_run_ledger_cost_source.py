"""The ledger's cost and token columns come from telemetry, not the notebook.

Tokens and cost used to be parsed back out of `solution.md` or the stamped
`pipeline.ipynb`. Two consequences, both of which bite the ablation work:

1. A study with `pipeline_deliverable: false` has neither file, so the run
   recorded NO cost data at all (BACKLOG #31).
2. Only two of the four token fields were carried. The two dropped ones are
   the cache fields — precisely what a prompt-section change moves.

`debug/telemetry/summary.json` is written per LLM call regardless of
deliverable shape and carries all four, so it is the source of truth. The
deliverable-derived path stays as a fallback for runs predating telemetry and
must never overwrite a telemetry value.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_STUDIES = Path(__file__).resolve().parents[1] / "studies"
if str(_STUDIES) not in sys.path:
    sys.path.insert(0, str(_STUDIES))

import run_ledger  # noqa: E402

_TOTALS = {
    "calls": 4,
    "cost_calls": 4,
    "input_tokens": 1200,
    "output_tokens": 340,
    "cache_read_input_tokens": 90_000,
    "cache_creation_input_tokens": 5_000,
    "total_cost_usd": 1.2345,
    "total_tokens": 1540,
    "wall_time_s": 500.0,
}


def _run(tmp_path: Path, *, totals=None, status_extra=None) -> Path:
    run_dir = tmp_path / "studyX" / "runs" / "20260101T000000"
    debug = run_dir / "debug"
    debug.mkdir(parents=True)
    status = {"status": "GATED", "wall_s": 1234.5}
    status.update(status_extra or {})
    (debug / "run_status.json").write_text(json.dumps(status))
    if totals is not None:
        (debug / "telemetry").mkdir()
        (debug / "telemetry" / "summary.json").write_text(
            json.dumps({"totals": totals, "by_role": {}}))
    return run_dir


def test_a_run_with_no_notebook_still_records_its_cost(tmp_path):
    """The BACKLOG #31 regression: no deliverable used to mean no cost row."""
    row = run_ledger.extract(_run(tmp_path, totals=_TOTALS))

    assert row["cost_usd"] == 1.2345
    assert row["input_tokens"] == 1200
    assert row["output_tokens"] == 340


def test_the_cache_token_fields_are_recorded(tmp_path):
    """Dropped by the deliverable-derived path, and the ones a prompt-section
    ablation actually changes."""
    row = run_ledger.extract(_run(tmp_path, totals=_TOTALS))

    assert row["cache_read_tokens"] == 90_000
    assert row["cache_creation_tokens"] == 5_000


def test_an_unpriced_run_leaves_cost_blank_rather_than_zero(tmp_path):
    """Open-weight backends report no price. Blank means unmeasured; 0 would
    claim the run was free, which is a different and false statement."""
    totals = dict(_TOTALS, total_cost_usd=None, cost_calls=0)
    row = run_ledger.extract(_run(tmp_path, totals=totals))

    assert row["cost_usd"] == ""
    # tokens are still fully measured — only the price is unavailable
    assert row["input_tokens"] == 1200


def test_wall_s_comes_from_the_runs_own_close(tmp_path):
    """Preferred over telemetry's first-call-to-last-call span, which excludes
    setup and the final write."""
    row = run_ledger.extract(_run(tmp_path, totals=_TOTALS))

    assert row["wall_s"] == 1234.5


def test_a_run_predating_telemetry_still_reports_cost(tmp_path):
    """The deliverable-derived fallback must keep working for old run dirs."""
    run_dir = _run(tmp_path, totals=None)
    (run_dir.parent.parent / "solution.md").write_text(
        "- input_tokens: 111\n- output_tokens: 222\n"
        "- estimated_cost: $0.5000\n- time_used: 00:10:00\n"
    )

    row = run_ledger.extract(run_dir)

    assert row["input_tokens"] == "111"
    assert row["cost_usd"] == "0.5000"
    assert row["time_used"] == "00:10:00"


def test_the_deliverable_never_overwrites_a_telemetry_value(tmp_path):
    """Telemetry counts every call; the notebook stamp is a summary written
    once. Where they disagree, the instrument wins."""
    run_dir = _run(tmp_path, totals=_TOTALS)
    (run_dir.parent.parent / "solution.md").write_text(
        "- input_tokens: 111\n- output_tokens: 222\n"
        "- estimated_cost: $0.5000\n- time_used: 00:10:00\n"
    )

    row = run_ledger.extract(run_dir)

    assert row["input_tokens"] == 1200
    assert row["cost_usd"] == 1.2345
    # time_used has no telemetry equivalent, so it still comes from here
    assert row["time_used"] == "00:10:00"
