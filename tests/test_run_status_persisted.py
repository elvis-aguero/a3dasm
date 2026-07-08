"""Regression: a NORMALLY-closed run left no debug/run_status.json — the gate
outcome was written only to the notebook metadata + the ledger, so the §1 analysis
protocol's first KPI (gate outcome ← run_status.json) was unreadable for every
clean run (3 GATED runs in a row had no run_status.json). The close path now
persists it via AgenticRun._write_run_status on every terminal state.
"""
from __future__ import annotations

import json

from a3dasm._src.agent_runtime import AgenticRun


def test_write_run_status_persists_gate_outcome(tmp_path):
    AgenticRun._write_run_status(
        tmp_path, status="GATED", model="claude-haiku-4-5-20251001",
        evals_used=826, run=str(tmp_path), thread_id="t1",
    )
    data = json.loads((tmp_path / "run_status.json").read_text())
    assert data["status"] == "GATED"          # the field §1 reads first
    assert data["evals_used"] == 826
    assert data["model"] == "claude-haiku-4-5-20251001"


def test_write_run_status_handles_each_terminal_state(tmp_path):
    for outcome in ("GATED", "UNGATED", "FAILED", "crashed", "watchdog_killed"):
        AgenticRun._write_run_status(tmp_path, status=outcome)
        assert json.loads((tmp_path / "run_status.json").read_text())["status"] == outcome


def test_write_run_status_never_raises_on_bad_dir(tmp_path):
    # Best-effort contract: a status write must never break a run, even if the
    # debug dir is missing (OSError swallowed).
    AgenticRun._write_run_status(tmp_path / "does_not_exist", status="GATED")
    # no exception == pass
