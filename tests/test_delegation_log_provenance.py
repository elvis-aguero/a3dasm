"""Run-1 back door: orphaned ledger rows from a killed/cancelled delegation.

A delegation flushes provenance-stamped rows into the canonical ledger DURING
execution, but the terminal record() (DONE/FAILED) is written only at
COMPLETION. A delegation cancelled or killed mid-flight (wall/eval budget) left
ledgered rows with NO log entry — orphan evals that (a) broke the provenance
audit trail the critic rejects on and (b) escaped the eval-budget counter (the
v1 1523-vs-1000 overrun).

Fix: a RUNNING entry is logged at DISPATCH; _load_all collapses last-wins so the
terminal record supersedes it, and a killed delegation keeps its RUNNING entry
(still traceable). These tests pin that behaviour.
"""
from __future__ import annotations

from a3dasm._src.delegation_log import DelegationLog


def _started(log: DelegationLog, did: str, to_node: str = "implementer") -> None:
    log.record_started(
        id=did, from_node="strategizer", to_node=to_node,
        task="t", hypothesis_ids=[], started_at="2026-01-01T00:00:00+00:00")


def _done(log: DelegationLog, did: str, evals: int = 0) -> None:
    log.record(
        id=did, from_node="strategizer", to_node="implementer", task="t",
        deliverable="## Report\nok", hypothesis_ids=[],
        started_at="2026-01-01T00:00:00+00:00",
        completed_at="2026-01-01T00:05:00+00:00", status="DONE", evals=evals)


def test_killed_delegation_stays_traceable(tmp_path):
    """RUNNING-only (cancelled/killed before completion) → still ONE traceable
    record, not an invisible orphan."""
    log = DelegationLog(tmp_path / "delegation_log.jsonl")
    _started(log, "D003")                       # dispatched, then killed: no terminal
    recs = log.query_all()
    assert len(recs) == 1
    assert recs[0]["id"] == "D003"
    assert recs[0]["status"] == "RUNNING"       # traceable, not vanished


def test_terminal_record_supersedes_running(tmp_path):
    """A completed delegation collapses to its terminal record (last-wins),
    not two rows — consumers see one entry per id."""
    log = DelegationLog(tmp_path / "delegation_log.jsonl")
    _started(log, "D005")
    _done(log, "D005", evals=608)
    recs = log.query_all()
    assert len(recs) == 1
    assert recs[0]["status"] == "DONE"
    assert recs[0]["evals"] == 608
    assert log.last_completed_id("strategizer") == "D005"


def test_completion_order_preserved(tmp_path):
    """last_completed_id reflects terminal order even when a later-dispatched
    delegation finishes first."""
    log = DelegationLog(tmp_path / "delegation_log.jsonl")
    _started(log, "D001")
    _started(log, "D002")
    _done(log, "D002")            # D002 completes first
    _done(log, "D001")            # D001 completes later
    assert log.last_completed_id("strategizer") == "D001"
    assert [r["id"] for r in log.query_all()] == ["D002", "D001"]
