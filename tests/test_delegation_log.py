"""Tests for DelegationLog."""

from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

from a3dasm._src.delegation_log import DelegationLog


def _make_log(tmp_path: Path) -> DelegationLog:
    return DelegationLog(tmp_path / "delegation_log.jsonl")


def _record(log: DelegationLog, **kwargs) -> None:
    defaults = dict(
        id="D001",
        from_node="strategizer",
        to_node="implementer",
        task="Do something",
        deliverable="It is done.",
        hypothesis_ids=["H1"],
        started_at="2026-06-01T12:00:00+00:00",
        completed_at="2026-06-01T12:05:00+00:00",
        status="DONE",
        tokens_in=10,
        tokens_out=5,
        cost_usd=0.001,
    )
    defaults.update(kwargs)
    log.record(**defaults)


# ---------------------------------------------------------------------------
# Basic record / field correctness
# ---------------------------------------------------------------------------


def test_record_single_delegation(tmp_path):
    log = _make_log(tmp_path)
    _record(log)
    path = tmp_path / "delegation_log.jsonl"
    assert path.exists()
    lines = path.read_text().strip().splitlines()
    assert len(lines) == 1
    rec = json.loads(lines[0])
    assert rec["id"] == "D001"
    assert rec["from_node"] == "strategizer"
    assert rec["to_node"] == "implementer"
    assert rec["hypothesis_ids"] == ["H1"]
    assert rec["status"] == "DONE"
    assert rec["tokens_in"] == 10
    assert rec["tokens_out"] == 5
    assert rec["cost_usd"] == 0.001


def test_record_stores_full_task(tmp_path):
    long_task = "A" * 2000
    log = _make_log(tmp_path)
    _record(log, task=long_task)
    path = tmp_path / "delegation_log.jsonl"
    rec = json.loads(path.read_text().strip())
    assert rec["task"] == long_task


def test_record_stores_full_deliverable(tmp_path):
    long_deliverable = "B" * 3000
    log = _make_log(tmp_path)
    _record(log, deliverable=long_deliverable)
    path = tmp_path / "delegation_log.jsonl"
    rec = json.loads(path.read_text().strip())
    assert rec["deliverable"] == long_deliverable


# ---------------------------------------------------------------------------
# query_received
# ---------------------------------------------------------------------------


def test_query_received_filters_by_to_node(tmp_path):
    log = _make_log(tmp_path)
    _record(log, id="D001", to_node="implementer")
    _record(log, id="D002", to_node="debugger")
    _record(log, id="D003", to_node="implementer")

    result = log.query_received("implementer")
    assert len(result) == 2
    assert all(r["to_node"] == "implementer" for r in result)


def test_query_received_returns_n_newest_oldest_first(tmp_path):
    log = _make_log(tmp_path)
    for i in range(1, 6):
        _record(log, id=f"D{i:03d}", to_node="implementer")

    result = log.query_received("implementer", n=3)
    assert len(result) == 3
    # oldest-first among the last 3: D003, D004, D005
    assert result[0]["id"] == "D003"
    assert result[1]["id"] == "D004"
    assert result[2]["id"] == "D005"


def test_query_received_returns_all_when_n_is_none(tmp_path):
    log = _make_log(tmp_path)
    for i in range(1, 8):
        _record(log, id=f"D{i:03d}", to_node="implementer")

    result = log.query_received("implementer", n=None)
    assert len(result) == 7


def test_query_received_empty_when_no_records(tmp_path):
    log = _make_log(tmp_path)
    result = log.query_received("implementer")
    assert result == []


# ---------------------------------------------------------------------------
# last_completed_id
# ---------------------------------------------------------------------------


def test_last_completed_id_returns_most_recent_done(tmp_path):
    log = _make_log(tmp_path)
    _record(log, id="D001", from_node="strategizer", status="DONE")
    _record(log, id="D002", from_node="strategizer", status="DONE")

    result = log.last_completed_id("strategizer")
    assert result == "D002"


def test_last_completed_id_ignores_failed(tmp_path):
    log = _make_log(tmp_path)
    _record(log, id="D001", from_node="strategizer", status="DONE")
    _record(log, id="D002", from_node="strategizer", status="FAILED")

    result = log.last_completed_id("strategizer")
    assert result == "D001"


def test_last_completed_id_returns_none_when_empty(tmp_path):
    log = _make_log(tmp_path)
    result = log.last_completed_id("strategizer")
    assert result is None


# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------


def test_record_is_thread_safe(tmp_path):
    log = _make_log(tmp_path)
    n_threads = 20
    errors: list[Exception] = []

    def write_record(i: int) -> None:
        try:
            _record(log, id=f"D{i:03d}", from_node="strategizer")
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=write_record, args=(i,)) for i in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"Thread errors: {errors}"
    path = tmp_path / "delegation_log.jsonl"
    lines = path.read_text().strip().splitlines()
    assert len(lines) == n_threads
    # All lines must be valid JSON
    for line in lines:
        json.loads(line)


# ---------------------------------------------------------------------------
# is_falsification_attempt field
# ---------------------------------------------------------------------------


def test_record_stores_falsification_flag(tmp_path):
    log = _make_log(tmp_path)
    log.record(
        id="D001",
        from_node="s",
        to_node="i",
        task="t",
        deliverable="d",
        hypothesis_ids=["H1"],
        started_at="x",
        completed_at="y",
        status="DONE",
        is_falsification_attempt=True,
    )
    rec = log.query_received("i")[0]
    assert rec["is_falsification_attempt"] is True


def test_record_flag_defaults_false(tmp_path):
    log = _make_log(tmp_path)
    _record(log)
    rec = log.query_received("implementer")[0]
    assert "is_falsification_attempt" in rec
    assert rec["is_falsification_attempt"] is False


# ---------------------------------------------------------------------------
# query_all
# ---------------------------------------------------------------------------


def test_query_all_returns_every_record_oldest_first(tmp_path):
    log = _make_log(tmp_path)
    _record(log, id="D001", to_node="implementer", status="DONE")
    _record(log, id="D002", to_node="debugger", status="FAILED")
    _record(log, id="D003", to_node="implementer", status="DONE")

    result = log.query_all()
    assert len(result) == 3
    assert result[0]["id"] == "D001"
    assert result[1]["id"] == "D002"
    assert result[2]["id"] == "D003"


def test_query_all_returns_empty_when_no_records(tmp_path):
    log = _make_log(tmp_path)
    assert log.query_all() == []


# ---------------------------------------------------------------------------
# query_received coerces a string n (MCP string-in tools pass "6")
# ---------------------------------------------------------------------------

def test_query_received_coerces_string_n(tmp_path):
    """Regression (run 20260708T021335, critic-3): RecallHistory(n='6') reached
    query_received, whose `matching[-n:]` raised 'bad operand type for unary -:
    str'. A string n must be coerced, not crash."""
    log = _make_log(tmp_path)
    for i in range(4):
        _record(log, id=f"D00{i}", to_node="implementer")
    # string n (as an MCP tool would pass) must not raise and must honor the count
    got = log.query_received("implementer", "2")
    assert len(got) == 2
    # garbage n degrades gracefully to "all", not a crash
    assert len(log.query_received("implementer", "not-a-number")) == 4
