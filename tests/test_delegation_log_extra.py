"""Extra tests for DelegationLog — covers remaining uncovered lines."""
from __future__ import annotations

import json
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
# last_completed_id when only FAILED records exist → returns None
# ---------------------------------------------------------------------------


def test_last_completed_id_returns_none_when_only_failed(tmp_path):
    """last_completed_id returns None when only FAILED delegations are present."""
    log = _make_log(tmp_path)
    _record(log, id="D001", from_node="strategizer", status="FAILED")
    _record(log, id="D002", from_node="strategizer", status="FAILED")

    result = log.last_completed_id("strategizer")
    assert result is None


# ---------------------------------------------------------------------------
# _load_all skips corrupt JSON lines gracefully
# ---------------------------------------------------------------------------


def test_load_all_skips_corrupt_json_lines(tmp_path):
    """_load_all silently skips lines that are not valid JSON."""
    log_path = tmp_path / "delegation_log.jsonl"
    log = DelegationLog(log_path)

    # Write one valid record first
    _record(log)

    # Manually append a corrupt line
    with log_path.open("a", encoding="utf-8") as f:
        f.write("THIS IS NOT JSON\n")

    # Also append a second valid record
    _record(log, id="D002", status="DONE")

    # query_received should silently skip the corrupt line
    result = log.query_received("implementer")
    # Only the 2 valid records should be returned
    assert len(result) == 2
    ids = [r["id"] for r in result]
    assert "D001" in ids
    assert "D002" in ids


# ---------------------------------------------------------------------------
# _now_iso is callable and returns an ISO string
# ---------------------------------------------------------------------------


def test_now_iso_returns_iso_string():
    """_now_iso() returns a valid ISO 8601 timestamp string."""
    from a3dasm._src.delegation_log import _now_iso
    result = _now_iso()
    assert isinstance(result, str)
    # Should contain 'T' as the date/time separator
    assert "T" in result
    # Should be parseable
    from datetime import datetime
    # Should not raise
    datetime.fromisoformat(result)


# ---------------------------------------------------------------------------
# record() with cost_usd=None stores null in JSON
# ---------------------------------------------------------------------------


def test_record_stores_null_cost_usd(tmp_path):
    """record() with cost_usd=None stores JSON null for cost_usd field."""
    log = _make_log(tmp_path)
    _record(log, id="D001", cost_usd=None)

    line = (tmp_path / "delegation_log.jsonl").read_text().strip()
    rec = json.loads(line)
    assert rec["cost_usd"] is None


# ---------------------------------------------------------------------------
# DelegationLog creates parent directories on init
# ---------------------------------------------------------------------------


def test_delegation_log_creates_parent_dirs(tmp_path):
    """DelegationLog creates missing parent directories on init."""
    deep_path = tmp_path / "a" / "b" / "c" / "log.jsonl"
    assert not deep_path.parent.exists()
    DelegationLog(deep_path)
    assert deep_path.parent.exists()
