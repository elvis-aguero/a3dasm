"""Audit BF-0/BF-2: the persistent delegation log is the authoritative source
for delegation existence and terminal status; the in-memory ``_registry`` is
only a live-execution cache.

A stale or rebuilt-empty cache must never make a FINISHED delegation read as
"Working" (the run4 deadlock: Done()'s liveness gate refused forever, looping
back until _finish_attempts hit 3 and the run was killed UNGATED after it had
already found the optimum) or as "unknown" (the "Known IDs: []" symptom).
"""
from __future__ import annotations

from a3dasm._src.backends.base import Agent, Edge, Graph
from a3dasm._src.delegation_log import DelegationLog
from a3dasm._src.nodes import StrategizerNode


class _Stub:
    def __init__(self):
        self.closure_tools = {}
        self.route_watcher = None
        self.last_usage = {}

    def invoke(self, messages):
        return "ok"


def _spec():
    class S(Agent):
        role = "strategizer"
        description = "strategizer"

    class I(Agent):
        role = "implementer"
        description = "implementer"

    return Graph(
        nodes={"strategizer": S(), "implementer": I()},
        edges=(Edge("strategizer", "implementer"),),
        entry="strategizer",
    )


def _node(tmp_path):
    log = DelegationLog(tmp_path / "delegation_log.jsonl")
    node = StrategizerNode(
        _Stub(), name="strategizer", outgoing=["implementer"],
        spec=_spec(), worker_adapters={"implementer": _Stub()},
        delegation_log=log,
    )
    return node, log


def _record_started(log, did):
    log.record_started(
        id=did, from_node="strategizer", to_node="implementer",
        task="t", hypothesis_ids=[], started_at="t0",
    )


def _record_done(log, did, deliverable="## Report\nbest_f: -1.0"):
    _record_started(log, did)
    log.record(
        id=did, from_node="strategizer", to_node="implementer", task="t",
        deliverable=deliverable, hypothesis_ids=[], started_at="t0",
        completed_at="t1", status="DONE",
    )


def test_finished_delegation_not_pending_despite_stale_cache(tmp_path):
    """The run4 deadlock, pinned: the log shows D001 DONE, but the in-memory
    cache still says Working. Liveness must NOT report it pending."""
    node, log = _node(tmp_path)
    _record_done(log, "D001")
    node._registry = {"D001": {"status": "Working"}}  # stale cache
    assert node._pending_delegations() == []


def test_genuinely_running_delegation_is_pending(tmp_path):
    """A delegation with only a RUNNING record (no terminal) is still pending —
    the fix must not make close-while-running impossible to detect."""
    node, log = _node(tmp_path)
    _record_started(log, "D002")
    node._registry = {"D002": {"status": "Working"}}
    assert node._pending_delegations() == ["D002"]


def test_log_status_resolves_after_cache_rebuilt_empty(tmp_path):
    """The "Known IDs: []" symptom: registry rebuilt empty after a node
    reconstruction, but the log still has the completed delegation."""
    node, log = _node(tmp_path)
    _record_done(log, "D003")
    node._registry = {}
    status, deliverable = node._log_status("D003")
    assert status == "DONE"
    assert "best_f" in deliverable


def test_log_status_none_for_truly_unknown(tmp_path):
    node, _ = _node(tmp_path)
    assert node._log_status("D999") == (None, "")


def test_no_log_falls_back_to_cache_only(tmp_path):
    """With no persistent log, _pending_delegations uses the cache as-is (no
    crash); existence queries simply return None."""
    node = StrategizerNode(
        _Stub(), name="strategizer", outgoing=["implementer"],
        spec=_spec(), worker_adapters={"implementer": _Stub()},
        delegation_log=None,
    )
    node._registry = {"D001": {"status": "Working"}}
    assert node._pending_delegations() == ["D001"]
    assert node._log_status("D001") == (None, "")
