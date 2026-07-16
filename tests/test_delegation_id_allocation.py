"""A milestone-blocked delegation must NOT burn a delegation ID.

next_id() advances a monotonic counter on every call; it used to be called at
the top of Delegate(), before the milestone gate, so a blocked implementer
attempt consumed an ID that never reached the log — leaving a permanent gap in
the sequence (the always-present early MILESTONE_BLOCK reliably ate "D002").
The fix moves allocation to AFTER the gate; this pins it.
"""
from __future__ import annotations

from a3dasm._src.backends.base import Agent, Edge, Graph
from a3dasm._src.delegation_log import DelegationLog
from a3dasm._src.nodes import StrategizerNode


class _Stub:
    def __init__(self):
        self.closure_tools: dict = {}
        self.last_usage: dict = {}
        self.model = "m"

    def invoke(self, messages):
        return ""


def _node(tmp_path):
    class A(Agent):
        role = "strategizer"
        tools = frozenset({"Done"})
        description = "strategizer"

    class B(Agent):
        role = "implementer"
        description = "implementer"

    spec = Graph(
        nodes={"strategizer": A(), "implementer": B()},
        edges=(Edge("strategizer", "implementer"),), entry="strategizer")
    notes = tmp_path / "debug" / "strategizer_notes"
    notes.mkdir(parents=True)
    return StrategizerNode(
        _Stub(), name="strategizer", outgoing=["implementer"], spec=spec,
        worker_adapters={"implementer": _Stub()}, notes_dir=notes,
        delegation_log=DelegationLog(tmp_path / "dlog.jsonl"))


def test_milestone_blocked_delegation_does_not_consume_an_id(tmp_path):
    n = _node(tmp_path)
    # Seed a pending milestone → the implementer is gated until it is resolved.
    n._milestones.propose("resolve the process backlog first")
    assert n._milestones.pending(), "precondition: a milestone should be pending"

    # First implementer attempt is NUDGED (two-shot confirm) — must not record
    # or allocate an ID (the nudge returns before ID allocation, as the hard
    # block used to).
    blocked = n.adapter.closure_tools["Delegate"](
        "implementer", "run a sweep", "a report", wait=True)
    assert blocked.lstrip().startswith("[CONFIRM]"), blocked

    # The shared ID counter must NOT have advanced: the next allocation is still
    # the first ID. (With the bug — allocation before the gate — the blocked
    # attempt would have consumed D001, so this would return "D002".)
    assert n._delegation_log.next_id() == "D001", "blocked attempt burned an ID"


def test_delegation_log_resumes_past_existing_ids(tmp_path):
    """RESUME-SAFE (#3): a fresh DelegationLog over an existing log must CONTINUE
    the D### sequence, not restart at D001 — which collided with a crashed turn's
    D001/D003/… workspaces in run 20260715T191329."""
    p = tmp_path / "debug" / "delegation_log.jsonl"
    log = DelegationLog(p)
    for _ in range(3):
        did = log.next_id()                 # D001 D002 D003
        log.record_started(
            id=did, from_node="strategizer", to_node="implementer",
            task="t", hypothesis_ids=[],
            started_at="2026-01-01T00:00:00+00:00")

    resumed = DelegationLog(p)              # simulate a resumed session
    assert resumed.next_id() == "D004", "resume restarted the id sequence"


def test_delegation_log_fresh_run_starts_at_d001(tmp_path):
    """A fresh run (no/empty log) is unchanged: first id is D001."""
    log = DelegationLog(tmp_path / "debug" / "delegation_log.jsonl")
    assert log.next_id() == "D001"
