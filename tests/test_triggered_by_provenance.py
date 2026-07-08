"""Provenance of the `triggered_by` field stamped by HypothesisUpdate.

Regression for run 20260706T204732: a closing verdict recorded
`triggered_by` as the most-recently-completed delegation rather than the
delegation the agent explicitly CITED as evidence, so the audit trail
mislabelled its own source (H5 FALSIFIED cited D011 but recorded
triggered_by=D013, an unrelated later sweep). The verdict must be
attributable to the source it rests on.
"""
from __future__ import annotations

from a3dasm._src.backends.base import Agent, Edge, Graph
from a3dasm._src.delegation_log import DelegationLog
from a3dasm._src.nodes import StrategizerNode


class _Stub:
    def __init__(self) -> None:
        self.closure_tools: dict = {}
        self.last_usage: dict = {}
        self.model = "m"

    def invoke(self, messages):
        return ""


def _node(tmp_path):
    class S(Agent):
        role = "strategizer"
        tools = frozenset({
            "Done", "HypothesisPropose", "HypothesisUpdate", "HypothesisList",
            "HypothesisGet", "LinkFalsificationAttempt"})
        description = "strategizer"

    class I(Agent):
        role = "implementer"
        description = "implementer"

    spec = Graph(
        nodes={"strategizer": S(), "implementer": I()},
        edges=(Edge("strategizer", "implementer"),),
        entry="strategizer",
    )
    notes = tmp_path / "debug" / "strategizer_notes"
    notes.mkdir(parents=True)
    dlog = DelegationLog(tmp_path / "debug" / "delegation_log.jsonl")
    return StrategizerNode(
        _Stub(), name="strategizer", outgoing=["implementer"], spec=spec,
        worker_adapters={"implementer": _Stub()},
        notes_dir=notes, delegation_log=dlog,
    )


def _propose(n):
    return n.adapter.closure_tools["HypothesisPropose"](
        "thin walls buckle first",
        "any sweep produces a feasible f >= 2.0",
        "dense sweep finds nothing below 1.5",
        0.5,
    )


def _record_done(n, did, falsify=False, h_ids=None):
    n._delegation_log.record(
        id=did, from_node="strategizer", to_node="implementer", task="t",
        deliverable="## Report\n### Numbers\nbest_f: 1.47\n",
        hypothesis_ids=h_ids or [],
        started_at="2026-01-01T00:00:00+00:00",
        completed_at="2026-01-01T01:00:00+00:00",
        status="DONE", is_falsification_attempt=falsify,
    )


def test_triggered_by_is_the_cited_evidence_delegation(tmp_path):
    n = _node(tmp_path)
    for m in list(n._milestones.pending()):
        n._milestones.skip(m["id"], "test")
    h = _propose(n)
    _record_done(n, "D011", falsify=True, h_ids=[h])   # the CITED evidence
    _record_done(n, "D013", falsify=True, h_ids=[h])   # a LATER, unrelated one

    out = n.adapter.closure_tools["HypothesisUpdate"](
        h, "FALSIFIED", "attack succeeded on the registered prediction", 0.05,
        evidence={"delegation": "D011", "numbers": {"best_f": 1.47}},
    )
    assert not out.startswith("ERROR:"), out

    last = n._ledger.get(h)["status_log"][-1]
    assert last["triggered_by"] == "D011", (
        "triggered_by must be the cited evidence delegation D011, not the "
        f"most-recently-completed D013; got {last['triggered_by']!r}")
