"""Data-boundary enforcement in HypothesisUpdate (strategizer closure tool).

EVIDENCE_DELEGATION_EXISTS and SUPPORTED_WITHOUT_ATTACK moved from the
science monitor to the HypothesisUpdate tool itself so they fail fast at
the data boundary with a clear error return instead of a deferred monitor nag.
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
            "HypothesisGet", "LinkFalsificationAttempt", "RecallStore",
            "QueryStore"})
        description = "strategizer"

    class I(Agent):
        role = "implementer"
        description = "implementer"

    nodes = {"strategizer": S(), "implementer": I()}
    spec = Graph(
        nodes=nodes,
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
        "any sweep produces a feasible f ≥ 2.0",
        "dense sweep finds nothing below 1.5",
        0.5,
    )


def _record_done(n, did, falsify=False, h_ids=None):
    n._delegation_log.record(
        id=did, from_node="strategizer", to_node="implementer",
        task="t",
        deliverable="## Report\n### Numbers\nbest_f: 1.47\n",
        hypothesis_ids=h_ids or [],
        started_at="2026-01-01T00:00:00+00:00",
        completed_at="2026-01-01T01:00:00+00:00",
        status="DONE",
        is_falsification_attempt=falsify,
    )


# ---------------------------------------------------------------------------
# SUPPORTED_WITHOUT_ATTACK — hard block at update time
# ---------------------------------------------------------------------------

def test_supported_without_falsification_is_two_shot_confirm(tmp_path):
    """SUPPORTED without a completed falsification attempt is a TWO-SHOT CONFIRM
    (§4): the first call is refused with charter guidance; a re-call carrying a
    written justification in `comment` confirms (a verdict is reversible when
    justified — the validator + critic remain the downstream floor)."""
    n = _node(tmp_path)
    for m in list(n._milestones.pending()):
        n._milestones.skip(m["id"], "test")
    h = _propose(n)
    _record_done(n, "D001", falsify=False, h_ids=[h])
    update = n.adapter.closure_tools["HypothesisUpdate"]

    # First call: refused (CONFIRM), not silently accepted, not a hard ERROR.
    r1 = update(h, "SUPPORTED", "sweep passed", 0.85,
                evidence={"delegation": "D001", "numbers": {"best_f": 1.47}})
    assert r1.startswith("[CONFIRM]") and "falsification" in r1.lower()

    # Re-call WITHOUT a real justification: still refused (justification required).
    r2 = update(h, "SUPPORTED", "ok", 0.85,
                evidence={"delegation": "D001", "numbers": {"best_f": 1.47}})
    assert r2.startswith("[CONFIRM]")

    # Re-call WITH a written justification: confirms (proceeds).
    r3 = update(
        h, "SUPPORTED",
        "Accepting without a fresh attack: D001's dense in-range sweep already "
        "probed the refutation region and found nothing better; a further attack "
        "could still refute it via a denser grid.",
        0.85, evidence={"delegation": "D001", "numbers": {"best_f": 1.47}})
    assert not r3.startswith("[CONFIRM]") and not r3.startswith("ERROR:")


def test_supported_allowed_when_falsification_attempt_completed(tmp_path):
    """HypothesisUpdate must accept SUPPORTED when a completed falsification
    attempt targeting this hypothesis exists."""
    n = _node(tmp_path)
    for m in list(n._milestones.pending()):
        n._milestones.skip(m["id"], "test")
    h = _propose(n)
    _record_done(n, "D001", falsify=False, h_ids=[h])
    _record_done(n, "D002", falsify=True, h_ids=[h])

    result = n.adapter.closure_tools["HypothesisUpdate"](
        h, "SUPPORTED", "survived falsification attempt", 0.90,
        evidence={"delegation": "D002", "numbers": {"best_f": 1.47}},
    )
    assert not result.startswith("ERROR:"), (
        f"Expected success but got error: {result!r}")


def test_evidence_delegation_must_be_single_source(tmp_path):
    """Single-source attribution is enforced in CODE (not a prompt format-rule):
    a comma-joined or list delegation is rejected with the principle stated."""
    n = _node(tmp_path)
    for m in list(n._milestones.pending()):
        n._milestones.skip(m["id"], "test")
    h = _propose(n)
    _record_done(n, "D001", falsify=True, h_ids=[h])
    _record_done(n, "D002", falsify=True, h_ids=[h])
    out = n.adapter.closure_tools["HypothesisUpdate"](
        h, "SUPPORTED", "x", 0.8,
        evidence={"delegation": "D001,D002", "numbers": {"best_f": 1.47}})
    assert out.startswith("ERROR:") and "single source" in out.lower()


def test_supported_confirm_names_missing_criterion(tmp_path):
    """The CONFIRM message for SUPPORTED-without-attack cites the falsification
    criterion so the agent knows what to test."""
    n = _node(tmp_path)
    for m in list(n._milestones.pending()):
        n._milestones.skip(m["id"], "test")
    h = _propose(n)
    _record_done(n, "D001", falsify=False, h_ids=[h])

    result = n.adapter.closure_tools["HypothesisUpdate"](
        h, "SUPPORTED", "eager close", 0.8,
        evidence={"delegation": "D001", "numbers": {"best_f": 1.47}},
    )
    assert result.startswith("[CONFIRM]")
    # Should contain the criterion from the hypothesis
    assert "dense sweep" in result or "falsification_criterion" in result or "Falsification criterion" in result


# ---------------------------------------------------------------------------
# EVIDENCE_DELEGATION_EXISTS — hard block at update time
# ---------------------------------------------------------------------------

def test_phantom_delegation_blocked(tmp_path):
    """HypothesisUpdate must reject evidence citing a delegation that is
    not yet completed in the delegation log."""
    n = _node(tmp_path)
    for m in list(n._milestones.pending()):
        n._milestones.skip(m["id"], "test")
    h = _propose(n)
    # D999 was never recorded in the delegation log

    result = n.adapter.closure_tools["HypothesisUpdate"](
        h, "FALSIFIED", "phantom delegation", 0.1,
        evidence={"delegation": "D999", "numbers": {"best_f": 1.47}},
    )
    assert result.startswith("ERROR:"), (
        f"Expected error for phantom delegation D999, got: {result!r}")
    assert "D999" in result


def test_completed_delegation_allowed(tmp_path):
    """HypothesisUpdate must accept evidence citing a DONE delegation."""
    n = _node(tmp_path)
    for m in list(n._milestones.pending()):
        n._milestones.skip(m["id"], "test")
    h = _propose(n)
    _record_done(n, "D001", falsify=True, h_ids=[h])

    result = n.adapter.closure_tools["HypothesisUpdate"](
        h, "FALSIFIED", "attack succeeded", 0.05,
        evidence={"delegation": "D001", "numbers": {"best_f": 1.47}},
    )
    assert not result.startswith("ERROR:"), (
        f"Expected success for completed D001, got: {result!r}")


def test_d000_is_valid_evidence_anchor(tmp_path):
    """D000 (canonical ground-truth pool) must bypass the delegation-exists
    check — it is a special sentinel, not a user-submitted delegation."""
    n = _node(tmp_path)
    for m in list(n._milestones.pending()):
        n._milestones.skip(m["id"], "test")
    h = _propose(n)
    # No delegation log entries at all — D000 should still work
    _record_done(n, "D000-proxy", falsify=True, h_ids=[h])  # need a falsification attack

    result = n.adapter.closure_tools["HypothesisUpdate"](
        h, "SUPPORTED", "ground truth confirms", 0.95,
        evidence={"delegation": "D000", "numbers": {"max_y": 143.26}},
    )
    # D000 itself doesn't need to be in the delegation log
    assert not result.startswith("ERROR:") or "falsification" in result.lower(), (
        f"D000 evidence blocked unexpectedly by delegation-exists check: {result!r}")
