"""Falsification read-time ritual (friction: late SUPPORTED_WITHOUT_ATTACK spin).

When the strategizer reads a finished delegation's report, a checkpoint forces
it to classify whether that delegation was a falsification ATTEMPT of a
registered hypothesis — and if so, link it and record the verdict THEN, judged
against the hypothesis's pre-registered (immutable) prediction. The ritual only
LINKS; it never writes a verdict (no rubber-stamp) and never authors a criterion
(no §4 goalpost-move). Exploration delegations pass cleanly with no hypothesis.
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


def _node(tmp_path, with_critic=False):
    class A(Agent):
        role = "strategizer"
        # GetStatus is opt-in (plug-and-play) post-audit; opt in so the ritual
        # tests can drive a delegation to completion via polling.
        tools = frozenset({
            "Done", "GetStatus", "HypothesisPropose", "HypothesisUpdate",
            "HypothesisList", "HypothesisGet", "LinkFalsificationAttempt",
            "MilestoneList", "MilestonePropose", "MilestoneComplete",
            "MilestoneSkip", "RecallStore", "QueryStore"})
        description = "strategizer"

    class B(Agent):
        role = "implementer"
        description = "implementer"

    nodes = {"strategizer": A(), "implementer": B()}
    edges = [Edge("strategizer", "implementer")]
    spec = Graph(nodes=nodes, edges=tuple(edges), entry="strategizer")
    notes = tmp_path / "debug" / "strategizer_notes"
    notes.mkdir(parents=True)
    dlog = DelegationLog(tmp_path / "debug" / "delegation_log.jsonl")
    n = StrategizerNode(
        _Stub(), name="strategizer", outgoing=["implementer"], spec=spec,
        worker_adapters={"implementer": _Stub()},
        notes_dir=notes, delegation_log=dlog,
    )
    return n


def _propose(n, statement="thin walls buckle first", pred="best feasible f < 2.0"):
    return n.adapter.closure_tools["HypothesisPropose"](
        statement, "an adequate sweep finds a feasible f >= 2.0", pred, 0.5)


def _done_record(n, did="D001"):
    n._delegation_log.record(
        id=did, from_node="strategizer", to_node="implementer",
        task="t", deliverable="## Report\n### Numbers\nbest f: 1.5\n",
        hypothesis_ids=[], started_at="2026-01-01T00:00:00+00:00",
        completed_at="2026-01-01T01:00:00+00:00", status="DONE")


def test_hypothesis_link_deferred_while_process_backlog_open(tmp_path):
    """#3: the 'every delegation must link a hypothesis' error is DEFERRED while
    the process backlog is still open (setup phase — no hypotheses exist yet, so
    there's nothing to link). Once the backlog is cleared, the link is required.
    """
    n = _node(tmp_path)
    assert [m["id"] for m in n._milestones.pending()]  # backlog open at start
    deleg = n.adapter.closure_tools["Delegate"]

    # Backlog OPEN: an empty-hypothesis delegation is NOT bounced on the link
    # rule (it defers, then hits the separate milestone gate).
    out = deleg(target="implementer", intent="wrap the oracle",
                expected_report="validation", hypothesis_ids=[])
    assert "must not be empty" not in out

    # Backlog CLEARED: the link is required again.
    for m in n._milestones.list_all():
        n._milestones.skip(m["id"], "n/a")
    out2 = deleg(target="implementer", intent="run experiment",
                 expected_report="results", hypothesis_ids=[])
    assert "must not be empty" in out2


# --------------------------------------------------------------------------
# 1. DelegationLog.mark_attempt — retroactive post-hoc link
# --------------------------------------------------------------------------

def test_mark_attempt_flags_and_stamps_posthoc(tmp_path):
    dlog = DelegationLog(tmp_path / "dlog.jsonl")
    dlog.record(
        id="D001", from_node="s", to_node="i", task="t", deliverable="d",
        hypothesis_ids=[], started_at="t0", completed_at="t1", status="DONE")
    assert dlog.mark_attempt("D001", "H1") is True
    rec = dlog.query_all()[0]
    assert rec["is_falsification_attempt"] is True
    assert "H1" in rec["hypothesis_ids"]
    assert rec["attempt_linked_post_hoc"] is True


def test_mark_attempt_unknown_id_returns_false(tmp_path):
    dlog = DelegationLog(tmp_path / "dlog.jsonl")
    assert dlog.mark_attempt("D999", "H1") is False


# --------------------------------------------------------------------------
# 2. LinkFalsificationAttempt closure — links only, never a verdict
# --------------------------------------------------------------------------

def test_link_falsification_links_registry_and_log(tmp_path):
    n = _node(tmp_path)
    hid = _propose(n)
    n._registry["D001"] = {
        "status": "Done", "result": "r", "is_falsification_attempt": False,
        "hypothesis_ids": [], "reconciled": False}
    _done_record(n)
    out = n.adapter.closure_tools["LinkFalsificationAttempt"]("D001", hid)
    assert "Linked" in out and hid in out
    assert "best feasible f < 2.0" in out  # quotes the pre-registered prediction
    assert "does NOT" in out.lower() or "not" in out.lower()
    assert n._registry["D001"]["is_falsification_attempt"] is True
    assert hid in n._registry["D001"]["hypothesis_ids"]
    assert n._registry["D001"]["reconciled"] is True
    rec = n._delegation_log.query_all()[0]
    assert rec["is_falsification_attempt"] is True
    assert rec["attempt_linked_post_hoc"] is True


def test_link_does_not_record_a_verdict(tmp_path):
    """No rubber-stamp: linking leaves the hypothesis status untouched."""
    n = _node(tmp_path)
    hid = _propose(n)
    n._registry["D001"] = {
        "status": "Done", "result": "r", "is_falsification_attempt": False,
        "hypothesis_ids": [], "reconciled": False}
    _done_record(n)
    n.adapter.closure_tools["LinkFalsificationAttempt"]("D001", hid)
    statuses = [h["current_status"] for h in n._ledger.list_all()
                if h["id"] == hid]
    assert statuses == ["OPEN"]  # link did not close it


def test_link_errors_on_unknown_delegation_or_hypothesis(tmp_path):
    n = _node(tmp_path)
    hid = _propose(n)
    assert "unknown delegation" in n.adapter.closure_tools[
        "LinkFalsificationAttempt"]("D999", hid).lower()
    n._registry["D001"] = {"status": "Done", "hypothesis_ids": [],
                           "is_falsification_attempt": False}
    assert "not found" in n.adapter.closure_tools[
        "LinkFalsificationAttempt"]("D001", "H99").lower()


# --------------------------------------------------------------------------
# 3. Read-time ritual on the Done report
# --------------------------------------------------------------------------

def test_ritual_fires_once_then_reconciled(tmp_path):
    n = _node(tmp_path)
    _propose(n)
    n._registry["D001"] = {
        "status": "Done", "result": "the report", "start_time": 0.0,
        "is_falsification_attempt": False, "hypothesis_ids": [],
        "reconciled": False, "getstatus_count": 0}
    out1 = n.adapter.closure_tools["GetStatus"]("D001")
    assert out1.lstrip().startswith("Done")
    assert "FALSIFICATION CHECKPOINT" in out1
    assert n._registry["D001"]["reconciled"] is True
    # second read: no checkpoint (fire-once)
    out2 = n.adapter.closure_tools["GetStatus"]("D001")
    assert "FALSIFICATION CHECKPOINT" not in out2


def test_ritual_surfaces_pre_registered_prediction(tmp_path):
    n = _node(tmp_path)
    _propose(n, pred="resonance >= 8000 Hz")
    n._registry["D001"] = {
        "status": "Done", "result": "r", "start_time": 0.0,
        "is_falsification_attempt": False, "hypothesis_ids": [],
        "reconciled": False, "getstatus_count": 0}
    out = n.adapter.closure_tools["GetStatus"]("D001")
    assert "resonance >= 8000 Hz" in out  # the immutable, pre-registered text
    # the ritual must point at LinkFalsificationAttempt, never invite a new
    # criterion after seeing the result
    assert "LinkFalsificationAttempt" in out
    assert "do not retrofit" in out.lower()


def test_predeclared_attempt_collapses_to_verdict_reminder(tmp_path):
    n = _node(tmp_path)
    hid = _propose(n)
    n._registry["D001"] = {
        "status": "Done", "result": "r", "start_time": 0.0,
        "is_falsification_attempt": True, "hypothesis_ids": [hid],
        "reconciled": False, "getstatus_count": 0}
    out = n.adapter.closure_tools["GetStatus"]("D001")
    assert "FALSIFICATION CHECKPOINT" in out
    assert "declared a falsification attempt" in out.lower()
    assert hid in out and "HypothesisUpdate" in out


def test_exploration_with_no_hypotheses_passes_clean(tmp_path):
    n = _node(tmp_path)  # no hypotheses proposed
    n._registry["D001"] = {
        "status": "Done", "result": "r", "start_time": 0.0,
        "is_falsification_attempt": False, "hypothesis_ids": [],
        "reconciled": False, "getstatus_count": 0}
    out = n.adapter.closure_tools["GetStatus"]("D001")
    assert out.lstrip().startswith("Done")
    assert "CHECKPOINT" not in out


# --------------------------------------------------------------------------
# 4. Done()-gate lists dangling falsification attempts (WARNING, not ERROR)
# --------------------------------------------------------------------------

def test_done_gate_warns_on_dangling_falsification(tmp_path):
    n = _node(tmp_path)
    hid = _propose(n)  # stays OPEN
    n._registry["D001"] = {
        "status": "Done", "result": "r", "is_falsification_attempt": True,
        "hypothesis_ids": [hid], "reconciled": True}
    # clear the (orthogonal) milestone close-gate so we reach the falsification
    # warn this test is about
    if n._milestones is not None:
        for m in n._milestones.list_all():
            n._milestones.skip(m["id"], "n/a")
    out = n.adapter.closure_tools["Done"](summary="done")
    assert not out.lstrip().startswith("ERROR:")
    assert "D001" in out
    assert "OPEN" in out


# --------------------------------------------------------------------------
# 5. science_monitor self-heals once a post-hoc link is made
# --------------------------------------------------------------------------

def test_supported_confirm_without_attack_then_allowed_after_link(tmp_path):
    """SUPPORTED_WITHOUT_ATTACK is a TWO-SHOT CONFIRM at the HypothesisUpdate
    data boundary (§4), not a hard block. Verify:
    1. HypothesisUpdate returns [CONFIRM] for SUPPORTED with no falsification
       attack (the clean path is to run/link an attempt).
    2. After marking D001 as a falsification attempt, the update succeeds
       directly (the confirm is skipped — there IS an attempt now).
    """
    n = _node(tmp_path)
    hid = _propose(n)
    _done_record(n)
    # 1. No falsification attempt yet → two-shot confirm
    out = n.adapter.closure_tools["HypothesisUpdate"](
        hid, "SUPPORTED", "survived", 0.8,
        {"delegation": "D001", "numbers": {"best f": 1.5}})
    assert out.startswith("[CONFIRM]"), (
        f"Expected a two-shot confirm before a falsification attempt, got: {out!r}")
    # 2. Mark D001 as a falsification attempt → now allowed
    n._delegation_log.mark_attempt("D001", hid)
    out2 = n.adapter.closure_tools["HypothesisUpdate"](
        hid, "SUPPORTED", "survived falsification", 0.8,
        {"delegation": "D001", "numbers": {"best f": 1.5}})
    assert not out2.startswith("ERROR:"), (
        f"Expected success after falsification attempt, got: {out2!r}")


def test_hypothesis_list_tolerates_stray_kwarg(tmp_path):
    """ROOT 4: HypothesisList takes no real args, but agents confuse it with
    Delegate/AskForFeedback and pass hypothesis_ids. That stray kwarg must be
    absorbed (list ALL hypotheses) — never crash the turn with a TypeError."""
    n = _node(tmp_path)
    hid = _propose(n)
    hlist = n.adapter.closure_tools["HypothesisList"]
    out = hlist(hypothesis_ids=[hid])          # the exact observed fumble
    assert not out.startswith("ERROR")
    assert hid in out                          # listed despite the bad kwarg
    assert hlist() == out                      # arg is ignored: identical to no-arg
