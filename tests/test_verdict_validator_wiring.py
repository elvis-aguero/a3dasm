"""#9 live verdict validator wired into the HypothesisUpdate closure.

Advise-with-teeth (Q1=B): the critique lands in the tool result and persists on
the verdict, but the update ALWAYS stands. Closing verdicts only. Uses a stub
critic adapter with a canned reply — no model.
"""
from __future__ import annotations

import json

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


class _CriticStub:
    """Critic adapter whose reply is canned; counts invocations; can raise."""

    def __init__(self, reply: str = "", raises: bool = False) -> None:
        self.reply = reply
        self.raises = raises
        self.calls = 0
        self.last_usage: dict = {}
        self.model = "critic-model"
        self.closure_tools: dict = {}

    def invoke(self, messages):
        self.calls += 1
        if self.raises:
            raise RuntimeError("validator model unavailable")
        return self.reply


def _node(tmp_path, critic_reply: str = "", critic_raises: bool = False):
    class S(Agent):
        role = "strategizer"
        tools = frozenset({"Done", "HypothesisPropose", "HypothesisUpdate", "HypothesisList", "HypothesisGet", "LinkFalsificationAttempt", "MilestoneList", "MilestonePropose", "MilestoneComplete", "MilestoneSkip", "RecallStore", "QueryStore"})
        description = "strategizer"

    class I(Agent):
        role = "implementer"
        description = "implementer"

    class C(Agent):
        role = "critic"
        description = "critic"

    nodes = {"strategizer": S(), "implementer": I(), "critic": C()}
    spec = Graph(
        nodes=nodes,
        edges=(Edge("strategizer", "implementer"), Edge("strategizer", "critic")),
        entry="strategizer",
    )
    notes = tmp_path / "debug" / "strategizer_notes"
    notes.mkdir(parents=True)
    dlog = DelegationLog(tmp_path / "debug" / "delegation_log.jsonl")
    critic = _CriticStub(critic_reply, raises=critic_raises)
    n = StrategizerNode(
        _Stub(), name="strategizer", outgoing=["implementer", "critic"], spec=spec,
        worker_adapters={"implementer": _Stub(), "critic": critic},
        notes_dir=notes, delegation_log=dlog,
    )
    n._current_notes_dir = notes  # so diagnostics.jsonl has a home
    for m in list(n._milestones.pending()):
        n._milestones.skip(m["id"], "test")
    return n, critic, tmp_path / "debug" / "diagnostics.jsonl"


def _propose(n):
    return n.adapter.closure_tools["HypothesisPropose"](
        "thin walls buckle first",
        "any sweep produces a feasible f >= 2.0",
        "dense sweep finds nothing below 1.5",
        0.5,
    )


def _record_done(n, did, falsify=False, h_ids=None):
    n._delegation_log.record(
        id=did, from_node="strategizer", to_node="implementer", task="probe",
        deliverable="## Report\n### Numbers\nbest_f: 1.47\n",
        hypothesis_ids=h_ids or [], started_at="2026-01-01T00:00:00+00:00",
        completed_at="2026-01-01T01:00:00+00:00", status="DONE",
        is_falsification_attempt=falsify,
    )


def _last_note(n, h):
    entry = n._ledger.get(h)
    return entry["status_log"][-1].get("validator_note")


# ── OK verdict: update stands, terse note, no flag ───────────────────────────

def test_ok_verdict_persists_note_and_does_not_flag(tmp_path):
    n, critic, diag = _node(tmp_path, critic_reply="SUBSTANCE: OK")
    h = _propose(n)
    _record_done(n, "D001", h_ids=[h])
    result = n.adapter.closure_tools["HypothesisUpdate"](
        h, "FALSIFIED", "test contradicted the prediction", 0.1,
        evidence={"delegation": "D001", "numbers": {"best_f": 1.47}},
    )
    assert result.startswith("Updated ")
    assert critic.calls == 1               # validator ran on the closing verdict
    assert "[VERDICT VALIDATOR]" not in result
    assert _last_note(n, h) == "validated: no charter concern"
    assert not diag.exists() or "VERDICT_SUBSTANCE_FLAG" not in diag.read_text()


# ── FLAG verdict: critique lands, but update STILL stands (non-blocking) ──────

def test_flag_lands_in_result_and_persists_but_update_stands(tmp_path):
    reply = "SUBSTANCE: FLAG\nCRITIQUE: §3 verdict does not follow — the test did not contradict the registered prediction."
    n, critic, diag = _node(tmp_path, critic_reply=reply)
    h = _propose(n)
    _record_done(n, "D001", h_ids=[h])
    result = n.adapter.closure_tools["HypothesisUpdate"](
        h, "FALSIFIED", "calling it falsified", 0.1,
        evidence={"delegation": "D001", "numbers": {"best_f": 1.47}},
    )
    # non-blocking: the update landed
    assert result.startswith("Updated ")
    # the critique reached the agent this turn AND persists on the verdict
    assert "[VERDICT VALIDATOR]" in result and "§3" in result
    assert "§3" in (_last_note(n, h) or "")
    # a diagnostics event was emitted
    assert diag.exists()
    events = [json.loads(line) for line in diag.read_text().splitlines() if line.strip()]
    assert any(e.get("error_type") == "VERDICT_SUBSTANCE_FLAG" and e.get("hypothesis") == h
               for e in events)


# ── the advisory call must carry a TIGHT budget, not a full agent turn's ─────

def test_validator_call_uses_a_bounded_budget(tmp_path):
    """Regression for run 20260627T211310: the validator inherited the main
    loop's 5×600s stream/retry budget and a hung CLI stream froze the whole run
    for ~89 min. The advisory call must pass a tight idle + retry_max=1 so it
    aborts ~2 min after a silent stream."""
    class _RecordingCritic(_CriticStub):
        def __init__(self):
            super().__init__(reply="SUBSTANCE: OK")
            self.kwargs = None

        def invoke(self, messages, **kwargs):
            self.calls += 1
            self.kwargs = kwargs
            return self.reply

    n, _, _ = _node(tmp_path)
    rec = _RecordingCritic()
    n._worker_adapters["critic"] = rec
    h = _propose(n)
    _record_done(n, "D001", h_ids=[h])
    n.adapter.closure_tools["HypothesisUpdate"](
        h, "FALSIFIED", "test contradicted the prediction", 0.1,
        evidence={"delegation": "D001", "numbers": {"best_f": 1.47}},
    )
    assert rec.calls == 1
    assert rec.kwargs == {"idle_timeout": 120.0, "retry_max": 1}


# ── OPEN is not a closing verdict → validator must not fire ──────────────────

def test_open_retraction_skips_the_validator(tmp_path):
    n, critic, _ = _node(tmp_path, critic_reply="SUBSTANCE: OK")
    h = _propose(n)
    _record_done(n, "D001", h_ids=[h])
    # Close INCONCLUSIVE first (a closing verdict → validator fires once); a
    # retraction to OPEN is permitted from INCONCLUSIVE (unlike FALSIFIED).
    n.adapter.closure_tools["HypothesisUpdate"](
        h, "INCONCLUSIVE", "confounded test", 0.5,
        evidence={"delegation": "D001", "numbers": {"best_f": 1.47}},
    )
    assert critic.calls == 1
    # retract to OPEN (a real "Updated …" with a non-closing status)
    res = n.adapter.closure_tools["HypothesisUpdate"](
        h, "OPEN", "retracting — no adequate attempt yet (Charter §2)", 0.5,
        evidence=None,
    )
    assert res.startswith("Updated "), res
    assert critic.calls == 1  # OPEN did NOT trigger the validator


# ── validator failure must never break the update (advisory) ─────────────────

def test_validator_exception_does_not_break_update(tmp_path):
    n, critic, _ = _node(tmp_path, critic_raises=True)
    h = _propose(n)
    _record_done(n, "D001", h_ids=[h])
    result = n.adapter.closure_tools["HypothesisUpdate"](
        h, "FALSIFIED", "x", 0.1,
        evidence={"delegation": "D001", "numbers": {"best_f": 1.47}},
    )
    assert result.startswith("Updated ")
    assert "[VERDICT VALIDATOR]" not in result  # no advisory on failure, but no crash


# ── repeat flags on the same hypothesis escalate (the "teeth") ───────────────

def test_repeat_flag_escalates(tmp_path):
    reply = "SUBSTANCE: FLAG\nCRITIQUE: §2 the test was not severe."
    n, critic, _ = _node(tmp_path, critic_reply=reply)
    h = _propose(n)
    _record_done(n, "D001", h_ids=[h])
    r1 = n.adapter.closure_tools["HypothesisUpdate"](
        h, "FALSIFIED", "first", 0.1,
        evidence={"delegation": "D001", "numbers": {"best_f": 1.47}},
    )
    assert "scrutinise" not in r1  # first flag: no escalation line yet
    r2 = n.adapter.closure_tools["HypothesisUpdate"](
        h, "FALSIFIED", "second, corrected numbers", 0.1,
        evidence={"delegation": "D001", "numbers": {"best_f": 1.50}},
    )
    assert "[VERDICT VALIDATOR]" in r2
    assert "scrutinise" in r2 and "2" in r2  # escalation on the 2nd flag


# ── kill switch: F3DASM_VERDICT_VALIDATOR=0 fully bypasses the validator ──────

def test_env_kill_switch_disables_validator(tmp_path, monkeypatch):
    monkeypatch.setenv("F3DASM_VERDICT_VALIDATOR", "0")
    n, critic, diag = _node(tmp_path, critic_reply="SUBSTANCE: FLAG\nCRITIQUE: x")
    h = _propose(n)
    _record_done(n, "D001", h_ids=[h])
    result = n.adapter.closure_tools["HypothesisUpdate"](
        h, "FALSIFIED", "x", 0.1,
        evidence={"delegation": "D001", "numbers": {"best_f": 1.47}},
    )
    # update lands exactly as pre-#9: no judge call, no note, no diagnostics
    assert result.startswith("Updated ")
    assert "[VERDICT VALIDATOR]" not in result
    assert critic.calls == 0
    assert _last_note(n, h) is None
    assert not diag.exists() or "VERDICT_SUBSTANCE_FLAG" not in diag.read_text()
