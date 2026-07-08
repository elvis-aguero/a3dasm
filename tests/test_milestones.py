"""Milestone backlog: target-keyed gating (blocks only the implementer)."""
from __future__ import annotations

from a3dasm._src.backends.base import Agent, Edge, Graph
from a3dasm._src.delegation_log import DelegationLog
from a3dasm._src.milestones import (
    MilestoneLedger,
    implementer_block,
    render_backlog,
)
from a3dasm._src.nodes import StrategizerNode


class _Stub:
    def __init__(self):
        self.closure_tools: dict = {}
        self.last_usage: dict = {}
        self.model = "m"

    def invoke(self, messages):
        return ""


# --------------------------------------------------------------------------
# Ledger unit behaviour
# --------------------------------------------------------------------------

def test_seed_defaults_without_pipeline(tmp_path):
    led = MilestoneLedger(tmp_path)
    led.seed_defaults(include_pipeline=False)
    keys = {m["key"] for m in led.list_all()}
    assert keys == {"assess_literature_need", "oracle_gold_state"}
    assert all(m["status"] == "PENDING" for m in led.list_all())


def test_seed_defaults_with_pipeline_puts_it_first(tmp_path):
    led = MilestoneLedger(tmp_path)
    led.seed_defaults(include_pipeline=True)
    items = led.list_all()
    assert items[0]["key"] == "craft_pipeline"  # M001, read first
    assert {m["key"] for m in items} == {
        "craft_pipeline", "assess_literature_need", "oracle_gold_state"}


def test_seed_is_idempotent(tmp_path):
    led = MilestoneLedger(tmp_path)
    led.seed_defaults(include_pipeline=True)
    led.seed_defaults(include_pipeline=True)
    assert len(led.list_all()) == 3


def test_assess_literature_is_manual_no_predicate(tmp_path):
    led = MilestoneLedger(tmp_path)
    led.seed_defaults(include_pipeline=True)
    assess = [m for m in led.list_all() if m["key"] == "assess_literature_need"][0]
    assert assess["manual"] is True
    craft = [m for m in led.list_all() if m["key"] == "craft_pipeline"][0]
    assert craft["manual"] is False  # auto via pipeline.py


def test_propose_complete_skip(tmp_path):
    led = MilestoneLedger(tmp_path)
    mid = led.propose("my own step")
    assert led.get(mid)["source"] == "agent"
    led.complete(mid, "did it")
    assert led.get(mid)["status"] == "DONE"
    mid2 = led.propose("optional")
    led.skip(mid2, "n/a")
    assert led.get(mid2)["status"] == "SKIPPED"


# --------------------------------------------------------------------------
# Auto-satisfy + implementer block (against a real node)
# --------------------------------------------------------------------------

def _node(tmp_path):
    class A(Agent):
        role = "strategizer"
        # Declaration-driven exposure: declare the capability tools this stub
        # exercises (hypothesis + milestone closures are no longer injected).
        tools = frozenset({
            "Done", "HypothesisPropose", "HypothesisUpdate", "HypothesisList",
            "HypothesisGet", "LinkFalsificationAttempt", "MilestoneList",
            "MilestonePropose", "MilestoneComplete", "MilestoneSkip",
            "RecallStore", "QueryStore"})
        description = "strategizer"

    class Lit(Agent):
        role = "literature_reviewer"
        description = "lit"

    class B(Agent):
        role = "implementer"
        description = "impl"

    spec = Graph(
        nodes={"strategizer": A(), "literature_reviewer": Lit(),
               "implementer": B()},
        edges=(Edge("strategizer", "literature_reviewer"),
               Edge("strategizer", "implementer")), entry="strategizer")
    notes = tmp_path / "debug" / "strategizer_notes"
    notes.mkdir(parents=True)
    return StrategizerNode(
        _Stub(), name="strategizer",
        outgoing=["literature_reviewer", "implementer"], spec=spec,
        worker_adapters={"literature_reviewer": _Stub(), "implementer": _Stub()},
        study_dir=tmp_path, notes_dir=notes,
        delegation_log=DelegationLog(tmp_path / "debug" / "dlog.jsonl"))


def test_implementer_block_lists_all_pending(tmp_path):
    n = _node(tmp_path)
    pend = implementer_block(n._milestones, n)
    assert {m["key"] for m in pend} == {
        "craft_pipeline", "assess_literature_need", "oracle_gold_state"}


def test_craft_pipeline_auto_satisfies_when_pipeline_exists(tmp_path):
    n = _node(tmp_path)
    (tmp_path / "pipeline.ipynb").write_text("# candidate\n")
    pend_keys = {m["key"] for m in implementer_block(n._milestones, n)}
    assert "craft_pipeline" not in pend_keys      # auto-satisfied
    assert "assess_literature_need" in pend_keys   # manual, still pending


def test_milestone_propose_tool_does_not_crash(tmp_path):
    """Regression (run 20260629T191754): the MilestonePropose tool WRAPPER
    crashed on every call — it forwarded (description, phase, gate) to
    MilestoneLedger.propose(), which takes only `description`, raising
    TypeError: propose() takes 2 positional arguments but 4 were given. The
    feature was dead. Exercise the wrapper end-to-end (not the ledger method)
    and assert it returns a milestone id, and that the proposed milestone is
    pending (so it genuinely joins the implementer-gating backlog)."""
    n = _node(tmp_path)
    mid = n.adapter.closure_tools["MilestonePropose"]("verify both stage gates")
    assert mid.startswith("M"), mid
    pending_ids = {m["id"] for m in n._milestones.pending()}
    assert mid in pending_ids


def test_delegate_to_implementer_milestone_is_two_shot_nudge(tmp_path):
    """The milestone backlog NUDGES the implementer delegation once (not a hard
    block): the agent re-delegates to confirm and it fires. MilestoneComplete/
    Skip remain the clean path; literature_reviewer is never gated."""
    n = _node(tmp_path)
    hid = n.adapter.closure_tools["HypothesisPropose"](
        "stmt", "crit", "pred", 0.5)
    before = len(n._registry)
    # First attempt → confirm nudge, NOT a hard block, nothing fired.
    out = n.adapter.closure_tools["Delegate"](
        "implementer", "run experiments", "report", hypothesis_ids=[hid],
        wait=True)
    assert out.startswith("[CONFIRM]") and "backlog" in out
    assert len(n._registry) == before  # nothing fired yet
    # delegating to the literature_reviewer is NEVER gated (satisfies a gate)
    out_lit = n.adapter.closure_tools["Delegate"](
        "literature_reviewer", "survey", "report", hypothesis_ids=[hid],
        wait=True)
    assert not out_lit.startswith("[CONFIRM]")
    # Re-delegate to the implementer → proceeds past the soft gate (fires).
    out2 = n.adapter.closure_tools["Delegate"](
        "implementer", "run experiments", "report", hypothesis_ids=[hid],
        wait=True)
    assert not out2.startswith("[CONFIRM]")
    assert len(n._registry) > before  # it fired


def test_milestone_nudge_recurs_per_namespace(tmp_path):
    """The nudge fires once PER namespace — opening a new design re-prompts the
    setup milestones for that design rather than silently inheriting the ack."""
    n = _node(tmp_path)
    hid = n.adapter.closure_tools["HypothesisPropose"](
        "stmt", "crit", "pred", 0.5)
    # Confirm the default namespace.
    n.adapter.closure_tools["Delegate"](
        "implementer", "run", "report", hypothesis_ids=[hid], wait=True)
    n.adapter.closure_tools["Delegate"](
        "implementer", "run", "report", hypothesis_ids=[hid], wait=True)
    # A NEW namespace nudges again (its own setup).
    out_ns = n.adapter.closure_tools["Delegate"](
        "implementer", "run", "report", hypothesis_ids=[hid],
        namespace="elliptical_rings", wait=True)
    assert out_ns.startswith("[CONFIRM]")


def test_milestone_complete_requires_a_brief(tmp_path):
    n = _node(tmp_path)
    mid = n._milestones.propose("my step")
    assert "ERROR" in n.adapter.closure_tools["MilestoneComplete"](mid, "")
    ok = n.adapter.closure_tools["MilestoneComplete"](mid, "done via D2")
    assert "ERROR" not in ok and n._milestones.get(mid)["status"] == "DONE"


def test_done_blocked_until_backlog_resolved(tmp_path):
    n = _node(tmp_path)
    out = n.adapter.closure_tools["Done"](summary="done")
    assert "Cannot close yet" in out
    for m in n._milestones.list_all():
        n.adapter.closure_tools["MilestoneSkip"](m["id"], "n/a")
    out2 = n.adapter.closure_tools["Done"](summary="done")
    assert "Cannot close yet" not in out2


def test_render_backlog_announcement(tmp_path):
    n = _node(tmp_path)
    bl = render_backlog(n._milestones)
    assert "<process_backlog>" in bl
    assert "f3dasm implementer" in bl
    assert "MilestoneSkip" in bl
    # the three milestones are listed
    for kw in ("Pipeline", "literature", "oracle"):
        assert kw.lower() in bl.lower()


def test_milestone_closures_registered(tmp_path):
    n = _node(tmp_path)
    for tool in ("MilestoneList", "MilestonePropose", "MilestoneComplete",
                 "MilestoneSkip"):
        assert tool in n.adapter.closure_tools
