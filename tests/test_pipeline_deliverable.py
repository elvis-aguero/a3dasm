"""Switchable Pipeline-deliverable milestone (Spec C3).

The draft-pipeline gate seeds ONLY when runtime.pipeline_deliverable is on; off
= byte-identical to today (no pipeline milestone). The gate carries the
baseline-to-beat / block-swap-as-hypothesis framing and auto-satisfies when a
pipeline.py deliverable exists.
"""
from __future__ import annotations

from a3dasm._src import settings
from a3dasm._src.backends.base import Agent, Edge, Graph
from a3dasm._src.delegation_log import DelegationLog
from a3dasm._src.milestones import MilestoneLedger
from a3dasm._src.nodes import StrategizerNode


class _Stub:
    def __init__(self):
        self.closure_tools: dict = {}
        self.last_usage: dict = {}
        self.model = "m"

    def invoke(self, messages):
        return ""


def test_seed_includes_pipeline_only_when_requested(tmp_path):
    led = MilestoneLedger(tmp_path)
    led.seed_defaults(include_pipeline=False)
    assert "craft_pipeline" not in {m["key"] for m in led.list_all()}

    led2 = MilestoneLedger(tmp_path / "b")
    led2.seed_defaults(include_pipeline=True)
    keys = {m["key"] for m in led2.list_all()}
    assert "craft_pipeline" in keys
    dp = [m for m in led2.list_all() if m["key"] == "craft_pipeline"][0]
    assert "single deliverable" in dp["description"].lower()
    assert "engage deeply" in dp["description"].lower()


def test_pipeline_milestone_auto_satisfies_when_file_exists(tmp_path):
    from a3dasm._src.milestones import _pipeline_drafted

    class _N:
        _study_dir = tmp_path
    assert _pipeline_drafted(_N()) is False
    (tmp_path / "pipeline.ipynb").write_text("# candidate pipeline\n")
    assert _pipeline_drafted(_N()) is True


def _node(tmp_path):
    class A(Agent):
        role = "strategizer"
        tools = frozenset({"Done"})
        description = "strategizer"

    class B(Agent):
        role = "implementer"
        description = "impl"

    spec = Graph(
        nodes={"strategizer": A(), "implementer": B()},
        edges=(Edge("strategizer", "implementer"),), entry="strategizer")
    notes = tmp_path / "debug" / "strategizer_notes"
    notes.mkdir(parents=True)
    return StrategizerNode(
        _Stub(), name="strategizer", outgoing=["implementer"], spec=spec,
        worker_adapters={"implementer": _Stub()},
        study_dir=tmp_path, notes_dir=notes,
        delegation_log=DelegationLog(tmp_path / "debug" / "dlog.jsonl"))


def test_knob_on_seeds_pipeline_gate(tmp_path):
    settings.configure({"pipeline_deliverable": True})
    try:
        n = _node(tmp_path)
        keys = {m["key"] for m in n._milestones.list_all()}
        assert "craft_pipeline" in keys
    finally:
        settings.configure({})


def test_knob_off_is_byte_identical_no_pipeline_gate(tmp_path):
    settings.configure({"pipeline_deliverable": False})
    try:
        n = _node(tmp_path)
        keys = {m["key"] for m in n._milestones.list_all()}
        assert "craft_pipeline" not in keys
        # the always-on backlog gates are unaffected by the C3 knob
        assert {"assess_literature_need", "oracle_gold_state"} <= keys
    finally:
        settings.configure({})
