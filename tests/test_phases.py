"""Phase tags (Spec C1): the f3dasm process vocabulary on delegations."""
from __future__ import annotations

from a3dasm._src.backends.base import Agent, Edge, Graph
from a3dasm._src.delegation_log import DelegationLog
from a3dasm._src.nodes import StrategizerNode
from a3dasm._src.phases import Phase, resolve_phase


class _Stub:
    def __init__(self):
        self.closure_tools: dict = {}
        self.last_usage: dict = {}
        self.model = "m"

    def invoke(self, messages):
        return ""


def test_resolve_phase_exact_normalized_synonym_and_unknown():
    assert resolve_phase("doe") is Phase.DOE
    assert resolve_phase("DATA_GENERATION") is Phase.DATA_GENERATION
    assert resolve_phase("data-generation") is Phase.DATA_GENERATION
    assert resolve_phase("optimisation") is Phase.OPTIMIZATION  # spelling variant
    assert resolve_phase("machine learning") is Phase.ML
    assert resolve_phase(Phase.ML) is Phase.ML
    assert resolve_phase("nonsense") is None   # soft: unknown -> None
    assert resolve_phase(None) is None
    assert resolve_phase("") is None


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
    return StrategizerNode(
        _Stub(), name="strategizer", outgoing=["implementer"], spec=spec,
        worker_adapters={"implementer": _Stub()},
        delegation_log=DelegationLog(tmp_path / "dlog.jsonl"))


def test_delegate_stamps_resolved_phase_on_registry(tmp_path):
    n = _node(tmp_path)
    out = n.adapter.closure_tools["Delegate"](
        "implementer", "run a sweep", "a report", wait=True, phase="DoE")
    # wait=True returns the (stub) result; the registry carries the phase
    did = next(iter(n._registry))
    assert n._registry[did]["phase"] == "doe"   # canonical value, resolved
    assert out.lstrip().startswith(("Done", "Errored"))


def test_delegate_unknown_phase_is_soft_none(tmp_path):
    n = _node(tmp_path)
    n.adapter.closure_tools["Delegate"](
        "implementer", "x", "y", wait=True, phase="banana")
    did = next(iter(n._registry))
    assert n._registry[did]["phase"] is None   # unknown -> None, not an error


def test_delegation_log_record_carries_phase(tmp_path):
    dlog = DelegationLog(tmp_path / "dlog.jsonl")
    dlog.record(
        id="D001", from_node="s", to_node="i", task="t", deliverable="d",
        hypothesis_ids=[], started_at="t0", completed_at="t1", status="DONE",
        phase="optimization")
    assert dlog.query_all()[0]["phase"] == "optimization"
