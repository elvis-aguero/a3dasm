import pytest
from a3dasm._src.graph_state import AgenticState, Task, Report, Delegation, StudyConfig
from a3dasm._src.backends.base import Agent, Edge, Graph


def test_agentic_state_is_dict_like():
    state = AgenticState(
        messages=[],
        study_dir="/tmp",
        done=False,
        last_report=None,
        total_delegations=0,
        budget_seconds=None,
    )
    assert state["done"] is False
    assert state["study_dir"] == "/tmp"
    assert state["total_delegations"] == 0


def test_graph_keeps_edge_validation():
    class A(Agent):
        description = "Test agent."

    with pytest.raises(ValueError, match="undeclared"):
        Graph(nodes={"a": A()}, edges=(Edge("a", "missing"),), entry="a")


def test_graph_outgoing():
    class A(Agent):
        description = "Test agent."

    g = Graph(nodes={"s": A(), "i": A()}, edges=(Edge("s", "i"),), entry="s")
    assert g.outgoing("s") == ["i"]
    assert g.outgoing("i") == []
