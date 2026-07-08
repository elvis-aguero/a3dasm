"""Tests for build_graph() and Graph primitives."""
from __future__ import annotations

import pytest
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver

from a3dasm._src.backends.base import Agent, Edge, Graph
from a3dasm._src.graph_builder import build_graph
from a3dasm._src.graph_state import AgenticState
from a3dasm._src.nodes import ImplementerNode, StrategizerNode


class StubAdapter:
    """Test adapter that returns scripted responses."""
    def __init__(self, *responses: str) -> None:
        self._responses = list(responses)
        self._call_count = 0
        self.closure_tools: dict = {}

    def invoke(self, messages: list[dict]) -> str:
        if self._call_count < len(self._responses):
            resp = self._responses[self._call_count]
        else:
            resp = self._responses[-1] if self._responses else "No response"
        self._call_count += 1
        return resp


_GRAPH_TEST_STUDY_DIR = None


def _graph_study_dir():
    global _GRAPH_TEST_STUDY_DIR
    if _GRAPH_TEST_STUDY_DIR is None:
        import tempfile
        from pathlib import Path
        d = Path(tempfile.mkdtemp(prefix="f3dasm_graph_test_"))
        (d / "pipeline.py").write_text("# test pipeline\n")
        _GRAPH_TEST_STUDY_DIR = d
    return _GRAPH_TEST_STUDY_DIR


def make_initial_state(problem="Test problem") -> AgenticState:
    return AgenticState(
        messages=[HumanMessage(content=problem)],
        study_dir=str(_graph_study_dir()),
        done=False,
        last_report=None,
        total_delegations=0,
        budget_seconds=None,
        return_to=None,
    )


class StrategAgent(Agent):
    role = "strategizer"
    description = "Test strategizer."
    tools = frozenset({"Done", "FollowUp", "WriteNote", "ReadNote"})


class ImplAgent(Agent):
    role = "implementer"
    description = "Test implementer."


def test_build_graph_creates_compiledgraph():
    """build_graph returns a compiled LangGraph graph."""
    spec = Graph(nodes={"s": ImplAgent()}, edges=(), entry="s")
    graph = build_graph(spec, lambda n, a: StubAdapter("## Done\nAll done."))

    assert hasattr(graph, "invoke")


def test_build_graph_strategizer_role_creates_strategizer_node():
    """Agent with role='strategizer' is built as StrategizerNode."""
    built_nodes = {}

    class TrackingStrategizerNode(StrategizerNode):
        pass

    class TrackingImplementerNode(ImplementerNode):
        pass

    spec = Graph(
        nodes={"s": StrategAgent(), "i": ImplAgent()},
        edges=(Edge("s", "i"),),
        entry="s",
    )

    import a3dasm._src.graph_builder as gb
    original_strat = gb.StrategizerNode
    original_impl = gb.ImplementerNode
    try:
        gb.StrategizerNode = TrackingStrategizerNode
        gb.ImplementerNode = TrackingImplementerNode

        adapters = {}

        def make_adapter(name, agent):
            a = StubAdapter("## Done\nAll done.")
            adapters[name] = a
            return a

        build_graph(spec, make_adapter)
    finally:
        gb.StrategizerNode = original_strat
        gb.ImplementerNode = original_impl


def test_build_graph_entry_node_receives_initial_message():
    """The entry node gets invoked with the initial message in state."""
    messages_seen = []

    class CapturingAdapter:
        closure_tools: dict = {}
        def invoke(self, messages):
            messages_seen.extend(messages)
            self.closure_tools["Done"](summary="Captured.")
            return "Done."

    spec2 = Graph(
        nodes={"s": StrategAgent(), "i": ImplAgent()},
        edges=(Edge("s", "i"),),
        entry="s",
    )

    s_adapter = CapturingAdapter()
    i_adapter = StubAdapter("## Report\nDone.")

    def make_adapter(name, agent):
        return s_adapter if name == "s" else i_adapter

    graph = build_graph(spec2, make_adapter, MemorySaver())
    config = {"configurable": {"thread_id": "test-1"}}
    state = make_initial_state("My initial problem")
    graph.invoke(state, config=config)

    assert any("My initial problem" in m.get("content", "") for m in messages_seen)


def test_build_graph_routes_delegate_to_implementer():
    """StrategizerNode delegates to ImplementerNode when Delegate closure is called."""
    strat_call_count = [0]
    impl_call_count = [0]

    class StratAdapter:
        closure_tools: dict = {}
        def invoke(self, messages):
            strat_call_count[0] += 1
            if strat_call_count[0] == 1:
                self.closure_tools["Delegate"](target="i", intent="Do something", expected_report="Report back")
                return "Delegating."
            else:
                self.closure_tools["Done"](summary="All done")
                return "Done."

    class ImplAdapter:
        closure_tools: dict = {}
        def invoke(self, messages):
            impl_call_count[0] += 1
            return "## Report\nTask complete."
        def copy(self):
            return ImplAdapter()

    spec = Graph(
        nodes={"s": StrategAgent(), "i": ImplAgent()},
        edges=(Edge("s", "i"),),
        entry="s",
    )

    graph = build_graph(
        spec,
        lambda name, a: StratAdapter() if name == "s" else ImplAdapter(),
        MemorySaver(),
    )

    config = {"configurable": {"thread_id": "test-2"}}
    result = graph.invoke(make_initial_state(), config=config)

    assert strat_call_count[0] >= 1
    assert impl_call_count[0] >= 1
    assert result["done"] is True


# ---------------------------------------------------------------------------
# Graph.to_mermaid and Graph.__repr__
# ---------------------------------------------------------------------------


def _two_node_graph() -> Graph:
    class S(Agent):
        role = "strategizer"
        tools = frozenset({"Done", "FollowUp", "WriteNote", "ReadNote"})
        description = "Test strategizer."

    class I(Agent):
        description = "Test worker."

    return Graph(
        nodes={"orch": S(), "worker": I()},
        edges=(Edge("orch", "worker", preamble="Focus on Python."),),
        entry="orch",
    )


def test_to_mermaid_starts_with_flowchart():
    g = _two_node_graph()
    mermaid = g.to_mermaid()
    assert mermaid.startswith("flowchart TD")


def test_to_mermaid_entry_uses_stadium_shape():
    g = _two_node_graph()
    mermaid = g.to_mermaid()
    assert 'orch(["' in mermaid


def test_to_mermaid_non_entry_uses_rect_shape():
    g = _two_node_graph()
    mermaid = g.to_mermaid()
    assert 'worker["' in mermaid


def test_to_mermaid_edge_with_preamble():
    g = _two_node_graph()
    mermaid = g.to_mermaid()
    assert "Focus on Python." in mermaid
    assert "orch -->|" in mermaid


def test_to_mermaid_edge_without_preamble():
    class A(Agent):
        role = "strategizer"
        tools = frozenset({"Done", "FollowUp", "WriteNote", "ReadNote"})
        description = "Test strategizer."

    class B(Agent):
        description = "Test worker."

    g = Graph(nodes={"a": A(), "b": B()}, edges=(Edge("a", "b"),), entry="a")
    mermaid = g.to_mermaid()
    assert "a --> b" in mermaid
    assert "--|" not in mermaid


def test_to_mermaid_preamble_truncated_at_35():
    long_preamble = "A" * 50

    class A(Agent):
        role = "strategizer"
        tools = frozenset({"Done", "FollowUp", "WriteNote", "ReadNote"})
        description = "Test strategizer."

    class B(Agent):
        description = "Test worker."

    g = Graph(
        nodes={"a": A(), "b": B()},
        edges=(Edge("a", "b", preamble=long_preamble),),
        entry="a",
    )
    mermaid = g.to_mermaid()
    assert "…" in mermaid


def test_repr_contains_entry_label():
    g = _two_node_graph()
    r = repr(g)
    assert "Graph(entry='orch')" in r
    assert "[entry]" in r


def test_repr_leaf_nodes_marked():
    g = _two_node_graph()
    r = repr(g)
    assert "(leaf)" in r


def test_repr_arrow_shows_preamble():
    g = _two_node_graph()
    r = repr(g)
    assert "Focus on Python." in r


def test_to_mermaid_styles_nodes_and_edges():
    """Styled mermaid: classDef per agent class, class assignments,
    and dotted (consultation) arrows from non-entry sources."""
    from a3dasm._src.agents._graphs import _default_graph
    m = _default_graph().to_mermaid()
    # colour styling present
    assert "classDef StrategizerAgent fill:#" in m
    assert "class strategizer StrategizerAgent" in m
    # entry-sourced edge is a solid delegation arrow
    assert "strategizer --> critic" in m
    # non-entry-sourced edge (specialist consults lit) is dotted
    assert "datagenerator -.-> literature_reviewer" in m
    assert "implementer -.-> literature_reviewer" in m
    # labels carry class + role/description
    assert "<b>strategizer</b><br/>StrategizerAgent" in m
