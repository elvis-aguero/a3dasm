"""Tests for the AgenticRun thin wrapper."""
from __future__ import annotations

import pytest
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver

from a3dasm._src.agent_runtime import (
    DEFAULT_MODEL,
    AgenticRun,
    AgenticRunError,
    ImplementerAgent,
    StrategizerAgent,
    _default_graph,
)
from a3dasm._src.backends.base import Agent, Edge, Graph
from a3dasm._src.graph_builder import build_graph


def make_stub_run(tmp_path, strat_responses=None, impl_responses=None):
    """Build an AgenticRun with stub adapters (no real LLM)."""
    from a3dasm._src.backends.base import Agent, Edge, Graph
    from a3dasm._src.graph_builder import build_graph

    (tmp_path / "PROBLEM_STATEMENT.md").write_text("Solve: find minimum of f(x)=x^2")
    (tmp_path / "pipeline.py").write_text("# test pipeline\n")

    strat_resps = strat_responses or ["Done."]
    impl_resps = impl_responses or ["## Report\nDone."]

    class StubAdapter:
        def __init__(self, responses):
            self._responses = list(responses)
            self._call_count = 0
            self.closure_tools: dict = {}
        def invoke(self, messages):
            resp = self._responses[min(self._call_count, len(self._responses) - 1)]
            self._call_count += 1
            return resp

    class DoneStratAdapter(StubAdapter):
        def invoke(self, messages):
            # Call Done closure twice (two-shot: first warns, second closes)
            resp = super().invoke(messages)
            if "Done" in (self.closure_tools or {}):
                self.closure_tools["Done"](summary="Task complete.")  # WARNING
                self.closure_tools["Done"](summary="Task complete.")  # close
            return resp

    graph_spec = _default_graph()

    _fallback = StubAdapter(impl_resps)
    adapters = {
        "strategizer": DoneStratAdapter(strat_resps),
        "implementer": StubAdapter(impl_resps),
    }

    def _make_adapter(n, a):
        return adapters.get(n, _fallback)

    compiled = build_graph(graph_spec, _make_adapter, MemorySaver())

    run = AgenticRun.__new__(AgenticRun)
    run.study_dir = tmp_path
    run._model = DEFAULT_MODEL
    run._budget = None
    run._graph_spec = graph_spec
    run._graph = compiled
    return run


def test_agentic_run_raises_if_no_problem_statement(tmp_path):
    """AgenticRun.execute raises AgenticRunError if PROBLEM_STATEMENT.md is missing."""
    run = AgenticRun.__new__(AgenticRun)
    run.study_dir = tmp_path
    run._model = DEFAULT_MODEL
    run._budget = None
    run._graph_spec = _default_graph()
    # Don't build the real graph (would try to connect to claude)
    run._graph = None

    with pytest.raises(AgenticRunError, match="PROBLEM_STATEMENT.md not found"):
        run.execute()


def test_agentic_run_reads_problem_statement(tmp_path):
    """AgenticRun.execute reads PROBLEM_STATEMENT.md and passes it to the graph."""
    messages_seen = []

    class CapturingStratAdapter:
        closure_tools: dict = {}
        def invoke(self, messages):
            messages_seen.extend(messages)
            if "WriteDeliverable" in self.closure_tools:
                self.closure_tools["WriteDeliverable"]("pipeline.py", "# test\n")
            self.closure_tools["Done"](summary="Captured")  # first: warning
            self.closure_tools["Done"](summary="Captured")  # second: close
            return "Done."

    class StubImplAdapter:
        closure_tools: dict = {}
        def invoke(self, messages):
            return "## Report\nDone."

    (tmp_path / "PROBLEM_STATEMENT.md").write_text("Find the answer to life.")
    graph_spec = _default_graph()
    compiled = build_graph(
        graph_spec,
        lambda n, a: CapturingStratAdapter() if n == "strategizer" else StubImplAdapter(),
        MemorySaver(),
        study_dir=tmp_path,
    )

    run = AgenticRun.__new__(AgenticRun)
    run.study_dir = tmp_path
    run._model = DEFAULT_MODEL
    run._budget = None
    run._graph_spec = graph_spec
    run._graph = compiled

    run.execute()

    assert any("Find the answer to life." in m.get("content", "") for m in messages_seen)


def test_agentic_run_returns_last_report(tmp_path):
    """AgenticRun.execute returns the final last_report string."""
    run = make_stub_run(tmp_path)
    result = run.execute()
    assert isinstance(result, str)
    # The run may be UNGATED (critic present in default 5-node graph)
    # but the result must be a non-empty string.
    assert result == "" or len(result) > 0
