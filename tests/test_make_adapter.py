"""Tests for AgenticRun._make_adapter() — preamble selection and backend routing."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from a3dasm._src.agent_runtime import (
    DEFAULT_MODEL,
    AgenticRun,
    _default_graph,
)
from a3dasm._src.backends.base import Agent, Edge, Graph


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_run(tmp_path: Path, backend: str = "claude") -> AgenticRun:
    """Build a bare AgenticRun via __new__ with minimal attributes set."""
    (tmp_path / "PROBLEM_STATEMENT.md").write_text("test")
    run = AgenticRun.__new__(AgenticRun)
    run.study_dir = tmp_path
    run._model = DEFAULT_MODEL
    run._backend = backend
    run._graph_spec = _default_graph()
    # Create a run_dir so _make_adapter can compute paths
    run_dir = tmp_path / "runs" / "test_run"
    (run_dir / "debug" / "strategizer_notes").mkdir(parents=True, exist_ok=True)
    run._run_dir = run_dir
    return run


def _agent_with_tools(*tools) -> Agent:
    class _A(Agent):
        description = "Test agent."

    _A.tools = frozenset(tools)
    return _A()


# ---------------------------------------------------------------------------
# Entry/orchestrator node → RUN_PATHS preamble + cwd=study_dir
# ---------------------------------------------------------------------------


def test_make_adapter_entry_node_uses_run_paths_preamble(tmp_path):
    """_make_adapter for the graph's ENTRY node uses RUN_PATHS_PREAMBLE and
    cwd=study_dir — this is keyed off being the entry node specifically, NOT
    off merely having outgoing edges (see the non-entry-with-outgoing-edges
    regression test below for why that distinction matters)."""
    run = _make_run(tmp_path)

    agent = _agent_with_tools("Bash")

    # Patch ClaudeAdapter so we don't need real credentials
    with patch("a3dasm._src.backends.claude.ClaudeAdapter") as MockClaude:
        mock_instance = MagicMock()
        mock_instance.closure_tools = {}
        MockClaude.return_value = mock_instance

        run._graph_spec = MagicMock()
        run._graph_spec.entry = "strategizer"
        run._graph_spec.outgoing.return_value = ["implementer"]

        result = run._make_adapter("strategizer", agent)

    call_kwargs = MockClaude.call_args[1]
    assert "system_prompt" in call_kwargs
    assert "<run_paths>" in call_kwargs["system_prompt"]
    assert call_kwargs["study_dir"] == run.study_dir  # cwd == study_dir


# ---------------------------------------------------------------------------
# Non-entry node (leaf, no outgoing edges) → WORKSPACE preamble
# ---------------------------------------------------------------------------


def test_make_adapter_worker_node_uses_workspace_preamble(tmp_path):
    """_make_adapter for a leaf node (not the entry, no outgoing) uses
    WORKSPACE_PREAMBLE and cwd=run_dir/debug/delegations."""
    run = _make_run(tmp_path)

    agent = _agent_with_tools("Bash")

    with patch("a3dasm._src.backends.claude.ClaudeAdapter") as MockClaude:
        mock_instance = MagicMock()
        mock_instance.closure_tools = {}
        MockClaude.return_value = mock_instance

        run._graph_spec = MagicMock()
        run._graph_spec.entry = "strategizer"
        run._graph_spec.outgoing.return_value = []

        result = run._make_adapter("implementer", agent)

    call_kwargs = MockClaude.call_args[1]
    assert "system_prompt" in call_kwargs
    assert "<workspace>" in call_kwargs["system_prompt"]
    assert call_kwargs["study_dir"] == run._run_dir / "debug" / "delegations"


def test_make_adapter_non_entry_node_with_outgoing_edges_uses_workspace_preamble(
    tmp_path
):
    """Regression (run 20260718T132852): datagenerator/implementer each have
    their OWN outgoing edge to literature_reviewer (for sub-delegating a
    lookup — see agents/_graphs.py) but are NOT the entry node. Before the
    fix, _make_adapter keyed the RUN_PATHS/cwd=study_dir choice off "has ANY
    outgoing edge", so these two roles wrongly got cwd=study_dir instead of
    the run-scoped workspace their OWN preamble promises — splitting their
    delegation output across two physical trees (study_dir/debug/delegations
    vs run_dir/debug/delegations) with the same D### ids in both. A non-entry
    node must get the WORKSPACE preamble and the run-scoped cwd regardless of
    whether it has its own outgoing edges."""
    run = _make_run(tmp_path)

    agent = _agent_with_tools("Bash")

    with patch("a3dasm._src.backends.claude.ClaudeAdapter") as MockClaude:
        mock_instance = MagicMock()
        mock_instance.closure_tools = {}
        MockClaude.return_value = mock_instance

        run._graph_spec = MagicMock()
        run._graph_spec.entry = "strategizer"
        # datagenerator has its own outgoing edge (to literature_reviewer)
        # but must still be treated as a worker, not the orchestrator.
        run._graph_spec.outgoing.return_value = ["literature_reviewer"]

        result = run._make_adapter("datagenerator", agent)

    call_kwargs = MockClaude.call_args[1]
    assert "<workspace>" in call_kwargs["system_prompt"]
    assert "<run_paths>" not in call_kwargs["system_prompt"]
    assert call_kwargs["study_dir"] == run._run_dir / "debug" / "delegations"


# ---------------------------------------------------------------------------
# Ollama backend → OllamaAdapter created
# ---------------------------------------------------------------------------


def test_make_adapter_ollama_backend_creates_ollama_adapter(tmp_path):
    """_make_adapter with backend='ollama' returns an OllamaAdapter instance."""
    run = _make_run(tmp_path, backend="ollama")

    agent = _agent_with_tools("Bash")

    # Registry dispatch resolves OllamaAdapter from its source module, so
    # patching it there is all that's needed.
    with patch("a3dasm._src.backends.ollama.OllamaAdapter") as MockOllama:
        mock_instance = MagicMock()
        mock_instance.closure_tools = {}
        MockOllama.return_value = mock_instance

        run._graph_spec = MagicMock()
        run._graph_spec.entry = "strategizer"
        run._graph_spec.outgoing.return_value = []

        result = run._make_adapter("implementer", agent)

    # The result should be the mock OllamaAdapter instance
    assert result is mock_instance


# ---------------------------------------------------------------------------
# _make_adapter when run_dir is None — returns adapter without RUN_PATHS
# ---------------------------------------------------------------------------


def test_make_adapter_no_run_dir_returns_adapter_without_run_paths(tmp_path):
    """When run_dir is None, _make_adapter falls through to WORKSPACE path."""
    run = _make_run(tmp_path)
    run._run_dir = None  # simulate pre-execute state

    agent = _agent_with_tools("Bash")

    with patch("a3dasm._src.backends.claude.ClaudeAdapter") as MockClaude:
        mock_instance = MagicMock()
        mock_instance.closure_tools = {}
        MockClaude.return_value = mock_instance

        run._graph_spec = MagicMock()
        run._graph_spec.entry = "strategizer"
        run._graph_spec.outgoing.return_value = []  # leaf node (run_dir is None anyway)

        # Should not raise even when run_dir is None
        result = run._make_adapter("implementer", agent)

    assert result is mock_instance
