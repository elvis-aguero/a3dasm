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
# Orchestrating node (has outgoing edges) → RUN_PATHS preamble
# ---------------------------------------------------------------------------


def test_make_adapter_orchestrating_node_uses_run_paths_preamble(tmp_path):
    """_make_adapter for a node with outgoing edges uses RUN_PATHS_PREAMBLE."""
    run = _make_run(tmp_path)

    agent = _agent_with_tools("Bash")

    # Patch ClaudeAdapter so we don't need real credentials
    with patch("a3dasm._src.backends.claude.ClaudeAdapter") as MockClaude:
        mock_instance = MagicMock()
        mock_instance.closure_tools = {}
        MockClaude.return_value = mock_instance

        # Mock graph_spec.outgoing to return a non-empty list
        run._graph_spec = MagicMock()
        run._graph_spec.outgoing.return_value = ["implementer"]

        result = run._make_adapter("strategizer", agent)

    # Should have been constructed with a system_prompt starting with <run_paths>
    call_kwargs = MockClaude.call_args[1]
    assert "system_prompt" in call_kwargs
    assert "<run_paths>" in call_kwargs["system_prompt"]


# ---------------------------------------------------------------------------
# Worker node (no outgoing edges) → WORKSPACE preamble
# ---------------------------------------------------------------------------


def test_make_adapter_worker_node_uses_workspace_preamble(tmp_path):
    """_make_adapter for a leaf node (no outgoing) uses WORKSPACE_PREAMBLE."""
    run = _make_run(tmp_path)

    agent = _agent_with_tools("Bash")

    with patch("a3dasm._src.backends.claude.ClaudeAdapter") as MockClaude:
        mock_instance = MagicMock()
        mock_instance.closure_tools = {}
        MockClaude.return_value = mock_instance

        # Mock graph_spec.outgoing to return empty list (leaf node)
        run._graph_spec = MagicMock()
        run._graph_spec.outgoing.return_value = []

        result = run._make_adapter("implementer", agent)

    call_kwargs = MockClaude.call_args[1]
    assert "system_prompt" in call_kwargs
    assert "<workspace>" in call_kwargs["system_prompt"]


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
        run._graph_spec.outgoing.return_value = []  # leaf node (run_dir is None anyway)

        # Should not raise even when run_dir is None
        result = run._make_adapter("implementer", agent)

    assert result is mock_instance
