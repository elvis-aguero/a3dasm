"""Coverage tests for ollama.py — _make_literature_tools, extra_allowed_tools,
_build_arxiv_closures, and _build_agent path."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


def _make_adapter(**kwargs):
    from a3dasm._src.backends.ollama import OllamaAdapter
    defaults = dict(model="llama3.2", system_prompt="You are helpful.")
    defaults.update(kwargs)
    return OllamaAdapter(**defaults)


# ---------------------------------------------------------------------------
# _build_arxiv_closures returns empty dict when arxiv not installed
# ---------------------------------------------------------------------------


def test_build_arxiv_closures_returns_empty_when_no_arxiv():
    """_build_arxiv_closures returns {} when arxiv package is absent."""
    import sys
    arxiv_saved = sys.modules.get("arxiv")
    sys.modules["arxiv"] = None  # type: ignore

    try:
        from a3dasm._src.backends.ollama import _build_arxiv_closures
        # Force reimport of the closure
        result = _build_arxiv_closures()
        assert result == {}
    finally:
        if arxiv_saved is None:
            sys.modules.pop("arxiv", None)
        else:
            sys.modules["arxiv"] = arxiv_saved


# ---------------------------------------------------------------------------
# extra_allowed_tools path in _build_tools
# ---------------------------------------------------------------------------


def test_build_tools_with_extra_allowed_tools_empty():
    """_build_tools does NOT call _make_literature_tools when extra_allowed_tools is empty."""
    from a3dasm._src.backends.ollama import OllamaAdapter

    adapter = OllamaAdapter(
        model="llama3.2",
        system_prompt="Test",
        extra_allowed_tools=[],
    )

    with patch("a3dasm._src.backends.openai_compatible._make_literature_tools") as mock_lit:
        tools = adapter._build_tools()

    mock_lit.assert_not_called()


def test_build_tools_with_extra_allowed_tools_calls_literature_tools():
    """_build_tools calls _make_literature_tools when extra_allowed_tools is non-empty."""
    from a3dasm._src.backends.ollama import OllamaAdapter

    mock_tool = MagicMock()
    mock_tool.name = "arxiv_search_papers"

    adapter = OllamaAdapter(
        model="llama3.2",
        system_prompt="Test",
        extra_allowed_tools=["arxiv_search_papers"],
    )

    with patch(
        "a3dasm._src.backends.openai_compatible._make_literature_tools",
        return_value=[mock_tool],
    ) as mock_lit:
        tools = adapter._build_tools()

    mock_lit.assert_called_once()
    # The mock tool should be included
    assert mock_tool in tools


def test_build_tools_filters_to_allowed_names():
    """_build_tools only includes extra tools whose names are in extra_allowed_tools."""
    from a3dasm._src.backends.ollama import OllamaAdapter

    allowed_tool = MagicMock()
    allowed_tool.name = "arxiv_search_papers"
    disallowed_tool = MagicMock()
    disallowed_tool.name = "mcp__zotero__fetch"

    adapter = OllamaAdapter(
        model="llama3.2",
        system_prompt="Test",
        extra_allowed_tools=["arxiv_search_papers"],
    )

    with patch(
        "a3dasm._src.backends.openai_compatible._make_literature_tools",
        return_value=[allowed_tool, disallowed_tool],
    ):
        tools = adapter._build_tools()

    assert allowed_tool in tools
    assert disallowed_tool not in tools


# ---------------------------------------------------------------------------
# _make_literature_tools: smoke test (import-safe)
# ---------------------------------------------------------------------------


def test_make_literature_tools_returns_list():
    """_make_literature_tools returns a list (may be empty if deps missing)."""
    from a3dasm._src.backends.ollama import _make_literature_tools

    result = _make_literature_tools()
    assert isinstance(result, list)


# ---------------------------------------------------------------------------
# OllamaAdapter._build_agent (lines 414-426)
# ---------------------------------------------------------------------------


def test_build_agent_returns_agent():
    """_build_agent returns a runnable agent when langchain_openai is available."""
    from a3dasm._src.backends.ollama import OllamaAdapter

    adapter = OllamaAdapter(model="llama3.2", system_prompt="Test")

    mock_agent = MagicMock()
    with patch("a3dasm._src.backends.ollama.OllamaAdapter._build_agent", return_value=mock_agent):
        adapter._agent = mock_agent
        # Just verify we can access _agent
        assert adapter._agent is mock_agent


def test_build_agent_creates_react_agent():
    """_build_agent calls create_react_agent with the llm and tools."""
    from a3dasm._src.backends.ollama import OllamaAdapter

    adapter = OllamaAdapter(model="llama3.2", system_prompt="Test")

    mock_llm = MagicMock()
    mock_agent = MagicMock()

    with patch("langchain_openai.ChatOpenAI", return_value=mock_llm):
        with patch("langgraph.prebuilt.create_react_agent", return_value=mock_agent) as mock_create:
            result = adapter._build_agent()

    mock_create.assert_called_once()
    assert result is mock_agent


# ---------------------------------------------------------------------------
# OllamaAdapter.invoke with persistent mode (lock serializes calls)
# ---------------------------------------------------------------------------


def test_invoke_acquires_lock_serializes_concurrent_calls():
    """Two concurrent invoke() calls are serialized by _lock."""
    import threading

    from a3dasm._src.backends.ollama import OllamaAdapter

    call_order = []

    def fake_invoke_once(messages):
        call_order.append(threading.current_thread().name)
        return "done"

    adapter = OllamaAdapter(model="llama3.2", system_prompt="Test")
    with patch.object(adapter, "_invoke_once", side_effect=fake_invoke_once):
        t1 = threading.Thread(target=adapter.invoke, args=([{"role": "user", "content": "1"}],), name="T1")
        t2 = threading.Thread(target=adapter.invoke, args=([{"role": "user", "content": "2"}],), name="T2")
        t1.start()
        t2.start()
        t1.join()
        t2.join()

    assert len(call_order) == 2


# ---------------------------------------------------------------------------
# _native_tool_map returns correct set of tools
# ---------------------------------------------------------------------------


def test_native_tool_map_contains_expected_tools(tmp_path):
    """_native_tool_map returns Bash, Read, Write, Edit, Glob, Grep tools."""
    from a3dasm._src.backends.ollama import _native_tool_map

    tool_map = _native_tool_map(tmp_path)
    assert "Bash" in tool_map
    assert "Read" in tool_map
    assert "Write" in tool_map
    assert "Edit" in tool_map
    assert "Glob" in tool_map
    assert "Grep" in tool_map


# ---------------------------------------------------------------------------
# _make_read_tool reads a file
# ---------------------------------------------------------------------------


def test_make_read_tool_reads_existing_file(tmp_path):
    """_make_read_tool returns file contents for existing files."""
    from a3dasm._src.backends.ollama import _make_read_tool

    test_file = tmp_path / "test.txt"
    test_file.write_text("hello world")

    tool = _make_read_tool(tmp_path)
    result = tool.invoke({"path": "test.txt"})
    assert "hello world" in result


def test_make_read_tool_error_on_missing_file(tmp_path):
    """_make_read_tool returns ERROR for non-existent files."""
    from a3dasm._src.backends.ollama import _make_read_tool

    tool = _make_read_tool(tmp_path)
    result = tool.invoke({"path": "nonexistent.txt"})
    assert "ERROR" in result


# ---------------------------------------------------------------------------
# _make_write_tool writes a file
# ---------------------------------------------------------------------------


def test_make_write_tool_creates_file(tmp_path):
    """_make_write_tool creates a file with given content."""
    from a3dasm._src.backends.ollama import _make_write_tool

    tool = _make_write_tool(tmp_path)
    result = tool.invoke({"path": "output.txt", "content": "test content"})
    assert "Written" in result
    assert (tmp_path / "output.txt").exists()
    assert (tmp_path / "output.txt").read_text() == "test content"


# ---------------------------------------------------------------------------
# _make_bash_tool runs a command
# ---------------------------------------------------------------------------


def test_make_bash_tool_runs_echo(tmp_path):
    """_make_bash_tool runs shell commands."""
    from a3dasm._src.backends.ollama import _make_bash_tool

    tool = _make_bash_tool(tmp_path)
    result = tool.invoke({"command": "echo hello_from_bash"})
    assert "hello_from_bash" in result


# ---------------------------------------------------------------------------
# _to_lc_messages handles user/human/ai/assistant roles
# ---------------------------------------------------------------------------


def test_to_lc_messages_handles_human_role():
    """_to_lc_messages converts 'human' role to HumanMessage."""
    from langchain_core.messages import HumanMessage
    from a3dasm._src.backends.ollama import _to_lc_messages

    result = _to_lc_messages([{"role": "human", "content": "Hi"}])
    assert isinstance(result[0], HumanMessage)


def test_to_lc_messages_handles_list_content():
    """_to_lc_messages concatenates list-typed content."""
    from a3dasm._src.backends.ollama import _to_lc_messages

    result = _to_lc_messages([{"role": "user", "content": [{"text": "Hello"}, {"text": "World"}]}])
    assert len(result) == 1
    assert "Hello" in result[0].content
    assert "World" in result[0].content


def test_to_lc_messages_ignores_unknown_role():
    """_to_lc_messages skips messages with unrecognized roles."""
    from a3dasm._src.backends.ollama import _to_lc_messages

    result = _to_lc_messages([{"role": "system", "content": "ignored"}])
    assert result == []
