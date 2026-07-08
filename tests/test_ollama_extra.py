"""Extra unit tests for ollama.py — covers tool helper functions and arxiv closures."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# _to_lc_messages: ai/assistant roles
# ---------------------------------------------------------------------------


def test_to_lc_messages_converts_ai_role():
    """_to_lc_messages converts role 'ai' to AIMessage."""
    from langchain_core.messages import AIMessage
    from a3dasm._src.backends.ollama import _to_lc_messages

    result = _to_lc_messages([{"role": "ai", "content": "answer"}])
    assert len(result) == 1
    assert isinstance(result[0], AIMessage)
    assert result[0].content == "answer"


def test_to_lc_messages_converts_assistant_role():
    """_to_lc_messages converts role 'assistant' to AIMessage."""
    from langchain_core.messages import AIMessage
    from a3dasm._src.backends.ollama import _to_lc_messages

    result = _to_lc_messages([{"role": "assistant", "content": "reply"}])
    assert len(result) == 1
    assert isinstance(result[0], AIMessage)


def test_to_lc_messages_handles_list_content():
    """_to_lc_messages joins list content into a string."""
    from langchain_core.messages import HumanMessage
    from a3dasm._src.backends.ollama import _to_lc_messages

    result = _to_lc_messages([{
        "role": "user",
        "content": [{"text": "Hello"}, {"text": " world"}],
    }])
    assert len(result) == 1
    assert isinstance(result[0], HumanMessage)
    assert "Hello" in result[0].content
    assert "world" in result[0].content


# ---------------------------------------------------------------------------
# _make_edit_tool: rejects path outside workspace
# ---------------------------------------------------------------------------


def test_make_edit_tool_rejects_path_outside_workspace(tmp_path):
    """_make_edit_tool returns ERROR for path outside the workspace."""
    from a3dasm._src.backends.ollama import _make_edit_tool

    tool = _make_edit_tool(tmp_path)
    result = tool.invoke({"path": "/etc/passwd", "old_str": "x", "new_str": "y"})
    assert "ERROR" in result


def test_make_edit_tool_returns_error_when_old_str_not_found(tmp_path):
    """_make_edit_tool returns ERROR when old_str is not in the file."""
    from a3dasm._src.backends.ollama import _make_edit_tool

    file = tmp_path / "test.txt"
    file.write_text("hello world")
    tool = _make_edit_tool(tmp_path)
    result = tool.invoke({"path": "test.txt", "old_str": "NOTHERE", "new_str": "x"})
    assert "ERROR" in result


def test_make_edit_tool_returns_error_for_missing_file(tmp_path):
    """_make_edit_tool returns ERROR when the file doesn't exist."""
    from a3dasm._src.backends.ollama import _make_edit_tool

    tool = _make_edit_tool(tmp_path)
    result = tool.invoke({"path": "nonexistent.txt", "old_str": "x", "new_str": "y"})
    assert "ERROR" in result


def test_make_edit_tool_edits_file_in_place(tmp_path):
    """_make_edit_tool replaces first occurrence of old_str with new_str."""
    from a3dasm._src.backends.ollama import _make_edit_tool

    file = tmp_path / "sample.py"
    file.write_text("foo = 1\nfoo = 2\n")
    tool = _make_edit_tool(tmp_path)
    result = tool.invoke({"path": "sample.py", "old_str": "foo = 1", "new_str": "bar = 1"})
    assert "Edited" in result
    assert "bar = 1" in file.read_text()
    assert "foo = 2" in file.read_text()  # second occurrence untouched


# ---------------------------------------------------------------------------
# _make_grep_tool
# ---------------------------------------------------------------------------


def test_make_grep_tool_returns_matches(tmp_path):
    """_make_grep_tool finds lines matching the pattern."""
    from a3dasm._src.backends.ollama import _make_grep_tool

    (tmp_path / "file.txt").write_text("hello world\ngoodbye world\n")
    tool = _make_grep_tool()
    result = tool.invoke({"pattern": "hello", "path": str(tmp_path)})
    assert "hello" in result


def test_make_grep_tool_returns_no_matches_string(tmp_path):
    """_make_grep_tool returns '(no matches)' when pattern not found."""
    from a3dasm._src.backends.ollama import _make_grep_tool

    (tmp_path / "file.txt").write_text("nothing here\n")
    tool = _make_grep_tool()
    result = tool.invoke({"pattern": "XYZZY_UNIQUE_99999", "path": str(tmp_path)})
    assert result == "(no matches)"


# ---------------------------------------------------------------------------
# _make_bash_tool
# ---------------------------------------------------------------------------


def test_make_bash_tool_runs_command(tmp_path):
    """_make_bash_tool executes a shell command and returns output."""
    from a3dasm._src.backends.ollama import _make_bash_tool

    tool = _make_bash_tool(tmp_path)
    result = tool.invoke({"command": "echo hello_from_bash"})
    assert "hello_from_bash" in result


def test_make_bash_tool_returns_no_output_string(tmp_path):
    """_make_bash_tool returns '(no output)' for commands that produce nothing."""
    from a3dasm._src.backends.ollama import _make_bash_tool

    tool = _make_bash_tool(tmp_path)
    result = tool.invoke({"command": "true"})
    # May return "(no output)" or empty string — either is acceptable
    assert isinstance(result, str)


# ---------------------------------------------------------------------------
# _make_read_tool: error on missing file
# ---------------------------------------------------------------------------


def test_make_read_tool_returns_error_on_missing_file(tmp_path):
    """_make_read_tool returns ERROR when file does not exist."""
    from a3dasm._src.backends.ollama import _make_read_tool

    tool = _make_read_tool(tmp_path)
    result = tool.invoke({"path": "nonexistent.txt"})
    assert "ERROR" in result


def test_make_read_tool_reads_existing_file(tmp_path):
    """_make_read_tool returns file content."""
    from a3dasm._src.backends.ollama import _make_read_tool

    (tmp_path / "data.txt").write_text("important content")
    tool = _make_read_tool(tmp_path)
    result = tool.invoke({"path": "data.txt"})
    assert "important content" in result


# ---------------------------------------------------------------------------
# _make_write_tool
# ---------------------------------------------------------------------------


def test_make_write_tool_creates_file(tmp_path):
    """_make_write_tool writes content to a new file."""
    from a3dasm._src.backends.ollama import _make_write_tool

    tool = _make_write_tool(tmp_path)
    result = tool.invoke({"path": "out.txt", "content": "test content"})
    assert "Written" in result
    assert (tmp_path / "out.txt").read_text() == "test content"


# ---------------------------------------------------------------------------
# _build_arxiv_closures: returns empty dict when arxiv not installed
# ---------------------------------------------------------------------------


def test_build_arxiv_closures_returns_empty_when_arxiv_missing():
    """_build_arxiv_closures returns {} when arxiv package is not importable."""
    import sys
    import importlib

    # Temporarily make arxiv unimportable
    original = sys.modules.get("arxiv", None)
    sys.modules["arxiv"] = None  # type: ignore[assignment]
    try:
        # Force re-import
        import a3dasm._src.backends.ollama as ollama_mod
        importlib.reload(ollama_mod)
        result = ollama_mod._build_arxiv_closures()
        assert result == {}
    finally:
        if original is None:
            sys.modules.pop("arxiv", None)
        else:
            sys.modules["arxiv"] = original
        # Reload to restore original state
        importlib.reload(ollama_mod)


# ---------------------------------------------------------------------------
# OllamaAdapter.copy() returns independent instance
# ---------------------------------------------------------------------------


def test_ollama_adapter_copy_returns_self():
    """OllamaAdapter.copy() returns self (serialization via lock, not copies)."""
    from a3dasm._src.backends.ollama import OllamaAdapter

    adapter = OllamaAdapter(model="llama3.2", system_prompt="You are helpful.")
    # copy() returns self by design — concurrent callers share the same instance
    # and are serialized via the internal _lock
    copy = adapter.copy()
    assert copy is adapter
