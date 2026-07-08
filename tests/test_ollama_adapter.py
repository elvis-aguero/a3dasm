"""Tests for OllamaAdapter."""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch

# Smallest Ollama model with confirmed tool-calling support.
# Pull with: ollama pull qwen2.5:0.5b
_OLLAMA_TEST_MODEL = "qwen2.5:0.5b"


def _ollama_available() -> bool:
    try:
        import urllib.request
        urllib.request.urlopen("http://localhost:11434", timeout=2)
        return True
    except Exception:
        return False


requires_ollama = pytest.mark.skipif(
    not _ollama_available(),
    reason="Ollama server not running at localhost:11434",
)


def _make_adapter(**kwargs):
    from a3dasm._src.backends.ollama import OllamaAdapter
    defaults = dict(model="llama3.2", system_prompt="You are helpful.")
    defaults.update(kwargs)
    return OllamaAdapter(**defaults)


# ---------------------------------------------------------------------------
# Interface parity with ClaudeAdapter
# ---------------------------------------------------------------------------


def test_has_closure_tools_dict():
    adapter = _make_adapter()
    assert isinstance(adapter.closure_tools, dict)


def test_closure_tools_mutable_after_init():
    adapter = _make_adapter()
    adapter.closure_tools["foo"] = lambda x: x
    assert "foo" in adapter.closure_tools


# ---------------------------------------------------------------------------
# _build_tools maps native names to LangChain tools
# ---------------------------------------------------------------------------


def test_native_tools_mapped():
    adapter = _make_adapter(native_tools=["Bash", "Read", "Write", "Edit", "Glob", "Grep"])
    tools = adapter._build_tools()
    names = {t.name for t in tools}
    assert names == {"Bash", "Read", "Write", "Edit", "Glob", "Grep"}


def test_unknown_native_tool_ignored():
    adapter = _make_adapter(native_tools=["Bash", "NonExistent"])
    tools = adapter._build_tools()
    names = {t.name for t in tools}
    assert "NonExistent" not in names
    assert "Bash" in names


def test_closure_tools_become_structured_tools():
    def my_tool(x: str) -> str:
        """A test tool."""
        return x
    adapter = _make_adapter()
    adapter.closure_tools["MyTool"] = my_tool
    tools = adapter._build_tools()
    names = {t.name for t in tools}
    assert "MyTool" in names


# ---------------------------------------------------------------------------
# Glob tool works correctly (pattern-based, not directory listing)
# ---------------------------------------------------------------------------


def test_glob_tool_matches_pattern(tmp_path):
    (tmp_path / "a.py").write_text("x")
    (tmp_path / "b.txt").write_text("y")
    adapter = _make_adapter(native_tools=["Glob"], study_dir=tmp_path)
    tools = {t.name: t for t in adapter._build_tools()}
    result = tools["Glob"].invoke({"pattern": "*.py"})
    assert "a.py" in result
    assert "b.txt" not in result


# ---------------------------------------------------------------------------
# invoke() uses fresh thread_id per call (no state leakage)
# ---------------------------------------------------------------------------


def test_invoke_uses_fresh_thread_id_each_call():
    thread_ids = []
    fake_result = {"messages": [MagicMock(content="done")]}

    def fake_invoke(state, config=None):
        thread_ids.append(config["configurable"]["thread_id"])
        return fake_result

    fake_agent = MagicMock()
    fake_agent.invoke.side_effect = fake_invoke

    adapter = _make_adapter()
    adapter._agent = fake_agent

    adapter.invoke([{"role": "user", "content": "hello"}])
    adapter.invoke([{"role": "user", "content": "world"}])

    assert len(thread_ids) == 2
    assert thread_ids[0] != thread_ids[1]


# ---------------------------------------------------------------------------
# invoke() returns last message content
# ---------------------------------------------------------------------------


def test_invoke_returns_last_message_content():
    fake_agent = MagicMock()
    fake_agent.invoke.return_value = {
        "messages": [
            MagicMock(content="intermediate"),
            MagicMock(content="final answer"),
        ]
    }
    adapter = _make_adapter()
    adapter._agent = fake_agent
    result = adapter.invoke([{"role": "user", "content": "go"}])
    assert result == "final answer"


# ---------------------------------------------------------------------------
# Agent built lazily — closure_tools populated before first invoke
# ---------------------------------------------------------------------------


def test_agent_built_lazily():
    adapter = _make_adapter()
    assert adapter._agent is None
    adapter.closure_tools["Done"] = lambda summary: "done"

    fake_agent = MagicMock()
    fake_agent.invoke.return_value = {"messages": [MagicMock(content="ok")]}

    with patch.object(adapter, "_build_agent", return_value=fake_agent) as mock_build:
        adapter.invoke([{"role": "user", "content": "hi"}])
        mock_build.assert_called_once()
        adapter.invoke([{"role": "user", "content": "hi again"}])
        mock_build.assert_called_once()  # not rebuilt on second call


def test_done_closure_in_tools_when_agent_built():
    """Closure added before first invoke appears in built tools."""
    route = {}

    def Done(summary: str) -> str:
        """Signal done."""
        route["done"] = summary
        return "done"

    adapter = _make_adapter()
    adapter.closure_tools["Done"] = Done
    tools = adapter._build_tools()
    names = {t.name for t in tools}
    assert "Done" in names


# ---------------------------------------------------------------------------
# Integration tests — require running Ollama with qwen2.5:0.5b
# Run with: pytest -m ollama
# Pull model with: ollama pull qwen2.5:0.5b
# ---------------------------------------------------------------------------


@pytest.mark.ollama
@requires_ollama
def test_integration_invoke_returns_string():
    """OllamaAdapter.invoke returns a non-empty string against a live model."""
    adapter = _make_adapter(model=_OLLAMA_TEST_MODEL)
    result = adapter.invoke([{"role": "user", "content": "Reply with the single word: hello"}])
    assert isinstance(result, str)
    assert len(result.strip()) > 0


@pytest.mark.ollama
@requires_ollama
def test_integration_closure_tool_called(tmp_path):
    """Closure tool is actually invoked by the model during a live turn."""
    called = {}

    def RecordResult(value: str) -> str:
        """Record a result value. Call this with your answer."""
        called["value"] = value
        return "Recorded."

    adapter = _make_adapter(model=_OLLAMA_TEST_MODEL)
    adapter.closure_tools["RecordResult"] = RecordResult

    adapter.invoke([{
        "role": "user",
        "content": "Use the RecordResult tool to record the value '42'. Do not say anything else.",
    }])

    assert "value" in called, "RecordResult was never called"
    assert "42" in called["value"]


@pytest.mark.ollama
@requires_ollama
def test_integration_bash_tool_executes(tmp_path):
    """Bash tool actually runs a shell command and returns output."""
    adapter = _make_adapter(
        model=_OLLAMA_TEST_MODEL,
        native_tools=["Bash"],
        study_dir=tmp_path,
    )
    result = adapter.invoke([{
        "role": "user",
        "content": "Use the Bash tool to run: echo hello_world. Then tell me exactly what the output was.",
    }])
    assert "hello_world" in result


@pytest.mark.ollama
@requires_ollama
def test_integration_write_then_read_tool(tmp_path):
    """Write tool creates a file; Read tool retrieves its contents."""
    adapter = _make_adapter(
        model=_OLLAMA_TEST_MODEL,
        native_tools=["Write", "Read"],
        study_dir=tmp_path,
    )
    adapter.invoke([{
        "role": "user",
        "content": (
            f"Use the Write tool to write the text 'test_content_xyz' to the file "
            f"{tmp_path}/out.txt. Then use the Read tool to read it back and confirm the content."
        ),
    }])
    assert (tmp_path / "out.txt").exists()
    assert "test_content_xyz" in (tmp_path / "out.txt").read_text()


@pytest.mark.ollama
@requires_ollama
def test_integration_stateless_between_calls():
    """Two consecutive invoke() calls are independent (no state leakage)."""
    adapter = _make_adapter(model=_OLLAMA_TEST_MODEL)
    r1 = adapter.invoke([{"role": "user", "content": "Say only: FIRST"}])
    r2 = adapter.invoke([{"role": "user", "content": "Say only: SECOND"}])
    # Neither response should bleed context from the other call
    assert isinstance(r1, str) and isinstance(r2, str)


# ---------------------------------------------------------------------------
# Blindspot 3: last_usage populated after invoke
# ---------------------------------------------------------------------------


def test_last_usage_populated_from_usage_metadata():
    """last_usage is populated with input/output token counts from AIMessage.usage_metadata."""
    from langchain_core.messages import AIMessage

    fake_msg = AIMessage(content="result")
    fake_msg.usage_metadata = {"input_tokens": 75, "output_tokens": 30}

    fake_agent = MagicMock()
    fake_agent.invoke.return_value = {"messages": [fake_msg]}

    adapter = _make_adapter()
    adapter._agent = fake_agent

    adapter.invoke([{"role": "user", "content": "go"}])

    assert adapter.last_usage["input_tokens"] == 75
    assert adapter.last_usage["output_tokens"] == 30


def test_last_usage_empty_when_no_metadata():
    """last_usage defaults to zeros when AIMessage has no usage_metadata."""
    from langchain_core.messages import AIMessage

    # AIMessage without usage_metadata attribute set
    fake_msg = AIMessage(content="result")
    # Ensure usage_metadata is None (not set by default in some versions)
    if not hasattr(fake_msg, "usage_metadata"):
        object.__setattr__(fake_msg, "usage_metadata", None)
    else:
        fake_msg.usage_metadata = None

    fake_agent = MagicMock()
    fake_agent.invoke.return_value = {"messages": [fake_msg]}

    adapter = _make_adapter()
    adapter._agent = fake_agent

    # Must not raise
    adapter.invoke([{"role": "user", "content": "go"}])

    # Defaults to zeros (not a crash)
    assert adapter.last_usage.get("input_tokens", 0) == 0
    assert adapter.last_usage.get("output_tokens", 0) == 0
