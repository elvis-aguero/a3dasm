"""Tests for ClaudeAdapter — stub out claude_agent_sdk.query."""
import types
import sys


# ---------------------------------------------------------------------------
# Helpers to build realistic fake SDK event streams
# ---------------------------------------------------------------------------

class _TextBlock:
    def __init__(self, text: str) -> None:
        self.text = text


class _AssistantMessage:
    def __init__(self, blocks) -> None:
        self.content = blocks


class _ResultMessage:
    usage = None
    total_cost_usd = None


class _StreamEvent:
    pass


class _UserMessage:
    def __init__(self, blocks=None) -> None:
        self.content = blocks or []


class _ToolUseBlockType:
    pass


def make_async_gen_with_messages(*text_contents):
    """Yield AssistantMessage-like events with TextBlock content, then ResultMessage."""
    async def _gen(prompt, options):
        yield _AssistantMessage([_TextBlock(t) for t in text_contents])
        yield _ResultMessage()
    return _gen


def _install_fake_sdk(**extra):
    """Install (or update) a fake claude_agent_sdk in sys.modules."""
    mod = sys.modules.get("claude_agent_sdk")
    if mod is None:
        mod = types.ModuleType("claude_agent_sdk")
        sys.modules["claude_agent_sdk"] = mod

    mod.AssistantMessage = _AssistantMessage
    mod.ResultMessage = _ResultMessage
    mod.TextBlock = _TextBlock
    mod.SdkMcpTool = object  # not used in these tests
    mod.StreamEvent = _StreamEvent
    mod.UserMessage = _UserMessage
    mod.ToolUseBlock = _ToolUseBlockType
    mod.ClaudeAgentOptions = lambda **kw: kw
    mod.create_sdk_mcp_server = lambda name=None, tools=None: {"name": name}

    for k, v in extra.items():
        setattr(mod, k, v)

    return mod


def _get_adapter():
    """Return the (possibly reloaded) ClaudeAdapter with SDK pre-marked available."""
    import a3dasm._src.backends.claude as cmod
    cmod._SDK_AVAILABLE = True
    return cmod.ClaudeAdapter


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_single_text_block():
    """Single TextBlock → returns that text exactly."""
    _install_fake_sdk(query=make_async_gen_with_messages("hello world"))
    ClaudeAdapter = _get_adapter()
    adapter = ClaudeAdapter("claude-3", "sys", None, [])
    result = adapter.invoke([{"role": "user", "content": "hi"}])
    assert result == "hello world"


def test_transcript_captured_when_debug_on(tmp_path, monkeypatch):
    """With F3DASM_DEBUG on and a sink set, ainvoke streams assistant text +
    result records to the transcript JSONL."""
    import json
    from a3dasm._src.backends.base import set_transcript_sink
    monkeypatch.setenv("F3DASM_DEBUG", "1")
    _install_fake_sdk(query=make_async_gen_with_messages("reasoning here"))
    ClaudeAdapter = _get_adapter()
    adapter = ClaudeAdapter("claude-3", "sys", None, [])
    sink = tmp_path / "D001.jsonl"
    set_transcript_sink(str(sink))
    adapter.invoke([{"role": "user", "content": "hi"}])
    set_transcript_sink(None)

    recs = [json.loads(x) for x in sink.read_text().strip().splitlines()]
    types = [r["type"] for r in recs]
    assert "assistant" in types
    asst = next(r for r in recs if r["type"] == "assistant")
    assert asst["text"] == "reasoning here"


def test_partials_flushed_for_incomplete_turn(tmp_path, monkeypatch):
    """An agent that never completes a message (infinite thinking) emits only
    StreamEvents — the transcript must STILL disclose its partial output via
    periodic flushes, not stay empty."""
    import json

    from a3dasm._src.backends.base import set_transcript_sink

    class _StreamEv:
        def __init__(self, text):
            self.event = {"delta": {"type": "text_delta", "text": text}}

    def _gen_never_completes(prompt, options):
        async def _g(prompt, options):
            # 60 stream events, NO AssistantMessage, NO ResultMessage.
            for i in range(60):
                yield _StreamEv(f"tok{i} ")
        return _g

    monkeypatch.setenv("F3DASM_DEBUG", "1")
    mod = _install_fake_sdk(query=_gen_never_completes(None, None))
    mod.StreamEvent = _StreamEv  # adapter isinstance-checks this
    ClaudeAdapter = _get_adapter()
    adapter = ClaudeAdapter("claude-3", "sys", None, [])
    sink = tmp_path / "D001.jsonl"
    set_transcript_sink(str(sink))
    adapter.invoke([{"role": "user", "content": "hi"}])
    set_transcript_sink(None)

    recs = [json.loads(x) for x in sink.read_text().strip().splitlines()]
    partials = [r for r in recs if r["type"] == "partial"]
    assert partials, "incomplete turn disclosed nothing"
    # 60 events → flush at 25, 50, plus the teardown flush for the tail.
    assert any("tok49" in r["text"] for r in partials)


def _capture_options_gen(captured: dict):
    """A fake query that records the ClaudeAgentOptions dict it was given."""
    async def _gen(prompt, options):
        captured["options"] = options
        yield _AssistantMessage([_TextBlock("ok")])
        yield _ResultMessage()
    return _gen


def test_infer_schema_skips_underscore_closure_params():
    """Closure capture-args (_ws=..., _did=...) must NOT appear in the tool
    schema — else a model can pass them (e.g. _ws as a string) and crash a
    write tool with str / path. Real params are still exposed."""
    from a3dasm._src.backends.claude import _infer_schema_from_callable

    def _write(path: str, body: str, _ws="x", _did="D001"):
        return ""

    schema = _infer_schema_from_callable(_write)
    props = schema["properties"]
    assert "path" in props and "body" in props
    assert "_ws" not in props and "_did" not in props
    assert set(schema.get("required", [])) == {"path", "body"}


def test_max_buffer_size_is_set_and_tunable(monkeypatch):
    """The 1MB default crashed the lit reviewer on a >1MB PDF result; we set
    30MB (env-tunable) so realistic large tool results don't overflow."""
    cap: dict = {}
    _install_fake_sdk(query=_capture_options_gen(cap))
    ClaudeAdapter = _get_adapter()
    ClaudeAdapter("claude-3", "sys", None, []).invoke(
        [{"role": "user", "content": "hi"}])
    assert cap["options"]["max_buffer_size"] == 30 * 1024 * 1024
    cap.clear()
    monkeypatch.setenv("F3DASM_LLM_MAX_BUFFER_MB", "50")
    _install_fake_sdk(query=_capture_options_gen(cap))
    _get_adapter()("claude-3", "sys", None, []).invoke(
        [{"role": "user", "content": "hi"}])
    assert cap["options"]["max_buffer_size"] == 50 * 1024 * 1024


def test_buffer_overflow_is_graceful_not_fatal(monkeypatch):
    """A buffer-overflow mid-stream must NOT crash the delegation — the turn
    ends with a clear marker so the agent can retry smaller (gracefully
    contour), instead of the old fatal non-retried FAILED."""
    def _overflow_gen(prompt, options):
        async def _g(prompt, options):
            yield _AssistantMessage([_TextBlock("partial work")])
            raise Exception(
                "Failed to decode JSON: JSON message exceeded maximum "
                "buffer size of 1048576 bytes")
        return _g
    _install_fake_sdk(query=_overflow_gen(None, None))
    ClaudeAdapter = _get_adapter()
    # Does NOT raise — returns gracefully with the marker appended.
    out = ClaudeAdapter("claude-3", "sys", None, []).invoke(
        [{"role": "user", "content": "hi"}])
    assert "partial work" in out
    assert "overflowed the message buffer" in out


def test_non_buffer_stream_error_still_raises():
    """Only buffer-overflow is contoured; other stream errors still propagate
    (so retry_on_transient / FAILED handling stays intact)."""
    import pytest as _pytest

    def _err_gen(prompt, options):
        async def _g(prompt, options):
            yield _AssistantMessage([_TextBlock("x")])
            raise RuntimeError("some other fatal error")
        return _g
    _install_fake_sdk(query=_err_gen(None, None))
    ClaudeAdapter = _get_adapter()
    with _pytest.raises(RuntimeError):
        ClaudeAdapter("claude-3", "sys", None, []).invoke(
            [{"role": "user", "content": "hi"}])


def test_session_is_hermetic_setting_sources_empty():
    """#1 fresh hooks: sessions load NO filesystem settings, so worker/critic
    subprocesses don't inherit the developer's global ~/.claude hooks."""
    cap: dict = {}
    _install_fake_sdk(query=_capture_options_gen(cap))
    ClaudeAdapter = _get_adapter()
    ClaudeAdapter("claude-3", "sys", None, []).invoke(
        [{"role": "user", "content": "hi"}])
    assert cap["options"]["setting_sources"] == []


def test_delegation_id_injected_into_session_env(monkeypatch):
    """Finding 2: the bound delegation id reaches the session env as
    F3DASM_DELEGATION_ID so get_evaluator() resolves without a cd into D###."""
    from a3dasm._src.backends.base import set_delegation_id
    cap: dict = {}
    _install_fake_sdk(query=_capture_options_gen(cap))
    ClaudeAdapter = _get_adapter()
    adapter = ClaudeAdapter("claude-3", "sys", None, [])

    set_delegation_id("D007")
    adapter.invoke([{"role": "user", "content": "hi"}])
    set_delegation_id(None)
    assert cap["options"]["env"].get("F3DASM_DELEGATION_ID") == "D007"

    # With no delegation bound, the key is absent (no stray injection).
    cap.clear()
    adapter.invoke([{"role": "user", "content": "hi"}])
    assert "F3DASM_DELEGATION_ID" not in cap["options"]["env"]


def test_stream_event_types_captured_for_ping_measurement(tmp_path, monkeypatch):
    """Records each non-delta StreamEvent's type + inter-event gap, so a run
    reveals whether ping/lifecycle events arrive during silent phases (the
    data that settles whether 60s silence is a dead stream or slow prefill)."""
    import json
    from a3dasm._src.backends.base import set_transcript_sink

    class _SE:
        def __init__(self, etype):
            self.event = {"type": etype}

    def _gen_factory(prompt, options):
        async def _g(prompt, options):
            yield _SE("message_start")
            yield _SE("ping")
            yield _SE("content_block_delta")  # delta, fast → not recorded
            yield _AssistantMessage([_TextBlock("hi")])
            yield _ResultMessage()
        return _g

    monkeypatch.setenv("F3DASM_DEBUG", "1")
    mod = _install_fake_sdk(query=_gen_factory(None, None))
    mod.StreamEvent = _SE
    ClaudeAdapter = _get_adapter()
    sink = tmp_path / "D001.jsonl"
    set_transcript_sink(str(sink))
    ClaudeAdapter("claude-3", "sys", None, []).invoke(
        [{"role": "user", "content": "hi"}])
    set_transcript_sink(None)

    recs = [json.loads(x) for x in sink.read_text().strip().splitlines()]
    evts = [r["evt"] for r in recs if r["type"] == "stream_evt"]
    assert "message_start" in evts and "ping" in evts
    assert all("gap_s" in r for r in recs if r["type"] == "stream_evt")


def test_no_transcript_when_debug_off(tmp_path, monkeypatch):
    from a3dasm._src.backends.base import set_transcript_sink
    monkeypatch.delenv("F3DASM_DEBUG", raising=False)
    _install_fake_sdk(query=make_async_gen_with_messages("x"))
    ClaudeAdapter = _get_adapter()
    adapter = ClaudeAdapter("claude-3", "sys", None, [])
    sink = tmp_path / "D001.jsonl"
    set_transcript_sink(str(sink))
    adapter.invoke([{"role": "user", "content": "hi"}])
    set_transcript_sink(None)
    assert not sink.exists()


def test_multiple_text_blocks_concatenated():
    """Multiple TextBlocks in one AssistantMessage → concatenated."""
    _install_fake_sdk(query=make_async_gen_with_messages("foo", "bar", "baz"))
    ClaudeAdapter = _get_adapter()
    adapter = ClaudeAdapter("claude-3", "sys", None, [])
    result = adapter.invoke([{"role": "user", "content": "hi"}])
    assert result == "foobarbaz"


def test_non_text_blocks_are_ignored():
    """Non-TextBlock content blocks are skipped; only TextBlock.text is collected."""

    class _ToolUseBlock:
        """Simulates a tool_use content block — should be ignored."""
        name = "Bash"
        input = {"command": "ls"}

    async def _gen_mixed(prompt, options):
        # AssistantMessage with mixed block types
        yield _AssistantMessage([
            _TextBlock("kept"),
            _ToolUseBlock(),
            _TextBlock(" also kept"),
        ])
        yield _ResultMessage()

    _install_fake_sdk(query=_gen_mixed)
    ClaudeAdapter = _get_adapter()
    adapter = ClaudeAdapter("claude-3", "sys", None, [])
    result = adapter.invoke([{"role": "user", "content": "hi"}])
    assert result == "kept also kept"


# ---------------------------------------------------------------------------
# Blindspot 2: last_usage populated from ResultMessage
# ---------------------------------------------------------------------------


def test_last_usage_populated_from_result_message():
    """last_usage is populated with usage fields from ResultMessage after invoke."""

    class _ResultMessageWithUsage:
        usage = {"input_tokens": 50, "output_tokens": 20}
        total_cost_usd = 0.002

    async def _gen_with_usage(prompt, options):
        yield _AssistantMessage([_TextBlock("response text")])
        yield _ResultMessageWithUsage()

    _install_fake_sdk(
        query=_gen_with_usage,
        ResultMessage=_ResultMessageWithUsage,
    )
    ClaudeAdapter = _get_adapter()
    adapter = ClaudeAdapter("claude-3", "sys", None, [])
    adapter.invoke([{"role": "user", "content": "hi"}])

    assert adapter.last_usage["input_tokens"] == 50
    assert adapter.last_usage["output_tokens"] == 20
    assert adapter.last_usage["total_cost_usd"] == 0.002


def test_last_usage_empty_when_no_result_message():
    """last_usage is empty dict when no ResultMessage is yielded."""

    async def _gen_no_result(prompt, options):
        yield _AssistantMessage([_TextBlock("response text")])
        # No ResultMessage yielded — generator just ends

    _install_fake_sdk(query=_gen_no_result)
    ClaudeAdapter = _get_adapter()
    adapter = ClaudeAdapter("claude-3", "sys", None, [])
    adapter.invoke([{"role": "user", "content": "hi"}])

    # last_usage should be empty (or all zeroes / None) — not a crash
    assert adapter.last_usage == {} or not any(
        v for v in adapter.last_usage.values() if v
    )
