"""Unit tests for the OpenAI-compatible backends (OpenRouter, vLLM).

Fully mocked — no API keys, no network. They verify the per-backend endpoint
and auth conventions (explicit arg > env > class default) and that the shared
invoke/usage machinery works through the subclass. Cross-backend interface
parity is enforced separately in test_backend_parity.py.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from a3dasm._src.backends.openrouter import OpenRouterAdapter
from a3dasm._src.backends.vllm import VLLMAdapter

# --------------------------------------------------------------------------
# Endpoint + auth resolution: explicit > env > class default
# --------------------------------------------------------------------------

def test_openrouter_defaults(monkeypatch):
    monkeypatch.delenv("OPENROUTER_BASE_URL", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    a = OpenRouterAdapter(model="anthropic/claude-3.5-sonnet", system_prompt="s")
    assert a._base_url == "https://openrouter.ai/api/v1"
    assert a._api_key is None  # no key baked in — must be supplied


def test_openrouter_reads_env(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-secret")
    monkeypatch.setenv("OPENROUTER_BASE_URL", "https://proxy.example/v1")
    a = OpenRouterAdapter(model="m", system_prompt="s")
    assert a._base_url == "https://proxy.example/v1"
    assert a._api_key == "or-secret"


def test_vllm_defaults(monkeypatch):
    monkeypatch.delenv("VLLM_BASE_URL", raising=False)
    monkeypatch.delenv("VLLM_API_KEY", raising=False)
    a = VLLMAdapter(model="meta-llama/Llama-3.1-8B-Instruct", system_prompt="s")
    assert a._base_url == "http://localhost:8000/v1"
    assert a._api_key == "EMPTY"  # vLLM usually needs no auth


def test_vllm_reads_env(monkeypatch):
    monkeypatch.setenv("VLLM_BASE_URL", "http://gpu-box:8001/v1")
    monkeypatch.setenv("VLLM_API_KEY", "served-key")
    a = VLLMAdapter(model="m", system_prompt="s")
    assert a._base_url == "http://gpu-box:8001/v1"
    assert a._api_key == "served-key"


def test_explicit_args_beat_env_and_default(monkeypatch):
    monkeypatch.setenv("OPENROUTER_BASE_URL", "https://env/v1")
    monkeypatch.setenv("OPENROUTER_API_KEY", "env-key")
    a = OpenRouterAdapter(
        model="m", system_prompt="s",
        base_url="https://explicit/v1", api_key="explicit-key")
    assert a._base_url == "https://explicit/v1"
    assert a._api_key == "explicit-key"


# --------------------------------------------------------------------------
# Wiring: _build_agent points ChatOpenAI at the resolved endpoint + key
# --------------------------------------------------------------------------

def test_build_agent_passes_endpoint_and_key(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-secret")
    a = OpenRouterAdapter(model="anthropic/claude-3.5-sonnet", system_prompt="s")
    with patch("langchain_openai.ChatOpenAI") as MockLLM, \
            patch("langgraph.prebuilt.create_react_agent") as mock_cra:
        a._build_agent()
        MockLLM.assert_called_once()
        kw = MockLLM.call_args.kwargs
        assert kw["model"] == "anthropic/claude-3.5-sonnet"
        assert kw["base_url"] == "https://openrouter.ai/api/v1"
        assert kw["api_key"] == "or-secret"
        mock_cra.assert_called_once()


# --------------------------------------------------------------------------
# Shared machinery works through the subclass (mock the agent — no network)
# --------------------------------------------------------------------------

def test_invoke_returns_content_and_populates_usage():
    from langchain_core.messages import AIMessage

    msg = AIMessage(content="the answer")
    msg.usage_metadata = {"input_tokens": 11, "output_tokens": 7}
    fake_agent = MagicMock()
    fake_agent.invoke.return_value = {"messages": [msg]}

    a = VLLMAdapter(model="m", system_prompt="s")
    a._agent = fake_agent  # inject — no real server call
    out = a.invoke([{"role": "user", "content": "go"}])

    assert out == "the answer"
    assert a.last_usage["input_tokens"] == 11
    assert a.last_usage["output_tokens"] == 7
    assert a.last_usage["total_cost_usd"] is None


def test_copy_returns_self():
    a = VLLMAdapter(model="m", system_prompt="s")
    assert a.copy() is a


def test_select_native_tools_excludes_closures():
    tools = ["Bash", "Read", "Done", "FollowUp", "WriteNote", "Write"]
    native = OpenRouterAdapter.select_native_tools(tools)
    assert native == ["Bash", "Read", "Write"]  # closures dropped


# ---------------------------------------------------------------------------
# A request with no user turn is refused here, not by the provider
#
# Ollama answers a user-less payload with an opaque 500 ("no user query found
# in messages") that names nothing and reads as intermittent. It is reachable:
# nodes.parsing._to_adapter_messages and _to_lc_messages BOTH keep only
# Human/AI messages and silently discard every other role, so a history of
# system/tool messages filters to an empty list, and thread_id is fresh per
# invoke so nothing server-side backfills the turn.
#
# Whether a real run reaches that shape is still unestablished — these pin the
# guard that will say so attributably when it next happens.
# ---------------------------------------------------------------------------

def test_a_userless_payload_is_refused_with_a_diagnosable_error():
    from a3dasm._src.backends.openai_compatible import UserlessPayloadError

    a = VLLMAdapter(model="m", system_prompt="s")

    class _NeverCalled:
        def invoke(self, *args, **kwargs):
            raise AssertionError("must not reach the provider")

    a._agent = _NeverCalled()
    with pytest.raises(UserlessPayloadError) as exc:
        a.invoke([{"role": "system", "content": "rules"},
                  {"role": "tool", "content": "output"}])

    msg = str(exc.value)
    # The error must carry the evidence: what arrived, and what survived.
    assert "system" in msg and "tool" in msg
    assert "2 message(s) in" in msg
    assert "0 survived" in msg


def test_the_userless_guard_is_not_retried_as_transient():
    """Retrying an unanswerable payload five times with backoff burns wall
    budget and buries the cause deeper."""
    from a3dasm._src.backends.base import is_transient_error
    from a3dasm._src.backends.openai_compatible import UserlessPayloadError

    assert is_transient_error(UserlessPayloadError("no user turn")) is False


def test_a_normal_payload_still_passes_the_guard():
    """The guard must not fire on the ordinary case."""
    a = VLLMAdapter(model="m", system_prompt="s")

    class _Fake:
        def invoke(self, state, config=None):
            from langchain_core.messages import AIMessage
            return {"messages": [AIMessage(content="fine")]}

    a._agent = _Fake()
    assert a.invoke([{"role": "user", "content": "go"}]) == "fine"


def test_both_converters_drop_unrecognised_roles():
    """Pins the shape that makes the guard necessary, in both stages. This is
    the defect the guard instruments; fixing it means preserving these roles,
    which changes what EVERY backend sees and is a separate change."""
    from langchain_core.messages import SystemMessage, ToolMessage

    from a3dasm._src.backends.openai_compatible import _to_lc_messages
    from a3dasm._src.nodes.parsing import _to_adapter_messages

    state = [SystemMessage(content="rules"),
             ToolMessage(content="out", tool_call_id="t1")]
    assert _to_adapter_messages(state) == []
    assert _to_lc_messages([{"role": "system", "content": "x"},
                            {"role": "tool", "content": "y"}]) == []
