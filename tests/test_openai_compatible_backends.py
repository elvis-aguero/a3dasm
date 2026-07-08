"""Unit tests for the OpenAI-compatible backends (OpenRouter, vLLM).

Fully mocked — no API keys, no network. They verify the per-backend endpoint
and auth conventions (explicit arg > env > class default) and that the shared
invoke/usage machinery works through the subclass. Cross-backend interface
parity is enforced separately in test_backend_parity.py.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

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
