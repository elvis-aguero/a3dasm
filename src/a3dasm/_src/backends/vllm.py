"""vLLM-backed adapter for the f3dasm LangGraph agentic runtime.

vLLM serves models behind an OpenAI-compatible HTTP server (``vllm serve``),
so this is a thin subclass of ``OpenAICompatibleAdapter``. It declares vLLM's
local-server endpoint default and the ``VLLM_BASE_URL`` / ``VLLM_API_KEY`` env
conventions; all invoke/tool/usage machinery is inherited and shared with the
Ollama and OpenRouter backends.

Auth: a vLLM server usually needs no key, so the API key defaults to the
conventional placeholder ``"EMPTY"`` (overridable via ``VLLM_API_KEY`` when the
server was started with ``--api-key``).

The model string is a config decision (whatever the server was launched with,
e.g. ``"meta-llama/Llama-3.1-8B-Instruct"``); there is no baked default.
"""

from __future__ import annotations

from .openai_compatible import OpenAICompatibleAdapter

__all__ = ["VLLMAdapter"]


class VLLMAdapter(OpenAICompatibleAdapter):
    """Adapter for models served by a vLLM OpenAI-compatible server."""

    DEFAULT_BASE_URL = "http://localhost:8000/v1"
    BASE_URL_ENV = "VLLM_BASE_URL"
    API_KEY = "EMPTY"  # vLLM usually needs no auth; placeholder key
    API_KEY_ENV = "VLLM_API_KEY"
