"""OpenRouter-backed adapter for the f3dasm LangGraph agentic runtime.

OpenRouter exposes an OpenAI-compatible endpoint that proxies many hosted
models (OpenAI, Anthropic, Llama, …), so this is a thin subclass of
``OpenAICompatibleAdapter``. It declares OpenRouter's endpoint default and the
``OPENROUTER_API_KEY`` / ``OPENROUTER_BASE_URL`` env conventions; all
invoke/tool/usage machinery is inherited and shared with the Ollama and vLLM
backends.

Auth: OpenRouter requires a real API key (``OPENROUTER_API_KEY``). None is
baked in — if no key is configured the underlying client raises at call time,
which is the correct, explicit failure.

The model string is a config decision (e.g. ``"anthropic/claude-3.5-sonnet"``,
``"meta-llama/llama-3.1-70b-instruct"``); there is no baked default.
"""

from __future__ import annotations

from .openai_compatible import OpenAICompatibleAdapter

__all__ = ["OpenRouterAdapter"]


class OpenRouterAdapter(OpenAICompatibleAdapter):
    """Adapter for models served via OpenRouter's OpenAI-compatible API."""

    DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
    BASE_URL_ENV = "OPENROUTER_BASE_URL"
    API_KEY = None  # no default — a real OPENROUTER_API_KEY is required
    API_KEY_ENV = "OPENROUTER_API_KEY"
