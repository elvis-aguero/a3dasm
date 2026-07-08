"""Ollama-backed adapter for the f3dasm LangGraph agentic runtime.

Ollama exposes an OpenAI-compatible endpoint, so this is a thin subclass of
``OpenAICompatibleAdapter`` (see backends/openai_compatible.py) that only
declares Ollama's local endpoint default + env override. All invoke/tool/usage
machinery is inherited and shared with the OpenRouter and vLLM backends.

The module re-exports the shared tool-builder helpers
(``_build_arxiv_closures``, ``_make_literature_tools``, ``_native_tool_map``,
``_to_lc_messages``, the ``_make_*_tool`` factories) so existing
``from ...backends.ollama import _build_arxiv_closures`` style imports keep
working.
"""

from __future__ import annotations

from .openai_compatible import (
    OpenAICompatibleAdapter,
    _build_arxiv_closures,
    _make_bash_tool,
    _make_edit_tool,
    _make_glob_tool,
    _make_grep_tool,
    _make_literature_tools,
    _make_read_tool,
    _make_write_tool,
    _native_tool_map,
    _to_lc_messages,
)

__all__ = [
    "OllamaAdapter",
    "_build_arxiv_closures",
    "_make_bash_tool",
    "_make_edit_tool",
    "_make_glob_tool",
    "_make_grep_tool",
    "_make_literature_tools",
    "_make_read_tool",
    "_make_write_tool",
    "_native_tool_map",
    "_to_lc_messages",
]


class OllamaAdapter(OpenAICompatibleAdapter):
    """Adapter for Ollama-served open-weight models.

    Uses ChatOpenAI pointed at Ollama's local OpenAI-compatible endpoint
    (default ``http://localhost:11434/v1``, overridable via ``OLLAMA_BASE_URL``
    or an explicit ``base_url``). Ollama needs no real auth, so the API key is
    the conventional placeholder ``"local"``.
    """

    DEFAULT_BASE_URL = "http://localhost:11434/v1"
    BASE_URL_ENV = "OLLAMA_BASE_URL"
    API_KEY = "local"
    API_KEY_ENV = None
