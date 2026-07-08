"""Backend registry — the single source of truth mapping a backend name to its
adapter class.

Both the runtime dispatch (``AgenticRun._make_adapter``) and the interface
parity tests (tests/agentic/test_backend_parity.py) iterate this registry, so
registering a new backend here makes it BOTH dispatchable AND automatically
held to interface parity — there is no second place to update.

Imports are lazy (by dotted path) so a heavy backend SDK is only imported when
that backend is actually selected.
"""

from __future__ import annotations

import importlib

# backend name -> (module path, adapter class name)
_BACKENDS: dict[str, tuple[str, str]] = {
    "claude": ("a3dasm._src.backends.claude", "ClaudeAdapter"),
    "ollama": ("a3dasm._src.backends.ollama", "OllamaAdapter"),
    "openrouter": (
        "a3dasm._src.backends.openrouter", "OpenRouterAdapter"),
    "vllm": ("a3dasm._src.backends.vllm", "VLLMAdapter"),
}


def available_backends() -> list[str]:
    """Sorted list of registered backend names."""
    return sorted(_BACKENDS)


def get_adapter_class(name: str):
    """Return the adapter class registered under ``name`` (lazy import).

    Raises ValueError on an unknown backend, listing the valid names.
    """
    try:
        module_path, class_name = _BACKENDS[name]
    except KeyError:
        raise ValueError(
            f"unknown backend {name!r}; available: {available_backends()}"
        ) from None
    module = importlib.import_module(module_path)
    return getattr(module, class_name)
