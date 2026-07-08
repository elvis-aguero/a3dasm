"""The backend registry: single source of truth for name -> adapter class.

Both the runtime dispatch and the parity tests iterate it, so these tests pin
that every advertised backend resolves to a class exposing the adapter
contract, and that unknown names fail loudly.
"""
from __future__ import annotations

import pytest

from a3dasm._src.backends import registry


def test_available_backends_includes_all_four():
    names = registry.available_backends()
    assert set(names) >= {"claude", "ollama", "openrouter", "vllm"}
    assert names == sorted(names)  # stable, sorted


@pytest.mark.parametrize("name", ["ollama", "openrouter", "vllm"])
def test_openai_compatible_backends_resolve_to_a_class(name):
    cls = registry.get_adapter_class(name)
    assert isinstance(cls, type)
    # contract the runtime relies on
    assert hasattr(cls, "select_native_tools")
    assert hasattr(cls, "invoke")
    assert hasattr(cls, "copy")


def test_unknown_backend_raises_with_valid_names():
    with pytest.raises(ValueError) as exc:
        registry.get_adapter_class("gpt5-turbo-ultra")
    msg = str(exc.value)
    assert "unknown backend" in msg
    assert "claude" in msg  # lists the valid names
