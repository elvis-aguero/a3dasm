"""Tripwire: every tool an agent declares MUST be documented in
docs/agentic/FEATURES.md. Adding a tool to an agent's toolset without a catalog
entry fails the build — so the feature catalog can't silently drift behind the code.

Limit (honest): this enforces the statically-declared `tools` surface. Tools
injected dynamically (e.g. Delegate, Hypothesis*) and non-tool infrastructure
features rely on the written contract at the top of FEATURES.md, not this test.
"""
from __future__ import annotations

import importlib
import inspect
import pkgutil
from pathlib import Path

import a3dasm._src.agents as agents_pkg

_FEATURES = (
    Path(__file__).resolve().parents[1]
    / "internal" / "FEATURES.md"
)


def _declared_tools() -> set[str]:
    tools: set[str] = set()
    for m in pkgutil.iter_modules(agents_pkg.__path__):
        mod = importlib.import_module(f"a3dasm._src.agents.{m.name}")
        for _, obj in inspect.getmembers(mod, inspect.isclass):
            t = getattr(obj, "tools", None)
            if t and isinstance(t, (set, frozenset)):
                tools |= {str(x) for x in t}
    return tools


def test_features_md_exists():
    assert _FEATURES.is_file(), f"missing feature catalog: {_FEATURES}"


def test_every_declared_tool_is_documented():
    catalog = _FEATURES.read_text(encoding="utf-8")
    declared = _declared_tools()
    assert declared, "no agent tools found — enumeration broke"
    missing = sorted(t for t in declared if t not in catalog)
    assert not missing, (
        "Tools declared by an agent but ABSENT from docs/agentic/FEATURES.md: "
        f"{missing}. Add each to the catalog (contract at the top of FEATURES.md)."
    )
