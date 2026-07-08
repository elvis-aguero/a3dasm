"""Forward-compatible tool catalog.

The prompt's ``<tools>`` section is GENERATED from the live closure set, so it
can never drift from the actual tools an agent has, and a newly-registered tool
self-documents with no hand-edit. The tool owns its guidance: its docstring is
the description (already what the backends feed the model as the tool's schema
description), plus optional usage examples attached via ``@tool_examples``.

Used by both backends (Claude + OpenAI-compatible) at prompt-assembly time, so
parity is automatic.
"""

from __future__ import annotations

from collections.abc import Callable

__all__ = ["tool_examples", "render_tool_catalog"]


def tool_examples(*examples: str):
    """Attach usage examples to a tool closure (the tool owns them).

    ``render_tool_catalog`` renders them under the tool. Closures without this
    decorator simply have no examples — rendering still works.
    """
    def deco(fn: Callable) -> Callable:
        fn._tool_examples = list(examples)
        return fn
    return deco


def render_tool_catalog(closure_tools: dict[str, Callable]) -> str:
    """Render a ``<tools>`` block from the LIVE closure dict.

    Per tool: the exact registered name (the dict key — so a tool can never be
    missing or misnamed in the prompt), its docstring, and its examples. Does
    NOT re-emit parameter types/schema — the backends already carry those to the
    model via the tool-use API. Deterministic (sorted by name) for cache
    stability. Returns "" when there are no closures.
    """
    if not closure_tools:
        return ""
    blocks: list[str] = []
    for name in sorted(closure_tools):
        fn = closure_tools[name]
        doc = (getattr(fn, "__doc__", None) or "").strip() or "(no description)"
        block = f"### {name}\n{doc}"
        examples = getattr(fn, "_tool_examples", None)
        if examples:
            block += "\nExamples:\n" + "\n".join(f"  - {e}" for e in examples)
        blocks.append(block)
    return (
        "\n\n<tools>\n"
        "Your available tools, generated from the live tool set (AUTHORITATIVE "
        "— these exact names are the ones you call; anything not listed here is "
        "not available):\n\n"
        + "\n\n".join(blocks)
        + "\n</tools>"
    )


def system_prompt_with_catalog(base_prompt: str, closure_tools: dict) -> str:
    """Base prompt + the generated tool catalog. Computed at prompt-assembly
    time so it always reflects the current closures; never mutates state."""
    return base_prompt + render_tool_catalog(closure_tools)
