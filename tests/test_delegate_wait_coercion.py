"""Regression: Delegate(wait=...) must coerce MCP string-in booleans.

MCP string-in tools pass booleans as strings: wait="false" arrived at the
Delegate closure and `if wait:` treated the non-empty string as truthy, so
every delegation blocked synchronously — serializing the whole run (no two
workers alive at once → Confer dead; summed wall-clock → watchdog kill).
"""
from __future__ import annotations

from a3dasm._src.backends.base import Agent, Edge, Graph
from a3dasm._src.nodes import StrategizerNode


class _Stub:
    def __init__(self) -> None:
        self.closure_tools: dict = {}
        self.last_usage: dict = {}
        self.model = "m"

    def invoke(self, messages):
        return ""

    def copy(self):
        s = self.__class__.__new__(self.__class__)
        _Stub.__init__(s)
        s.closure_tools = dict(self.closure_tools)
        return s


def _node():
    class A(Agent):
        role = "strategizer"
        tools = frozenset({"Done", "Delegate"})
        description = "strategizer"

    class B(Agent):
        role = "implementer"
        description = "implementer"

    spec = Graph(
        nodes={"strategizer": A(), "implementer": B()},
        edges=(Edge("strategizer", "implementer"),),
        entry="strategizer",
    )
    return StrategizerNode(
        _Stub(), name="strategizer", outgoing=["implementer"], spec=spec,
        worker_adapters={"implementer": _Stub()},
    )


def _delegate(n, wait):
    return n.adapter.closure_tools["Delegate"](
        target="implementer", intent="t", expected_report="r",
        hypothesis_ids=["H1"], wait=wait,
    )


def test_wait_string_false_is_async():
    # The bug: "false" (string) is truthy → blocked. Coerced, it must NOT block.
    out = _delegate(_node(), "false")
    assert "Delegation started" in out, out


def test_wait_string_true_blocks():
    # Positive control: "true" still blocks (returns the report body, not the
    # async "Delegation started" handle).
    out = _delegate(_node(), "true")
    assert "Delegation started" not in out, out


def test_wait_real_bool_false_is_async():
    out = _delegate(_node(), False)
    assert "Delegation started" in out, out
