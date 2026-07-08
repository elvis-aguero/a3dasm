"""Forward-compatible tool catalog (Spec B).

The <tools> prompt section is GENERATED from the live closure set, so it can't
drift from the actual tools and a new tool self-registers. The tool owns its
guidance (docstring + optional @tool_examples).
"""
from __future__ import annotations

from a3dasm._src.tool_catalog import (
    render_tool_catalog,
    system_prompt_with_catalog,
    tool_examples,
)


def test_catalog_includes_every_live_tool_by_exact_name():
    def Alpha():
        """Do the alpha thing."""
    def Beta():
        """Do the beta thing."""
    cat = render_tool_catalog({"Alpha": Alpha, "Beta": Beta})
    assert "### Alpha" in cat and "Do the alpha thing." in cat
    assert "### Beta" in cat and "Do the beta thing." in cat


def test_new_tool_self_registers_forward_compatible():
    """Adding a closure makes it appear with no other change — the whole point."""
    def Existing():
        """existing"""
    base = render_tool_catalog({"Existing": Existing})
    assert "### NewlyAdded" not in base

    def NewlyAdded():
        """brand new"""
    grown = render_tool_catalog({"Existing": Existing, "NewlyAdded": NewlyAdded})
    assert "### NewlyAdded" in grown and "brand new" in grown


def test_examples_render_when_present():
    @tool_examples("Foo(1, 2)", "Foo('x')")
    def Foo():
        """foo desc"""
    cat = render_tool_catalog({"Foo": Foo})
    assert "Examples:" in cat
    assert "Foo(1, 2)" in cat and "Foo('x')" in cat


def test_missing_docstring_marked_not_crashed():
    def NoDoc():
        pass
    cat = render_tool_catalog({"NoDoc": NoDoc})
    assert "### NoDoc" in cat and "(no description)" in cat


def test_deterministic_sorted_order():
    def B():
        """b"""
    def A():
        """a"""
    cat = render_tool_catalog({"B": B, "A": A})
    assert cat.index("### A") < cat.index("### B")


def test_empty_closures_render_nothing():
    assert render_tool_catalog({}) == ""


def test_system_prompt_with_catalog_appends():
    def T():
        """t"""
    out = system_prompt_with_catalog("BASE PROMPT", {"T": T})
    assert out.startswith("BASE PROMPT")
    assert "<tools>" in out and "### T" in out


def test_catalog_fixes_observed_drift_on_a_real_strategizer():
    """Regression for the drift the audit found: CancelDelegation and ReadNote
    are live closures that were MISSING from the hand-written prompt prose, and
    the prose said 'Read' not 'ReadNote'. The generated catalog uses the live
    closure keys, so they're present and correctly named."""
    from a3dasm._src.backends.base import Agent, Edge, Graph
    from a3dasm._src.nodes import StrategizerNode

    class _Stub:
        def __init__(self):
            self.closure_tools: dict = {}
            self.last_usage: dict = {}
            self.model = "m"

        def invoke(self, messages):
            return ""

    class A(Agent):
        role = "strategizer"
        # Opt into CancelDelegation/ReadNote (now plug-and-play opt-in) so the
        # generated catalog includes them — the no-drift property under test.
        tools = frozenset({"Done", "ReadNote", "CancelDelegation"})
        description = "strategizer"

    class B(Agent):
        role = "implementer"
        description = "implementer"

    spec = Graph(
        nodes={"strategizer": A(), "implementer": B()},
        edges=(Edge("strategizer", "implementer"),), entry="strategizer")
    n = StrategizerNode(
        _Stub(), name="strategizer", outgoing=["implementer"], spec=spec,
        worker_adapters={"implementer": _Stub()})

    cat = render_tool_catalog(n.adapter.closure_tools)
    assert "### CancelDelegation" in cat   # was missing from the old prose
    # the catalog name-set is EXACTLY the live closure key-set — so a tool can
    # never be missing (CancelDelegation) or misnamed ('Read' vs 'ReadNote')
    # in the prompt again, by construction.
    catalog_names = {
        line[4:].strip() for line in cat.splitlines() if line.startswith("### ")}
    assert catalog_names == set(n.adapter.closure_tools)


def test_claude_catalog_names_match_sdk_registration():
    """lit-bug #2 ('No such tool available'): the Claude backend exposes each
    closure to the model ONLY by its MCP-qualified name (which is also what
    allowed_tools carries), so the AUTHORITATIVE <tools> catalog must advertise
    the SAME qualified names — never the bare ones, which the model cannot call.
    One helper feeds both, so they cannot drift."""
    from a3dasm._src.backends.claude import (
        ClaudeAdapter,
        _qualify_closure_names,
    )

    closures = {"arxiv_search_papers": lambda q: q, "CorpusSearch": lambda q: q}
    qualified = _qualify_closure_names(closures)
    assert set(qualified) == {
        "mcp__f3dasm_agent_tools__arxiv_search_papers",
        "mcp__f3dasm_agent_tools__CorpusSearch",
    }

    # The catalog the Claude backend assembles advertises the qualified names...
    catalog = render_tool_catalog(qualified)
    for qn in qualified:
        assert f"### {qn}" in catalog
    # ...and never the bare name as a callable header (the lit-bug #2 trap).
    assert "### arxiv_search_papers" not in catalog
    assert "### CorpusSearch" not in catalog

    # The names in that catalog are EXACTLY the names allowed_tools grants the
    # model — same qualification helper, so registration and prompt can't drift.
    adapter = ClaudeAdapter(
        model="x", system_prompt="s",
        native_tools=["Read", "Bash"], closure_tools=closures,
    )
    allowed = set(adapter._compute_allowed_tools(list(qualified)))
    assert set(qualified) <= allowed
    # native tools stay bare (they are SDK built-ins, not MCP-namespaced)
    assert "Read" in allowed and "Bash" in allowed
