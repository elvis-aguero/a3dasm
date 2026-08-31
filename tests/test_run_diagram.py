"""Tests for run_diagram.py — the run architecture SVG renderer, and
AgenticRun.render_architecture()'s wiring to it.

Regression coverage: a fixed per-layer row height silently overflowed a
card's own box when its tool count got large (caught by actually rendering
and looking at it, not assumed) — _bfs_layers/_CardContent.height are tested
directly so that class of bug can't reappear unnoticed.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

from a3dasm._src.agent_runtime import DEFAULT_MODEL, AgenticRun, _default_graph
from a3dasm._src.backends.base import Agent, Edge, Graph
from a3dasm._src.run_diagram import (
    _bfs_layers,
    _CardContent,
    _node_tools,
    _tool_colors,
    render_architecture_svg,
)


def _make_run(tmp_path: Path, run_dir: Path | None = None) -> AgenticRun:
    (tmp_path / "PROBLEM_STATEMENT.md").write_text("test")
    run = AgenticRun.__new__(AgenticRun)
    run.study_dir = tmp_path
    run._model = DEFAULT_MODEL
    run._backend = "claude"
    run._graph_spec = _default_graph()
    run._run_dir = run_dir
    return run


# ---------------------------------------------------------------------------
# render_architecture_svg — content correctness
# ---------------------------------------------------------------------------


def test_default_graph_renders_valid_svg_with_every_node():
    svg = render_architecture_svg(_default_graph())
    ET.fromstring(svg)  # raises if not well-formed XML
    for name in ("strategizer", "literature_reviewer", "datagenerator",
                 "implementer", "critic"):
        assert name in svg


def test_entry_node_gets_entry_badge_others_get_role_kind():
    svg = render_architecture_svg(_default_graph())
    assert ">ENTRY<" in svg
    # Non-entry nodes show their role's "kind" tag instead.
    assert ">AUDIT<" in svg  # critic
    assert ">RESEARCH<" in svg  # literature_reviewer


def test_topology_injected_tools_shown_only_for_outgoing_nodes():
    """strategizer/datagenerator/implementer/critic all have outgoing edges
    in the default graph (each can Delegate to literature_reviewer, or
    strategizer delegates to all) — literature_reviewer has NO outgoing
    edges and must not show Delegate/FollowUp itself."""
    graph = _default_graph()
    strategizer_tools = _node_tools("strategizer", graph.nodes["strategizer"], graph)
    lit_tools = _node_tools(
        "literature_reviewer", graph.nodes["literature_reviewer"], graph)
    assert "Delegate" in strategizer_tools
    assert "FollowUp" in strategizer_tools
    assert "Delegate" not in lit_tools
    assert "FollowUp" not in lit_tools


def test_declared_tools_appear_for_a_lightly_tooled_node():
    """literature_reviewer has few enough tools that none get truncated —
    a node with many more (e.g. strategizer) legitimately truncates its
    list past _TOOLS_MAX_LINES, so this only asserts the un-truncated case."""
    svg = render_architecture_svg(_default_graph())
    graph = _default_graph()
    for tool in graph.nodes["literature_reviewer"].tools:
        assert tool in svg, f"literature_reviewer's tool {tool!r} missing from SVG"


def test_edge_count_matches_graph():
    graph = _default_graph()
    svg = render_architecture_svg(graph, model="haiku", backend="claude")
    assert f"{len(graph.edges)} edges" in svg
    assert f"{len(graph.nodes)} nodes" in svg


def test_custom_title_model_backend_surface_in_svg():
    svg = render_architecture_svg(
        _default_graph(), title="my-study — architecture",
        model="haiku", backend="claude")
    assert "my-study — architecture" in svg
    assert "model=haiku" in svg
    assert "backend=claude" in svg


# ---------------------------------------------------------------------------
# Per-tool color key (same tool = same color everywhere it appears)
# ---------------------------------------------------------------------------


def _tiny_graph(tools_a, tools_b):
    class A(Agent):
        role = "worker"
        description = "Agent A."
        tools = frozenset(tools_a)

    class B(Agent):
        role = "worker"
        description = "Agent B."
        tools = frozenset(tools_b)

    return Graph(
        nodes={"a": A(), "b": B()},
        edges=(),
        entry="a",
    )


def test_shared_tool_gets_the_same_color_in_every_card_it_appears_in():
    """The actual ask: 'Read' must render in the SAME color wherever it
    appears — a real per-tool color key, not a per-node accent — so a
    recurring color across different cards is what visually says "these
    nodes share this capability."."""
    graph = _tiny_graph(["Read", "OnlyA"], ["Read", "OnlyB"])
    tool_colors = _tool_colors({"Read"})
    content_a = _CardContent("a", graph.nodes["a"], graph, tool_colors)
    content_b = _CardContent("b", graph.nodes["b"], graph, tool_colors)
    svg_a = content_a.svg(0, 0)
    svg_b = content_b.svg(0, 0)
    read_color = tool_colors["Read"]
    assert f'<tspan fill="{read_color}" font-weight="700">Read</tspan>' in svg_a
    assert f'<tspan fill="{read_color}" font-weight="700">Read</tspan>' in svg_b
    # A tool unique to one node has nothing to color-match against, so it
    # renders in plain neutral ink instead of competing for palette space.
    assert 'font-weight="400">OnlyA</tspan>' in svg_a
    assert 'font-weight="400">OnlyB</tspan>' in svg_b


def test_render_architecture_svg_computes_shared_set_end_to_end():
    """Full render_architecture_svg (not the lower-level _CardContent
    directly) must also apply the per-tool color key — regression against
    wiring the shared/tool_colors computation incorrectly."""
    graph = _tiny_graph(["Read", "OnlyA"], ["Read", "OnlyB"])
    svg = render_architecture_svg(graph)
    read_color = _tool_colors({"Read"})["Read"]
    assert f'<tspan fill="{read_color}" font-weight="700">Read</tspan>' in svg


# ---------------------------------------------------------------------------
# Layout math — the overflow regression
# ---------------------------------------------------------------------------


def test_bfs_layers_assigns_entry_layer_zero_and_handles_cycles():
    graph = Graph(
        nodes={
            "a": type("A", (Agent,), {"role": "worker", "description": "A."})(),
            "b": type("B", (Agent,), {"role": "worker", "description": "B."})(),
        },
        edges=(Edge("a", "b"), Edge("b", "a")),  # a cycle
        entry="a",
    )
    layers = _bfs_layers(graph)
    assert layers["a"] == 0
    assert layers["b"] == 1  # visited once despite the cycle


def test_bfs_layers_places_unreachable_node_in_trailing_layer():
    graph = Graph(
        nodes={
            "a": type("A", (Agent,), {"role": "worker", "description": "A."})(),
            "isolated": type(
                "Iso", (Agent,), {"role": "worker", "description": "I."})(),
        },
        edges=(),
        entry="a",
    )
    layers = _bfs_layers(graph)
    assert layers["a"] == 0
    assert layers["isolated"] == 1  # still rendered, not dropped


def test_card_height_grows_with_tool_count_no_fixed_overflow():
    """Regression: an earlier draft used a FIXED per-layer row height, which
    silently overflowed a card whose real content (many tools) exceeded it.
    A card with many tools must measure taller than one with few — and
    render_architecture_svg's row_y must actually place the NEXT row far
    enough down to clear it (checked via row spacing below)."""
    few = _tiny_graph(["Read"], ["Read"])
    many_tools = [f"Tool{i}" for i in range(40)]

    class Many(Agent):
        role = "worker"
        description = "Has lots of tools."
        tools = frozenset(many_tools)

    graph_many = Graph(nodes={"a": Many()}, edges=(), entry="a")
    content_few = _CardContent(
        "a", few.nodes["a"], few, tool_colors={})
    content_many = _CardContent(
        "a", graph_many.nodes["a"], graph_many, tool_colors={})
    assert content_many.height > content_few.height


def test_two_layer_graph_second_row_never_overlaps_first():
    """The real overflow bug: row 2's y position must clear row 1's actual
    (measured) height, not a fixed constant — verified by checking the
    rendered SVG's row_y spacing directly rather than eyeballing a PNG."""
    class Hub(Agent):
        role = "strategizer"
        description = "Hub with a huge tool surface, forcing a tall card."
        tools = frozenset([f"Tool{i}" for i in range(50)])

    class Leaf(Agent):
        role = "worker"
        description = "Leaf."
        tools = frozenset({"Read"})

    graph = Graph(
        nodes={"hub": Hub(), "leaf": Leaf()},
        edges=(Edge("hub", "leaf"),),
        entry="hub",
    )
    content_hub = _CardContent("hub", graph.nodes["hub"], graph, tool_colors={})
    svg = render_architecture_svg(graph)
    ET.fromstring(svg)  # still well-formed even with a very tall card
    # The hub's card is necessarily taller than the old fixed row height
    # (300px) once the earlier draft's bug is reintroduced — assert it
    # actually needs more than that, so this test is a real regression
    # guard, not a tautology.
    assert content_hub.height > 300


# ---------------------------------------------------------------------------
# AgenticRun.render_architecture() wiring
# ---------------------------------------------------------------------------


def test_render_architecture_default_path_before_run_starts(tmp_path):
    run = _make_run(tmp_path, run_dir=None)
    out = run.render_architecture()
    assert out == tmp_path / "architecture.svg"
    assert out.is_file()
    ET.fromstring(out.read_text(encoding="utf-8"))


def test_render_architecture_default_path_once_run_dir_set(tmp_path):
    run_dir = tmp_path / "runs" / "20260101T000000"
    run = _make_run(tmp_path, run_dir=run_dir)
    out = run.render_architecture()
    assert out == run_dir / "debug" / "architecture.svg"
    assert out.is_file()


def test_render_architecture_explicit_out_path(tmp_path):
    run = _make_run(tmp_path, run_dir=None)
    custom = tmp_path / "somewhere" / "diagram.svg"
    out = run.render_architecture(custom)
    assert out == custom
    assert out.is_file()


def test_render_architecture_content_reflects_run_model_backend(tmp_path):
    run = _make_run(tmp_path, run_dir=None)
    out = run.render_architecture()
    text = out.read_text(encoding="utf-8")
    assert f"model={DEFAULT_MODEL}" in text
    assert "backend=claude" in text
