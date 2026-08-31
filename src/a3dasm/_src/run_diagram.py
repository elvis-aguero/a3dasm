"""Hand-laid-out SVG rendering of an AgenticRun's agent graph.

Shows the run's actual anatomy: every node, its role, its description, its
full tool surface (declared + topology-injected), and the delegation edges
between nodes — generated from the live ``Graph``/``Agent`` objects, never
hand-typed, so it cannot go stale the way this repo's previous diagram did
(a hand-authored ``internal/class_diagram.dot``, describing classes that no
longer existed — deleted in favor of this module).

Deliberately NOT run through Graphviz or any other auto-layout engine: a
run's graph is small (a handful of nodes), so a hand-computed layered
layout — BFS distance from the entry node, one row per layer — reads as a
designed diagram rather than a force-directed tool's generic output. Row
height is computed from each node's REAL content (two-pass: measure every
card, then lay out cumulative y from the actual heights) — a fixed
per-layer height silently overflowed the entry node's card in an earlier
draft once its tool count got large; caught by actually rendering and
looking at it, not assumed. SVG is the native, checked-in format: it is
resolution-independent by construction, so any DPI/pixel size (a
10000x10000 PNG included) is one rasterization step away with zero quality
loss — this module does not attempt to also ship a PNG rasterizer.
"""
from __future__ import annotations

from collections import Counter, deque
from datetime import datetime, timezone
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .backends.base import Graph

__all__ = ["render_architecture_svg"]

# ---------------------------------------------------------------------------
# Design tokens — a deliberate palette/type system, not defaults.
#   Background: warm paper, not stark white.
#   Accent: the amber already used by docs/mkdocs.yml's palette, so a
#   diagram that ends up embedded in the docs site doesn't clash with it.
#   Per-role hues: restrained, desaturated family (not a rainbow) so roles
#   read as siblings, with the entry node's accent picked out in amber.
# ---------------------------------------------------------------------------
_BG = "#FBF7EE"
_INK = "#231F1B"
_INK_SOFT = "#5B564C"
_EDGE_PRIMARY = "#B4690E"
_EDGE_LATERAL = "#9C9280"
_AMBER = "#B4690E"
_AMBER_SOFT = "#F3E3CC"

_ROLE_COLORS = {
    "strategizer": (_AMBER, _AMBER_SOFT),
    "literature_reviewer": ("#3E6E63", "#DCE9E4"),
    "datagenerator": ("#4A6B8A", "#DDE5EE"),
    "implementer": ("#7A5A9E", "#E7DEF0"),
    "critic": ("#9E4A3E", "#F1DED9"),
    "debugger": ("#8A7A3E", "#EFE9D3"),
}
_ROLE_DEFAULT = ("#6B6558", "#E7E3D8")

_ROLE_KIND = {
    "strategizer": "ORCHESTRATOR",
    "literature_reviewer": "RESEARCH",
    "datagenerator": "EXECUTION",
    "implementer": "EXECUTION",
    "critic": "AUDIT",
    "debugger": "DIAGNOSIS",
}
_ROLE_KIND_DEFAULT = "WORKER"

_DISPLAY_FONT = (
    "'Iowan Old Style', Palatino, 'Palatino Linotype', Georgia, serif"
)
_MONO_FONT = (
    "'SF Mono', 'Menlo', 'Consolas', 'Liberation Mono', monospace"
)
_BODY_FONT = (
    "-apple-system, 'Helvetica Neue', Helvetica, Arial, sans-serif"
)

# Topology-injected tools: see backends/base.py's Agent docstring + the
# runtime's actual routing.py wiring. Only a node with outgoing edges gets
# these (routing.py's _build_routing_closures is only invoked for
# orchestrating nodes) — NOT every name the Agent docstring lists (its
# "Parallel"/"Debate"/"Retry"/"Ask" are aspirational; only Delegate/Wait/
# Reply/FollowUp/RecallHistory are actually wired up as of this writing).
_TOPOLOGY_TOOLS_IF_OUTGOING = (
    "Delegate", "Wait", "Reply", "FollowUp", "RecallHistory",
)

_CARD_W = 400
_COL_GAP = 120          # generous — the earlier draft's arrows were
                        # barely visible with cards packed close together
_ROW_GAP = 170          # dedicated space between rows for lateral edges
_MARGIN = 80
_HEADER_H = 130
_FOOTER_H = 60

_LINE_H = 20
_DESC_LINE_H = 17
_TOOLS_MAX_LINES = 8

# Tool-name color coding: a real per-tool KEY, not a per-node accent — the
# SAME tool name renders in the SAME color everywhere it appears, so a
# recurring color across different cards is what visually says "these nodes
# share this capability." Only worth doing for tools shared by >1 node
# (nothing to visually match for a tool that appears exactly once); those
# render in plain neutral ink instead, so they recede rather than compete
# with the palette. Colors chosen to stay distinguishable from the
# per-role header accents (_ROLE_COLORS) so the two coding schemes never
# get confused for one another.
_TOOL_PALETTE = [
    "#2E6E8E", "#8E5B2E", "#4E7A3E", "#9E3E7A", "#B08A2E",
    "#3E7A94", "#8E3E4E", "#5B5B94", "#6E7A2E", "#3E5B8E",
]
_TOOL_DISTINCTIVE_COLOR = "#6B6558"


def _tool_colors(shared_tools: set[str]) -> dict[str, str]:
    """Deterministic tool-name -> color assignment, stable across a render
    (sorted first, so the SAME graph always assigns the SAME colors)."""
    return {
        name: _TOOL_PALETTE[i % len(_TOOL_PALETTE)]
        for i, name in enumerate(sorted(shared_tools))
    }


def _wrap(text: str, width: int) -> list[str]:
    """Greedy word-wrap, no external text-shaping — good enough for the
    monospace/near-monospace character widths this diagram uses."""
    words = (text or "").split()
    lines: list[str] = []
    cur = ""
    for w in words:
        cand = f"{cur} {w}".strip()
        if len(cand) > width and cur:
            lines.append(cur)
            cur = w
        else:
            cur = cand
    if cur:
        lines.append(cur)
    return lines


def _wrap_tokens(tokens: list[str], width: int) -> list[list[str]]:
    """Like _wrap, but keeps token identity (needed to color each tool name
    independently instead of gluing them into one opaque string)."""
    lines: list[list[str]] = []
    cur: list[str] = []
    cur_len = 0
    for t in tokens:
        add_len = len(t) + (2 if cur else 0)  # ", " separator
        if cur and cur_len + add_len > width:
            lines.append(cur)
            cur, cur_len = [t], len(t)
        else:
            cur.append(t)
            cur_len += add_len
    if cur:
        lines.append(cur)
    return lines


def _bfs_layers(graph: Graph) -> dict[str, int]:
    """Distance from entry, in hops. Handles cycles (Graph permits loops)
    by visiting each node once; a node unreachable from entry (a disconnected
    custom graph) lands in its own trailing layer so it still renders."""
    layers: dict[str, int] = {graph.entry: 0}
    q = deque([graph.entry])
    adj: dict[str, list[str]] = {n: [] for n in graph.nodes}
    for e in graph.edges:
        adj[e.source].append(e.target)
    while q:
        cur = q.popleft()
        for nxt in adj.get(cur, ()):
            if nxt not in layers:
                layers[nxt] = layers[cur] + 1
                q.append(nxt)
    stray_layer = (max(layers.values(), default=-1)) + 1
    for n in graph.nodes:
        if n not in layers:
            layers[n] = stray_layer
    return layers


def _topology_tools(graph: Graph, name: str) -> list[str]:
    has_outgoing = any(e.source == name for e in graph.edges)
    return list(_TOPOLOGY_TOOLS_IF_OUTGOING) if has_outgoing else []


def _node_tools(name: str, agent, graph: Graph) -> list[str]:
    """Declared tools (minus any that are actually topology names, defensively)
    plus this node's real topology-injected tools — see the module docstring
    on why the Agent docstring's full topology list isn't trusted blindly."""
    declared = sorted(set(agent.tools) - set(_TOPOLOGY_TOOLS_IF_OUTGOING))
    return _topology_tools(graph, name) + declared


def _esc(s: str) -> str:
    return (
        (s or "").replace("&", "&amp;").replace("<", "&lt;")
        .replace(">", "&gt;").replace('"', "&quot;")
    )


class _CardContent:
    """Precomputed, measured content for one node — built once, used both
    to compute row heights (pass 1) and to draw (pass 2), so the two never
    disagree about how tall a card is."""

    def __init__(
        self, name: str, agent, graph: Graph, tool_colors: dict[str, str],
    ) -> None:
        self.name = name
        self.accent, self.accent_soft = _ROLE_COLORS.get(
            agent.role, _ROLE_DEFAULT)
        self.kind = _ROLE_KIND.get(agent.role, _ROLE_KIND_DEFAULT)
        self.is_entry = name == graph.entry
        self.desc_lines = _wrap(agent.description or "(no description)", 52)[:4]

        self.all_tools = _node_tools(name, agent, graph)
        self.tool_colors = tool_colors
        wrapped = _wrap_tokens(
            self.all_tools if self.all_tools else ["(none)"], 50)
        self.truncated = len(wrapped) > _TOOLS_MAX_LINES
        self.tool_lines = wrapped[:_TOOLS_MAX_LINES]

        body_lines = 1 + len(self.desc_lines) + 1 + len(self.tool_lines) + (
            1 if self.truncated else 0)
        self.height = (
            58 + len(self.desc_lines) * _DESC_LINE_H
            + (body_lines - len(self.desc_lines)) * _LINE_H + 24)

    def svg(self, x: float, y: float) -> str:
        h = self.height
        parts = [
            f'<g transform="translate({x},{y})">',
            # Flat, unfiltered shadow: a blurred (feDropShadow) shadow risks
            # a renderer that rasterizes the WHOLE filtered group (blurring
            # the text along with it — caught by actually rendering a draft
            # and looking at it, not assumed safe). A solid offset rect
            # behind the card is filter-free, so it looks identical in every
            # viewer/rasterizer, which matters given this must stay crisp at
            # very high zoom/DPI.
            f'<rect x="4" y="6" width="{_CARD_W}" height="{h}" rx="14" '
            f'fill="{_INK}" opacity="0.10"/>',
            f'<rect width="{_CARD_W}" height="{h}" rx="14" '
            f'fill="{_BG}" stroke="{self.accent}" '
            f'stroke-width="{2.4 if self.is_entry else 1.4}"/>',
            f'<rect width="{_CARD_W}" height="46" rx="14" fill="{self.accent_soft}"/>',
            f'<rect y="32" width="{_CARD_W}" height="14" fill="{self.accent_soft}"/>',
            f'<text x="20" y="31" font-family="{_DISPLAY_FONT}" '
            f'font-size="23" font-weight="700" fill="{_INK}">'
            f'{_esc(self.name)}</text>',
        ]
        badge = "ENTRY" if self.is_entry else self.kind
        parts.append(
            f'<text x="{_CARD_W - 18}" y="30" text-anchor="end" '
            f'font-family="{_MONO_FONT}" font-size="10.5" '
            f'font-weight="700" letter-spacing="1.4" fill="{self.accent}">'
            f'{badge}</text>'
        )
        ty = 46 + 24
        for line in self.desc_lines:
            parts.append(
                f'<text x="20" y="{ty}" font-family="{_BODY_FONT}" '
                f'font-size="11.5" fill="{_INK_SOFT}">{_esc(line)}</text>'
            )
            ty += _DESC_LINE_H
        ty += 8
        parts.append(
            f'<text x="20" y="{ty}" font-family="{_MONO_FONT}" font-size="11" '
            f'font-weight="700" letter-spacing="1" fill="{self.accent}">'
            f'TOOLS ({len(self.all_tools)})</text>'
        )
        ty += _LINE_H
        for line in self.tool_lines:
            spans = []
            for i, tok in enumerate(line):
                shared_color = self.tool_colors.get(tok)
                color = shared_color or _TOOL_DISTINCTIVE_COLOR
                weight = "700" if shared_color else "400"
                spans.append(
                    f'<tspan fill="{color}" font-weight="{weight}">'
                    f'{_esc(tok)}</tspan>'
                )
                if i < len(line) - 1:
                    spans.append(
                        f'<tspan fill="{_TOOL_DISTINCTIVE_COLOR}">, </tspan>'
                    )
            parts.append(
                f'<text x="20" y="{ty}" font-family="{_MONO_FONT}" '
                f'font-size="12">{"".join(spans)}</text>'
            )
            ty += _LINE_H
        if self.truncated:
            parts.append(
                f'<text x="20" y="{ty}" font-family="{_MONO_FONT}" '
                f'font-size="11" font-style="italic" fill="{_INK_SOFT}">'
                "…truncated, see agent source for the full list</text>"
            )
        parts.append("</g>")
        return "\n".join(parts)


def render_architecture_svg(
    graph: Graph,
    *,
    title: str | None = None,
    model: str | None = None,
    backend: str | None = None,
) -> str:
    """Render *graph*'s full anatomy — nodes, roles, tools, descriptions,
    edges — as a single self-contained, hand-laid-out SVG string.

    Vector by construction: open it, or rasterize it at any pixel size
    (10000x10000 included) with any SVG tool — no separate high-DPI export
    path is needed since nothing here is raster to begin with.
    """
    layers = _bfs_layers(graph)
    by_layer: dict[int, list[str]] = {}
    for n, ly in layers.items():
        by_layer.setdefault(ly, []).append(n)
    n_layers = max(by_layer) + 1 if by_layer else 1

    # Pass 0: which tool names appear on more than one node — the "shared
    # baseline" every specialist gets vs. what's actually distinctive.
    tools_by_node = {
        n: _node_tools(n, graph.nodes[n], graph) for n in graph.nodes
    }
    _tool_counts = Counter(t for ts in tools_by_node.values() for t in set(ts))
    shared_tools = {t for t, c in _tool_counts.items() if c > 1}
    tool_colors = _tool_colors(shared_tools)

    # Pass 1: measure every card's real content height.
    content = {
        n: _CardContent(n, graph.nodes[n], graph, tool_colors)
        for n in graph.nodes
    }
    max_row = max((len(v) for v in by_layer.values()), default=1)
    canvas_w = max(
        900, _MARGIN * 2 + max_row * _CARD_W + (max_row - 1) * _COL_GAP)

    row_heights = {
        ly: max((content[n].height for n in names), default=0)
        for ly, names in by_layer.items()
    }

    # Pass 2: assign positions from CUMULATIVE real row heights (not a
    # fixed per-layer constant — that's what silently overflowed before).
    row_y: dict[int, float] = {}
    y = _HEADER_H
    for ly in range(n_layers):
        row_y[ly] = y
        y += row_heights.get(ly, 0) + _ROW_GAP
    canvas_h = y - _ROW_GAP + _FOOTER_H

    positions: dict[str, tuple[float, float, float]] = {}  # cx, y, h
    cards: list[str] = []
    for ly in range(n_layers):
        names = sorted(by_layer.get(ly, []))
        row_w = len(names) * _CARD_W + (len(names) - 1) * _COL_GAP
        start_x = (canvas_w - row_w) / 2
        for i, name in enumerate(names):
            x = start_x + i * (_CARD_W + _COL_GAP)
            cards.append(content[name].svg(x, row_y[ly]))
            positions[name] = (x + _CARD_W / 2, row_y[ly], content[name].height)

    # Edges, drawn UNDER the cards. Two kinds, visually distinguished:
    #   PRIMARY (cross-layer, hub -> specialist): solid amber, converges
    #   into the row below.
    #   LATERAL (same-layer, specialist -> specialist): dashed neutral,
    #   arcs through the dedicated row gap BELOW its row so it never
    #   competes visually with the primary convergence above it.
    edge_frags = []
    lateral_by_target: dict[str, int] = {}
    for e in graph.edges:
        (x1, y1t, h1) = positions[e.source]
        (x2, y2t, h2) = positions[e.target]
        same = layers[e.source] == layers[e.target]
        if same:
            idx = lateral_by_target.get(e.target, 0)
            lateral_by_target[e.target] = idx + 1
            y0 = y1t + h1
            dip = 34 + idx * 26 + abs(x2 - x1) * 0.05
            midx = (x1 + x2) / 2
            path = f"M {x1},{y0} Q {midx},{y0 + dip} {x2},{y0}"
            edge_frags.append(
                f'<path d="{path}" fill="none" stroke="{_EDGE_LATERAL}" '
                'stroke-width="1.8" stroke-dasharray="2,5" '
                'stroke-linecap="round" marker-end="url(#arrowLateral)"/>'
            )
        else:
            y1 = y1t + h1
            y2 = y2t
            dy = max(50, (y2 - y1) * 0.55)
            path = f"M {x1},{y1} C {x1},{y1 + dy} {x2},{y2 - dy} {x2},{y2}"
            edge_frags.append(
                f'<path d="{path}" fill="none" stroke="{_EDGE_PRIMARY}" '
                'stroke-width="2.2" marker-end="url(#arrowPrimary)"/>'
            )
        if e.preamble:
            label = _wrap(e.preamble, 30)[0]
            mx = (x1 + x2) / 2
            my = (y1t + h1 + y2t) / 2 if not same else y1t + h1 + 20
            edge_frags.append(
                f'<rect x="{mx - len(label) * 3.4 - 6}" y="{my - 12}" '
                f'width="{len(label) * 6.8 + 12}" height="17" rx="4" '
                f'fill="{_BG}" stroke="{_EDGE_LATERAL}" stroke-width="1"/>'
                f'<text x="{mx}" y="{my}" text-anchor="middle" '
                f'font-family="{_MONO_FONT}" font-size="10.5" '
                f'fill="{_INK_SOFT}">{_esc(label)}</text>'
            )

    stamp = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")
    subtitle_bits = []
    if backend:
        subtitle_bits.append(f"backend={backend}")
    if model:
        subtitle_bits.append(f"model={model}")
    subtitle_bits.append(f"{len(graph.nodes)} nodes · {len(graph.edges)} edges")
    subtitle = "  ·  ".join(subtitle_bits)

    legend_roles = [
        role for role in _ROLE_COLORS
        if role in {a.role for a in graph.nodes.values()}
    ]
    legend_frags = []
    lx = _MARGIN
    ly_ = canvas_h - _FOOTER_H + 30
    for role in legend_roles:
        accent, _soft = _ROLE_COLORS[role]
        legend_frags.append(
            f'<circle cx="{lx}" cy="{ly_ - 5}" r="6" fill="{accent}"/>'
            f'<text x="{lx + 14}" y="{ly_}" font-family="{_MONO_FONT}" '
            f'font-size="11" fill="{_INK_SOFT}">{_esc(role)}</text>'
        )
        lx += 24 + len(role) * 7 + 26
    lx += 20
    legend_frags.append(
        f'<line x1="{lx}" y1="{ly_ - 9}" x2="{lx + 26}" y2="{ly_ - 9}" '
        f'stroke="{_EDGE_PRIMARY}" stroke-width="2.2"/>'
        f'<text x="{lx + 34}" y="{ly_}" font-family="{_MONO_FONT}" '
        f'font-size="11" fill="{_INK_SOFT}">delegate</text>'
    )
    lx += 34 + 7 * 8 + 26
    legend_frags.append(
        f'<line x1="{lx}" y1="{ly_ - 9}" x2="{lx + 26}" y2="{ly_ - 9}" '
        f'stroke="{_EDGE_LATERAL}" stroke-width="1.8" stroke-dasharray="2,5"/>'
        f'<text x="{lx + 34}" y="{ly_}" font-family="{_MONO_FONT}" '
        f'font-size="11" fill="{_INK_SOFT}">sub-delegate (lateral)</text>'
    )

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" \
viewBox="0 0 {canvas_w} {canvas_h}" font-family="{_BODY_FONT}">
  <defs>
    <marker id="arrowPrimary" viewBox="0 0 10 10" refX="9" refY="5"
            markerWidth="8" markerHeight="8" orient="auto-start-reverse">
      <path d="M0,0 L10,5 L0,10 z" fill="{_EDGE_PRIMARY}"/>
    </marker>
    <marker id="arrowLateral" viewBox="0 0 10 10" refX="9" refY="5"
            markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M0,0 L10,5 L0,10 z" fill="{_EDGE_LATERAL}"/>
    </marker>
  </defs>
  <rect width="{canvas_w}" height="{canvas_h}" fill="{_BG}"/>
  <text x="{_MARGIN}" y="56" font-family="{_DISPLAY_FONT}" \
font-size="34" font-weight="700" fill="{_INK}">\
{_esc(title or 'a3dasm run architecture')}</text>
  <text x="{_MARGIN}" y="84" font-family="{_MONO_FONT}" font-size="13" \
fill="{_INK_SOFT}">{_esc(subtitle)}</text>
  <line x1="{_MARGIN}" y1="100" x2="{canvas_w - _MARGIN}" y2="100" \
stroke="{_EDGE_LATERAL}" stroke-width="1"/>
  <g>{"".join(edge_frags)}</g>
  <g>{"".join(cards)}</g>
  <g>{"".join(legend_frags)}</g>
  <text x="{canvas_w - _MARGIN}" y="{canvas_h - 16}" text-anchor="end" \
font-family="{_MONO_FONT}" font-size="10" fill="{_INK_SOFT}">\
generated {stamp} — a3dasm._src.run_diagram</text>
</svg>
"""
    return svg
