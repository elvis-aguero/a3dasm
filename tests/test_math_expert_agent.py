"""Tests for MathExpertAgent's graph wiring — spec 10, tests 10-11.

test 10 (native tools documented in FEATURES.md) is covered by the existing,
generic tests/test_features_documented.py — it walks every class in
a3dasm._src.agents, so MathExpertAgent is already exercised with no new test
needed (confirmed green as soon as the class existed).
"""

from __future__ import annotations

from a3dasm import Agent, Edge, Graph, MathExpertAgent
from a3dasm._src.agents import _default_graph


def test_math_expert_not_in_default_graph():
    """Opt-in, like DebuggerAgent: never silently added to the standard
    5-node topology that every existing study already uses."""
    graph = _default_graph()
    assert "math_expert" not in graph.nodes


def test_math_expert_wired_into_a_custom_graph_mirrors_literature_reviewer_fanin():
    class Strategist(Agent):
        role = "strategizer"
        description = "hub"

    class Implementer(Agent):
        description = "implementer"

    class DataGenerator(Agent):
        description = "datagenerator"

    class LiteratureReviewer(Agent):
        description = "literature reviewer"

    graph = Graph(
        nodes={
            "strategizer": Strategist(),
            "implementer": Implementer(),
            "datagenerator": DataGenerator(),
            "literature_reviewer": LiteratureReviewer(),
            "math_expert": MathExpertAgent(),
        },
        edges=(
            Edge("strategizer", "math_expert"),
            Edge("implementer", "math_expert"),
            Edge("datagenerator", "math_expert"),
            Edge("math_expert", "literature_reviewer"),
        ),
        entry="strategizer",
    )

    assert set(graph.incoming("math_expert")) == {
        "strategizer", "implementer", "datagenerator",
    }
    # The one outbound edge is what grants Delegate/Wait/Reply/FollowUp/
    # RecallHistory per the existing topology rule (base.py: any node with
    # >=1 outgoing edge) -- no special-casing needed for math_expert.
    assert graph.outgoing("math_expert") == ["literature_reviewer"]


def test_math_expert_agent_tools_mirror_implementer_kind_no_closure_tools():
    agent = MathExpertAgent()
    assert agent.role == "math_expert"
    assert {"Bash", "Edit", "Write", "Read", "Glob", "Grep"} <= agent.tools
    # No bespoke closure vocabulary: build_closure_tools is inherited
    # unchanged from the base default (read-only corpus lookup), unlike
    # LiteratureReviewAgent which overrides it entirely.
    assert type(agent).build_closure_tools is Agent.build_closure_tools
