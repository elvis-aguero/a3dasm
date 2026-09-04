"""Trivial 3-node graph (strategizer, implementer, critic) for live-watching
in the read-only run viewer -- see PROBLEM_STATEMENT.md. All-Haiku.

Usage:
  uv run python studies/three_node_ui_test/run.py
"""
from __future__ import annotations

from pathlib import Path

from a3dasm import (
    AdversarialCritiqueAgent,
    AgenticRun,
    Edge,
    F3dasmImplementerAgent,
    Graph,
    StrategizerAgent,
)

STUDY_DIR = Path(__file__).parent


def build_graph() -> Graph:
    return Graph(
        nodes={
            "strategizer": StrategizerAgent(),
            "implementer": F3dasmImplementerAgent(),
            "critic": AdversarialCritiqueAgent(),
        },
        edges=(
            Edge("strategizer", "implementer"),
            Edge("strategizer", "critic"),
        ),
        entry="strategizer",
    )


def main() -> None:
    report = AgenticRun(
        study_dir=STUDY_DIR,
        graph=build_graph(),
        interactive=False,
    ).execute()
    print(report)


if __name__ == "__main__":
    main()
