"""Custom-graph runner for the MathExpert validation study.

No DataGenerator/Implementer/critic -- see PROBLEM_STATEMENT.md. Graph:
    strategizer (entry) -> literature_reviewer
                         -> math_expert
    math_expert          -> literature_reviewer   (its one outgoing edge,
                                                    grants Delegate/Wait/Reply)

Usage: uv run python studies/mathexpert_kinematic_matching_test/run.py
"""
from __future__ import annotations

from pathlib import Path

from a3dasm import (
    AgenticRun,
    Edge,
    Graph,
    LiteratureReviewAgent,
    MathExpertAgent,
    StrategizerAgent,
)

STUDY_DIR = Path(__file__).parent


def build_graph() -> Graph:
    return Graph(
        nodes={
            "strategizer": StrategizerAgent(),
            "literature_reviewer": LiteratureReviewAgent(),
            "math_expert": MathExpertAgent(),
        },
        edges=(
            Edge("strategizer", "literature_reviewer"),
            Edge("strategizer", "math_expert"),
            Edge("math_expert", "literature_reviewer"),
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
