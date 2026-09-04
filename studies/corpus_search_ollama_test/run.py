"""Custom-graph runner verifying BACKLOG #32's CorpusSearch/CorpusAdd fix.

Graph: strategizer (entry) -> literature_reviewer.

literature_reviewer ONLY points at a local endpoint when OLLAMA_BASE_URL is
set in the environment; strategizer stays on Haiku regardless -- mirrors
studies/mathexpert_kinematic_matching_test/run.py's isolation pattern.

Usage:
  OLLAMA_BASE_URL=http://127.0.0.1:<port>/v1 \\
    uv run python studies/corpus_search_ollama_test/run.py
"""
from __future__ import annotations

import os
from pathlib import Path

from a3dasm import AgenticRun, Edge, Graph, LiteratureReviewAgent, StrategizerAgent

STUDY_DIR = Path(__file__).parent
OLLAMA_MODEL = "qwen3.8-27b-256k"


def build_graph() -> Graph:
    literature_reviewer = LiteratureReviewAgent()
    if os.environ.get("OLLAMA_BASE_URL"):
        literature_reviewer = LiteratureReviewAgent(model=OLLAMA_MODEL)
        literature_reviewer.backend = "ollama"

    return Graph(
        nodes={
            "strategizer": StrategizerAgent(),
            "literature_reviewer": literature_reviewer,
        },
        edges=(Edge("strategizer", "literature_reviewer"),),
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
