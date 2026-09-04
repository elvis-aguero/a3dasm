"""Quota probe: does a3dasm's own Claude Code CLI invocation still work at
all right now, as cheaply as possible?

Graph: strategizer (entry) -> critic. All-Haiku, run locally (not Oscar,
no SLURM) -- deliberately removes every other moving part (cluster, tunnel,
local model serving) so a failure here can only mean one thing: the
account-level block a prior Oscar run hit (BACKLOG #34,
"You've hit your org's monthly spend limit") is still active for ANY
Claude usage, not just heavy/automated runs.

Usage:
  uv run python studies/quota_probe_test/run.py
"""
from __future__ import annotations

from pathlib import Path

from a3dasm import AdversarialCritiqueAgent, AgenticRun, Edge, Graph, StrategizerAgent

STUDY_DIR = Path(__file__).parent


def build_graph() -> Graph:
    return Graph(
        nodes={
            "strategizer": StrategizerAgent(),
            "critic": AdversarialCritiqueAgent(),
        },
        edges=(Edge("strategizer", "critic"),),
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
