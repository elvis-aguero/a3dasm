"""Custom-graph runner for the MathExpert validation study.

No DataGenerator/Implementer/critic -- see PROBLEM_STATEMENT.md. Graph:
    strategizer (entry) -> literature_reviewer
                         -> math_expert
    math_expert          -> literature_reviewer   (its one outgoing edge,
                                                    grants Delegate/Wait/Reply)

Single-variable test of a local-model serving path: math_expert ONLY points
at a local endpoint when either VLLM_BASE_URL or OLLAMA_BASE_URL is set in
the environment; strategizer and literature_reviewer stay on Haiku
regardless -- deliberately isolating "does a local-model delegation work at
all" from "does a local model handle the full strategizer tool-calling
surface," which is a much bigger, separate question not being tested here.

vLLM path (~/.local/bin/vllm-serve-oscar.sh) hit a real dependency wall:
vllm==0.28.0's own precompiled C extension needs a CUDA 13 runtime this
cluster's L40S driver does not support (confirmed: torch itself correctly
resolved to a 12.9-compatible build via --torch-backend=auto, but vllm's
extension has libcudart.so.13 baked in at PyPI build time regardless).
Switched to Ollama (~/.local/bin/ollama-serve-oscar.sh), which ships its own
bundled runtime and already has qwen3.8:27b confirmed working on this
cluster's L40S via agentsoscar.sh's own prior real usage.

Usage:
  uv run python studies/mathexpert_kinematic_matching_test/run.py                     # all-Haiku
  OLLAMA_BASE_URL=http://127.0.0.1:<port>/v1 \\
    uv run python studies/mathexpert_kinematic_matching_test/run.py                   # math_expert on Ollama
"""
from __future__ import annotations

import os
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
VLLM_MODEL = "Qwen/Qwen3.8-27B-FP8"
OLLAMA_MODEL = "qwen3.8-27b-256k"  # the ctx-extended model ollama-serve-oscar.sh creates


def build_graph() -> Graph:
    math_expert = MathExpertAgent()
    if os.environ.get("VLLM_BASE_URL"):
        math_expert = MathExpertAgent(model=VLLM_MODEL)
        math_expert.backend = "vllm"
    elif os.environ.get("OLLAMA_BASE_URL"):
        math_expert = MathExpertAgent(model=OLLAMA_MODEL)
        math_expert.backend = "ollama"

    return Graph(
        nodes={
            "strategizer": StrategizerAgent(),
            "literature_reviewer": LiteratureReviewAgent(),
            "math_expert": math_expert,
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
