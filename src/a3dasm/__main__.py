"""CLI entry point for the agentic-f3dasm runtime.

Usage
-----
    python -m a3dasm <study-dir>

The study directory must contain a ``PROBLEM_STATEMENT.md`` file.

Options
-------
--model <id>      LLM model identifier (default: claude-haiku-4-5-20251001).
--budget SECONDS  Wall-clock time budget in seconds (default: unlimited).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

__author__ = "Elvis Aguero (elvis_alexander_aguero_vera@brown.edu)"
__credits__ = ["Elvis Aguero"]
__status__ = "Experimental"


def _build_parser() -> argparse.ArgumentParser:
    from ._src.agent_runtime import DEFAULT_MODEL

    parser = argparse.ArgumentParser(
        prog="python -m a3dasm",
        description="Run the agentic-f3dasm runtime against a study directory.",
    )
    parser.add_argument(
        "study_dir",
        metavar="study-dir",
        type=Path,
        help="Path to the study directory. Must contain PROBLEM_STATEMENT.md.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        metavar="MODEL",
        help=f"LLM model identifier (default: {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--budget",
        type=float,
        default=None,
        metavar="SECONDS",
        help="Wall-clock time budget in seconds (default: unlimited).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    from ._src.agent_runtime import AgenticRun, AgenticRunError

    parser = _build_parser()
    args = parser.parse_args(argv)

    run = AgenticRun(
        study_dir=Path(args.study_dir),
        model=args.model,
        budget=args.budget,
    )

    try:
        report = run.execute()
    except AgenticRunError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
