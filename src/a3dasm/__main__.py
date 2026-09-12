"""CLI entry point for the agentic-f3dasm runtime.

Usage
-----
    python -m a3dasm <study-dir>

The study directory must contain a ``PROBLEM_STATEMENT.md`` file.

Options
-------
--model <id>      LLM model identifier (default: claude-haiku-4-5-20251001).
--budget DURATION Wall-clock budget: seconds or HH:MM:SS (default: unlimited).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

__author__ = "Elvis Aguero (elvis_alexander_aguero_vera@brown.edu)"
__credits__ = ["Elvis Aguero"]
__status__ = "Experimental"


def _budget(value: str) -> float:
    """``--budget`` accepts what config.yaml's ``budget:`` accepts.

    Both spellings go through the SAME parser the study config uses
    (``run_setup._parse_budget_str``), so "45m of wall clock" is written the
    same way wherever it is written. This entry point used to take
    ``type=float`` while the container entrypoint
    (``_src/runtime/run.py``) took seconds OR HH:MM:SS, so the identical flag
    on two entry points accepted different values and ``--budget 00:45:00``
    failed here with an argparse float error.
    """
    from ._src.runtime.run_setup import _parse_budget_str
    try:
        parsed = _parse_budget_str(value)
    except (TypeError, ValueError):
        parsed = None
    if parsed is None:
        raise argparse.ArgumentTypeError(
            f"invalid duration {value!r}: expected seconds (e.g. 2700) "
            "or HH:MM:SS (e.g. 00:45:00)")
    return parsed


def _build_parser() -> argparse.ArgumentParser:
    from ._src.runtime.agent_runtime import DEFAULT_MODEL

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
        type=_budget,
        default=None,
        metavar="DURATION",
        help="Wall-clock time budget: seconds, or HH:MM:SS "
             "(default: unlimited).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    from ._src.runtime.agent_runtime import AgenticRun, AgenticRunError

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
