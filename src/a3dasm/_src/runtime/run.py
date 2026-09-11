"""Container entrypoint: python -m a3dasm._src.run [study_dir] [--model X] [--budget N]"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .agent_runtime import AgenticRun, AgenticRunError

__all__ = ["AgenticRun", "AgenticRunError", "main"]


def main() -> None:

    p = argparse.ArgumentParser(description="Run an f3dasm agentic study.")
    p.add_argument("study_dir", nargs="?", default="/study",
                   help="Path to the study directory (default: /study)")
    p.add_argument("--model", default=None)
    p.add_argument("--budget", default=None,
                   help="Time budget in seconds or HH:MM:SS")
    args = p.parse_args()

    try:
        AgenticRun(
            Path(args.study_dir),
            model=args.model,
            budget=args.budget,
        ).execute()
    except (AgenticRunError, Exception) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
