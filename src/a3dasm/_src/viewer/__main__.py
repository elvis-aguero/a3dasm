"""CLI entry point for the read-only live run viewer.

Usage
-----
    python -m a3dasm.viewer <study-dir>

Requires the ``viewer`` optional-dependency group
(``pip install a3dasm[viewer]``).

Options
-------
--host HOST   Interface to bind (default: 127.0.0.1 — local-only, no auth).
--port PORT   Port to bind (default: 8765).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m a3dasm.viewer",
        description=(
            "Serve a read-only live web viewer for a3dasm agentic runs."
        ),
    )
    parser.add_argument(
        "study_dir",
        metavar="study-dir",
        type=Path,
        help="Path to the study directory whose runs/ this viewer serves.",
    )
    parser.add_argument(
        "--host", default="127.0.0.1", metavar="HOST",
        help="Interface to bind (default: 127.0.0.1).",
    )
    parser.add_argument(
        "--port", type=int, default=8765, metavar="PORT",
        help="Port to bind (default: 8765).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        from .app import run_viewer
    except ImportError as exc:
        print(
            "Error: the viewer requires the 'viewer' optional-dependency "
            "group — install with `pip install a3dasm[viewer]`.\n"
            f"({exc})",
            file=sys.stderr,
        )
        return 1

    run_viewer(args.study_dir, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    sys.exit(main())
