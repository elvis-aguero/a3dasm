"""Run-3 hang: the pre-run problem-statement review called input() on a headless
background run (no TTY) and blocked forever (80 min, idle CPU, never reached
"Invoking graph"). The review keyed off self._interactive alone, missing the
isatty() guard the in-graph FollowUp path already had.

interactive must require a REAL terminal: in a no-TTY context (background run,
or pytest) it is forced False even when requested, so no input() path can block.
"""
from __future__ import annotations

from a3dasm._src.agent_runtime import AgenticRun


def test_headless_forces_noninteractive(tmp_path):
    # pytest runs with a non-TTY stdin → interactive=True must resolve to False.
    run = AgenticRun(study_dir=tmp_path, interactive=True)
    assert run._interactive is False


def test_interactive_false_stays_false(tmp_path):
    run = AgenticRun(study_dir=tmp_path, interactive=False)
    assert run._interactive is False
