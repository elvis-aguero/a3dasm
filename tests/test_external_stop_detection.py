"""Regression: a run stopped by an org-level Claude billing cap was
indistinguishable from an ordinary UNGATED close.

The Claude Code CLI backend returns a billing-cap notice as ORDINARY
assistant text (not a raised exception) — confirmed for real: a run's
strategizer turn ended with "You've hit your org's monthly spend limit ..."
as its final reply, made no further tool calls, and the run closed UNGATED
with no signal that this was an externally-caused stop rather than the
strategizer simply deciding it was done (or a real gate/science failure).
AgenticRun.execute() already supports cheap resumption via
`resume_from=<run_dir>` (test_resume.py) — the missing piece was detecting
and surfacing this specific stop condition so a user knows resuming is
appropriate. BACKLOG #34.
"""
from __future__ import annotations

import json
from pathlib import Path

from a3dasm._src.agent_runtime import AgenticRun


def _make_study(tmp_path: Path) -> Path:
    study = tmp_path / "study"
    study.mkdir()
    (study / "PROBLEM_STATEMENT.md").write_text("# trivial\n")
    return study


def test_org_spend_limit_text_recorded_as_stop_reason(tmp_path):
    study = _make_study(tmp_path)
    run = AgenticRun(study_dir=study, interactive=False)

    class _Stub:
        def invoke(self, state, config=None):
            return {
                "last_report": (
                    "## ⚠ UNGATED RUN\n\nYou've hit your org's monthly "
                    "spend limit · run /usage-credits to ask your admin "
                    "for a higher limit"
                ),
                "evals_used": 0,
            }

    run._graph = _Stub()
    run.execute()

    run_dir = next((study / "runs").iterdir())
    status = json.loads((run_dir / "debug" / "run_status.json").read_text())
    assert status["status"] == "UNGATED"
    assert status["stop_reason"] == "org_spend_limit"


def test_ordinary_ungated_close_has_no_stop_reason(tmp_path):
    """A normal UNGATED close (real gate/science failure, not a billing cap)
    must NOT be mislabelled — stop_reason stays null."""
    study = _make_study(tmp_path)
    run = AgenticRun(study_dir=study, interactive=False)

    class _Stub:
        def invoke(self, state, config=None):
            return {
                "last_report": (
                    "## ⚠ UNGATED RUN\n\nThis run is NOT validated: the "
                    "run terminated WITHOUT an accepted Done()."
                ),
                "evals_used": 0,
            }

    run._graph = _Stub()
    run.execute()

    run_dir = next((study / "runs").iterdir())
    status = json.loads((run_dir / "debug" / "run_status.json").read_text())
    assert status["status"] == "UNGATED"
    assert status["stop_reason"] is None
