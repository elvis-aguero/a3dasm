"""Regression: the ledger must report the TRUE gate outcome for notebook studies.

Before the fix, run_ledger.extract() mapped any stamped pipeline.ipynb to GATED
and only read UNGATED from solution.md (which notebook studies never produce).
So 3-strike (UNGATED) and failed closes were silently recorded as GATED. The
gate outcome is now persisted in nb.metadata.agentic.gate_outcome by
agent_runtime stamping; extract() must honor it.
"""
from __future__ import annotations

import sys
from pathlib import Path

import nbformat

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "studies"))
import run_ledger  # noqa: E402


def _run_dir_with_nb(tmp_path: Path, agentic_meta: dict | None) -> Path:
    study = tmp_path / "studies" / "demo"
    run_dir = study / "runs" / "20260622T000000"
    (run_dir / "debug").mkdir(parents=True)
    if agentic_meta is not None:
        nb = nbformat.v4.new_notebook()
        nb.cells.append(nbformat.v4.new_markdown_cell("# deliverable"))
        nb.metadata["agentic"] = agentic_meta
        nbformat.write(nb, str(study / "pipeline.ipynb"))
    return run_dir


def _outcome(tmp_path, meta):
    return run_ledger.extract(_run_dir_with_nb(tmp_path, meta))["outcome"]


def test_ungated_stamp_reported_as_ungated(tmp_path):
    assert _outcome(tmp_path, {"run": "x", "gate_outcome": "UNGATED"}) == "UNGATED"


def test_gated_stamp_reported_as_gated(tmp_path):
    assert _outcome(tmp_path, {"run": "x", "gate_outcome": "GATED"}) == "GATED"


def test_failed_stamp_reported_as_failed(tmp_path):
    assert _outcome(tmp_path, {"run": "x", "gate_outcome": "FAILED"}) == "FAILED"


def test_prefix_run_without_gate_field_falls_back_to_gated(tmp_path):
    # Pre-fix notebooks have no gate_outcome key; preserve old behavior.
    assert _outcome(tmp_path, {"run": "x"}) == "GATED"


def test_unstamped_notebook_is_failed(tmp_path):
    # Notebook exists but was never stamped (run never closed) → FAILED.
    assert _outcome(tmp_path, {}) == "FAILED"
