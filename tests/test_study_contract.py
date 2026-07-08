"""Executable study-folder contract.

Pins docs/agentic/authoring-a-study.md to the runtime by running the canonical
example study (docs/agentic/example_study/) through the REAL config loader and
get_evaluator() path. If the study-folder contract drifts, this fails — forcing
the doc to be updated. Same "executable docs" guarantee as idioms.py /
test_f3dasm_idioms.py.

Forward-compatibility: DOCUMENTED_*_KEYS below is the canonical config surface.
Adding a config key means updating both this set AND authoring-a-study.md; the
example is asserted to use only documented keys, so it can't silently drift.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from f3dasm._src.experimentsample import ExperimentSample, JobStatus

_REPO = Path(__file__).resolve().parents[1]
EXAMPLE = _REPO / "docs" / "example_study"

# The config surface the contract documents (authoring-a-study.md). Keep in sync.
DOCUMENTED_TOP_KEYS = {
    "model", "backend", "budget", "eval_budget",
    "required_deliverables", "evaluator",
}
DOCUMENTED_EVALUATOR_KEYS = {
    "entrypoint", "output_names", "lookup", "fidelity_column", "name",
}


def test_example_study_has_the_documented_layout():
    assert (EXAMPLE / "PROBLEM_STATEMENT.md").is_file(), "REQUIRED brief missing"
    assert (EXAMPLE / "config.yaml").is_file()
    assert (EXAMPLE / "workspace" / "evaluator.py").is_file()


def test_example_config_uses_only_documented_keys():
    cfg = yaml.safe_load((EXAMPLE / "config.yaml").read_text()) or {}
    extra_top = set(cfg) - DOCUMENTED_TOP_KEYS
    assert not extra_top, f"undocumented top-level config keys: {extra_top}"
    extra_eval = set(cfg.get("evaluator", {})) - DOCUMENTED_EVALUATOR_KEYS
    assert not extra_eval, f"undocumented evaluator keys: {extra_eval}"
    # checkpoint_every is a known no-op — the example must NOT model it as real.
    assert "checkpoint_every" not in cfg


def test_documented_config_keys_resolve_through_agenticrun():
    """The documented keys must actually flow through the real loader onto the
    run — if the loader stops reading one, this breaks."""
    from a3dasm import AgenticRun

    run = AgenticRun(study_dir=EXAMPLE)
    assert run._backend == "claude"
    assert "haiku" in run._model
    assert run._eval_budget == 200
    # pipeline.ipynb is auto-required (not listed in _required_deliverables,
    # which holds only EXTRA deliverables); the notebook is the only output.
    assert run._required_deliverables == []


def test_example_evaluator_resolves_and_runs(tmp_path, monkeypatch):
    """The documented evaluator entrypoint resolves via get_evaluator() and
    evaluates one sample — the core oracle contract, end to end."""
    from a3dasm._src.instrumented import get_evaluator

    store_dir = tmp_path / "store"
    store_dir.mkdir()
    debug_dir = tmp_path / "debug"
    delegation_dir = debug_dir / "delegations" / "D001"
    delegation_dir.mkdir(parents=True)
    cfg = {
        "store_dir": str(store_dir),
        "lock_path": str(store_dir / "experiment_data" / ".lock"),
        "evaluator_name": "example_study",
        "evaluator_entrypoint": "workspace/evaluator.py:evaluate",
        "evaluator_output_names": ["y"],
        "evaluator_lookup": None,
        "fidelity_column": None,
        "study_dir": str(EXAMPLE),
    }
    (debug_dir / "run_config.json").write_text(json.dumps(cfg))
    monkeypatch.chdir(delegation_dir)
    monkeypatch.delenv("F3DASM_DELEGATION_ID", raising=False)

    gen = get_evaluator()  # resolves docs/agentic/example_study/workspace/evaluator.py:evaluate
    sample = ExperimentSample(
        _input_data={"x1": 1.0, "x2": -2.0},
        _output_data={},
        job_status=JobStatus.OPEN,
    )
    out = gen.execute(sample)
    # global minimum: (x1-1)^2 + (x2+2)^2 = 0 at (1, -2)
    assert out._output_data["y"] == pytest.approx(0.0, abs=1e-9)
    assert out._output_data["_delegation_id"] == "D001"
