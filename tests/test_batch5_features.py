"""Pins for the 2026-06-11 batch:
  #5  the 3-strikes UNGATED escape is a SILENT backstop (not coached to the agent)
  #6  extensible, oracle-stamped provenance columns (open schema)
  #10b recursion_limit default raised to 2000
"""
from __future__ import annotations

import importlib
import json
from pathlib import Path

from f3dasm._src.experimentdata import ExperimentData
from f3dasm._src.experimentsample import ExperimentSample, JobStatus

_SRC = Path(__file__).resolve().parents[1] / "src" / "a3dasm" / "_src"


# --- #5: escape is a silent backstop, never advertised ----------------------

def test_three_strikes_escape_not_coached_to_agent():
    """The strategizer-facing critic-verdict message must not teach the N=3
    escape (advertising it makes the agent exhaust the critic instead of
    earning a PASS). The backstop logic itself stays."""
    nodes = (_SRC / "nodes" / "tools" / "routing.py").read_text()
    # The old coaching phrasing must be gone from the agent-facing message.
    assert "again anyway" not in nodes
    assert "after 3 attempts the run closes" not in nodes
    assert "(revision {node._revise_count}/3)" not in nodes
    # The silent backstop must still exist.
    assert "_revise_count >= 3" in nodes


# --- #6: extensible, oracle-stamped provenance -------------------------------

def test_extra_provenance_stamped_and_persisted(tmp_path, monkeypatch):
    from a3dasm._src.instrumented import get_evaluator

    study = tmp_path / "study"
    study.mkdir()
    (study / "tiny_eval.py").write_text(
        "def evaluate_kw(**k):\n    return float(sum(k.values()))\n")
    store = tmp_path / "store"
    store.mkdir()
    debug = tmp_path / "debug"
    deleg = debug / "delegations" / "D001"
    deleg.mkdir(parents=True)
    cfg = {
        "store_dir": str(store),
        "lock_path": str(store / "experiment_data" / ".lock"),
        "evaluator_name": "t",
        "evaluator_entrypoint": "tiny_eval.py:evaluate_kw",
        "evaluator_output_names": ["f"],
        "evaluator_lookup": None,
        "fidelity_column": None,
        "study_dir": str(study),
        "provenance": {"fidelity": 7, "regime": "linear"},
    }
    (debug / "run_config.json").write_text(json.dumps(cfg))
    monkeypatch.chdir(deleg)
    monkeypatch.delenv("F3DASM_DELEGATION_ID", raising=False)

    gen = get_evaluator()
    s = ExperimentSample(
        _input_data={"x0": 0.3, "x1": 0.2}, _output_data={},
        job_status=JobStatus.OPEN)
    out = gen.execute(s)

    # Stamped on the row (oracle-side, not agent-authored)...
    assert out._output_data["fidelity"] == 7
    assert out._output_data["regime"] == "linear"
    # ...and persisted in the canonical store as real columns.
    data = ExperimentData.from_file(project_dir=store)
    _, df_out = data.to_pandas()
    assert "fidelity" in df_out.columns
    assert int(df_out.iloc[0]["fidelity"]) == 7
    assert "regime" in df_out.columns


def test_no_provenance_block_means_no_extra_columns(tmp_path):
    """Open schema is opt-in: absent 'provenance' → only the fixed three."""
    from a3dasm._src.instrumented import InstrumentedDataGenerator
    from f3dasm._src.core import DataGenerator

    class _Const(DataGenerator):
        def execute(self, s, **k):
            s._output_data["y"] = 1.0
            return s

    gen = InstrumentedDataGenerator(
        inner=_Const(), store_dir=tmp_path / "s", delegation_id="D001")
    out = gen.execute(ExperimentSample(
        _input_data={"x": 0.0}, _output_data={}, job_status=JobStatus.OPEN))
    assert set(out._output_data) == {
        "y", "_delegation_id", "_source", "_ts", "_wall_ms"}


# --- #10b: recursion_limit default raised ------------------------------------

def test_recursion_limit_default_raised():
    # Now a config.yaml runtime knob (F3DASM_RECURSION_LIMIT overrides); the
    # default stays high (2000) so long multi-delegation runs don't crash.
    from a3dasm._src import settings
    settings.configure({})
    assert settings.get_int("recursion_limit", 2000) == 2000
    rt = (_SRC / "agent_runtime.py").read_text()
    assert 'settings.get_int("recursion_limit", 2000)' in rt
    assert '"500"' not in rt.split("recursion_limit")[1][:200]
