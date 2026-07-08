"""Tests for the datagenerator -> canonical-oracle registration handoff.

Phase 1: the pure registration function that atomically updates
run_config.json so get_evaluator() (re-reading config each call) resolves
an agent-authored DataGenerator with no manual config edit.
"""
from __future__ import annotations

import json
import textwrap
from pathlib import Path

from a3dasm._src.agent_runtime import register_evaluator_entrypoint

from tests.test_evaluator_resolution import (
    _make_delegation_dir,
    _write_run_config,
)


def test_register_updates_run_config_atomically(tmp_path):
    study_dir = tmp_path / "study"
    study_dir.mkdir()
    store_dir = tmp_path / "store"
    store_dir.mkdir()
    debug_dir, _ = _make_delegation_dir(tmp_path)
    _write_run_config(debug_dir, store_dir, study_dir)
    cfg_path = debug_dir / "run_config.json"

    gen_abs = study_dir / "runs" / "T" / "debug" / "delegations" / "D001" \
        / "generators" / "x.py"
    gen_abs.parent.mkdir(parents=True, exist_ok=True)
    gen_abs.write_text("def x_gen(**kw):\n    return sum(kw.values())\n")

    ep = register_evaluator_entrypoint(
        cfg_path, gen_abs, "x_gen", output_names=["f"]
    )

    cfg = json.loads(cfg_path.read_text())
    assert ep == "runs/T/debug/delegations/D001/generators/x.py:x_gen"
    assert cfg["evaluator_entrypoint"] == ep
    assert cfg["evaluator_output_names"] == ["f"]
    assert cfg["evaluator_lookup"] is None
    # untouched keys preserved
    assert cfg["study_dir"] == str(study_dir)
    assert cfg["store_dir"] == str(store_dir)
    # no temp file left behind
    assert not (debug_dir / "run_config.json.tmp").exists()


def test_register_then_get_evaluator_resolves(tmp_path, monkeypatch):
    """register -> chdir into a D### dir -> get_evaluator() resolves and runs."""
    from a3dasm._src.instrumented import get_evaluator
    from f3dasm._src.experimentdata import ExperimentData
    from f3dasm._src.design.domain import Domain
    from f3dasm._src.experimentsample import ExperimentSample, JobStatus

    study_dir = tmp_path / "study"
    study_dir.mkdir()
    store_dir = tmp_path / "store"
    store_dir.mkdir()
    debug_dir, delegation_dir = _make_delegation_dir(tmp_path, "D002")
    _write_run_config(debug_dir, store_dir, study_dir)
    cfg_path = debug_dir / "run_config.json"

    # Author a DataGenerator subclass under a (different) delegation dir.
    gen_abs = (study_dir / "runs" / "T" / "debug" / "delegations" / "D001"
               / "generators" / "authored.py")
    gen_abs.parent.mkdir(parents=True, exist_ok=True)
    gen_abs.write_text(textwrap.dedent("""\
        def authored(**kwargs):
            return float(sum(kwargs.values()))
    """))

    register_evaluator_entrypoint(
        cfg_path, gen_abs, "authored", output_names=["f"]
    )

    # Worker runs from its delegation dir; get_evaluator() resolves the
    # freshly-registered entrypoint from the updated run_config.json.
    monkeypatch.chdir(delegation_dir)
    gen = get_evaluator()  # no inner

    dom = Domain()
    dom.add_float("a", 0.0, 1.0)
    dom.add_float("b", 0.0, 1.0)
    sample = ExperimentSample(
        _input_data={"a": 1.0, "b": 2.0},
        _output_data={},
        job_status=JobStatus.OPEN,
    )
    out = gen.execute(sample)
    assert out._output_data["f"] == 3.0
    assert out._output_data["_delegation_id"] == "D002"  # metered to caller


def test_register_namespace_writes_oracles_block_not_default(tmp_path):
    """Registering for a namespace writes oracles[ns] and creates its own
    isolated store + sentinel — the canonical default oracle is untouched."""
    from a3dasm._src._f3dasm_compat import PROTECTED_STORE_SENTINEL

    study_dir = tmp_path / "study"
    study_dir.mkdir()
    store_dir = tmp_path / "store"
    store_dir.mkdir()
    debug_dir, _ = _make_delegation_dir(tmp_path)
    _write_run_config(debug_dir, store_dir, study_dir)
    cfg_path = debug_dir / "run_config.json"

    gen_abs = study_dir / "ell_gen.py"
    gen_abs.write_text("def g(**k):\n    return float(sum(k.values()))\n")

    ep = register_evaluator_entrypoint(
        cfg_path, gen_abs, "g", output_names=["f"], namespace="ell"
    )

    cfg = json.loads(cfg_path.read_text())
    # canonical default untouched
    assert cfg["evaluator_entrypoint"] is None
    # namespace block written
    block = cfg["oracles"]["ell"]
    assert block["evaluator_entrypoint"] == ep
    assert block["evaluator_output_names"] == ["f"]
    assert block["evaluator_lookup"] is None
    # isolated, protected store created for the namespace
    ns_store = Path(block["store_dir"])
    assert ns_store.exists()
    assert (ns_store / PROTECTED_STORE_SENTINEL).exists()
    assert ns_store != store_dir


def test_register_namespace_then_get_evaluator_resolves(tmp_path, monkeypatch):
    """register(namespace) → F3DASM_NAMESPACE worker → get_evaluator() resolves
    the namespace oracle and writes its own ledger."""
    from a3dasm._src.instrumented import get_evaluator
    from f3dasm._src.experimentdata import ExperimentData
    from f3dasm._src.experimentsample import ExperimentSample, JobStatus

    study_dir = tmp_path / "study"
    study_dir.mkdir()
    store_dir = tmp_path / "store"
    store_dir.mkdir()
    debug_dir, delegation_dir = _make_delegation_dir(tmp_path, "D004")
    _write_run_config(debug_dir, store_dir, study_dir)
    cfg_path = debug_dir / "run_config.json"

    (study_dir / "ell_gen.py").write_text(
        "def g(**k):\n    return float(sum(k.values()))\n")
    register_evaluator_entrypoint(
        cfg_path, study_dir / "ell_gen.py", "g",
        output_names=["f"], namespace="ell")

    monkeypatch.chdir(delegation_dir)
    monkeypatch.setenv("F3DASM_NAMESPACE", "ell")
    gen = get_evaluator()  # namespace from env
    out = gen.execute(ExperimentSample(
        _input_data={"a": 2.0, "b": 5.0}, _output_data={},
        job_status=JobStatus.OPEN))
    assert out._output_data["f"] == 7.0

    block = json.loads(cfg_path.read_text())["oracles"]["ell"]
    assert len(ExperimentData.from_file(project_dir=Path(block["store_dir"]))) == 1
    # canonical default store stayed empty
    assert not (store_dir / "output.csv").exists()


def test_register_relative_path_kept(tmp_path):
    study_dir = tmp_path / "study"
    study_dir.mkdir()
    store_dir = tmp_path / "store"
    store_dir.mkdir()
    debug_dir, _ = _make_delegation_dir(tmp_path)
    _write_run_config(debug_dir, store_dir, study_dir)
    cfg_path = debug_dir / "run_config.json"

    ep = register_evaluator_entrypoint(
        cfg_path, "runs/T/debug/delegations/D001/generators/x.py", "g"
    )
    assert ep == "runs/T/debug/delegations/D001/generators/x.py:g"


def test_register_idempotent(tmp_path):
    study_dir = tmp_path / "study"
    study_dir.mkdir()
    store_dir = tmp_path / "store"
    store_dir.mkdir()
    debug_dir, _ = _make_delegation_dir(tmp_path)
    _write_run_config(debug_dir, store_dir, study_dir)
    cfg_path = debug_dir / "run_config.json"

    ep1 = register_evaluator_entrypoint(cfg_path, "g/x.py", "g", ["f"])
    cfg1 = cfg_path.read_text()
    ep2 = register_evaluator_entrypoint(cfg_path, "g/x.py", "g", ["f"])
    cfg2 = cfg_path.read_text()
    assert ep1 == ep2
    assert cfg1 == cfg2
