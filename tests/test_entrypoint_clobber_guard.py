"""register_evaluator_entrypoint must not SILENTLY replace the canonical
evaluator_entrypoint.

Regression for run 20260706T204732: a datagenerator authored a NEW-family
oracle (elliptical rings) but the namespace did not propagate (Delegate arg
and manifest both null), so registration fell through to the namespace=None
branch and unconditionally overwrote the baseline canonical entrypoint with no
backup or warning. D006 replaced the baseline; D007 caught it and self-healed
via a hand-made .bak. Legitimate canonical re-registration (same or updated
oracle) must still be allowed — but overwriting with a DIFFERENT file must
snapshot the prior config and warn loudly so an accidental clobber is visible
and recoverable.
"""
from __future__ import annotations

import json
import logging

from a3dasm._src.agent_runtime import register_evaluator_entrypoint


def _write_config(tmp_path, entrypoint=None):
    cfg = {"study_dir": str(tmp_path),
           "store_dir": str(tmp_path / "experiment_data")}
    if entrypoint:
        cfg["evaluator_entrypoint"] = entrypoint
        cfg["evaluator_output_names"] = ["f"]
    p = tmp_path / "run_config.json"
    p.write_text(json.dumps(cfg))
    return p


def test_overwriting_a_different_canonical_entrypoint_snapshots_and_warns(
        tmp_path, caplog):
    (tmp_path / "baseline.py").write_text("def gen(x):\n    return x\n")
    cfg = _write_config(tmp_path, entrypoint="baseline.py:gen")
    (tmp_path / "elliptical.py").write_text("def gen(x):\n    return x\n")

    with caplog.at_level(logging.WARNING, logger="a3dasm"):
        ep = register_evaluator_entrypoint(
            cfg, tmp_path / "elliptical.py", "gen",
            output_names=["f"], namespace=None)

    # The overwrite still happens — legitimate re-registration is not blocked.
    assert ep == "elliptical.py:gen"
    assert json.loads(cfg.read_text())["evaluator_entrypoint"] == \
        "elliptical.py:gen"
    # A snapshot of the prior config preserves the OLD entrypoint.
    bak = cfg.with_name(cfg.name + ".bak_preclobber")
    assert bak.exists(), "prior config was not snapshotted before clobber"
    assert json.loads(bak.read_text())["evaluator_entrypoint"] == \
        "baseline.py:gen"
    # And a loud warning was emitted.
    assert any("canonical evaluator_entrypoint" in r.getMessage()
               for r in caplog.records), [r.getMessage() for r in caplog.records]


def test_same_entrypoint_reregistration_is_silent(tmp_path):
    (tmp_path / "baseline.py").write_text("def gen(x):\n    return x\n")
    cfg = _write_config(tmp_path, entrypoint="baseline.py:gen")
    register_evaluator_entrypoint(
        cfg, tmp_path / "baseline.py", "gen", output_names=["f"],
        namespace=None)
    assert not cfg.with_name(cfg.name + ".bak_preclobber").exists()


def test_first_canonical_registration_does_not_snapshot(tmp_path):
    cfg = _write_config(tmp_path, entrypoint=None)
    (tmp_path / "baseline.py").write_text("def gen(x):\n    return x\n")
    register_evaluator_entrypoint(
        cfg, tmp_path / "baseline.py", "gen", output_names=["f"],
        namespace=None)
    assert not cfg.with_name(cfg.name + ".bak_preclobber").exists()


def test_namespaced_registration_never_touches_canonical(tmp_path):
    (tmp_path / "baseline.py").write_text("def gen(x):\n    return x\n")
    cfg = _write_config(tmp_path, entrypoint="baseline.py:gen")
    (tmp_path / "elliptical.py").write_text("def gen(x):\n    return x\n")
    register_evaluator_entrypoint(
        cfg, tmp_path / "elliptical.py", "gen", output_names=["f"],
        namespace="elliptical_rings")
    data = json.loads(cfg.read_text())
    assert data["evaluator_entrypoint"] == "baseline.py:gen"  # untouched
    assert "elliptical_rings" in data["oracles"]
    assert not cfg.with_name(cfg.name + ".bak_preclobber").exists()
