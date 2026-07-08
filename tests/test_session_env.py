"""Regression: the worker session env must carry F3DASM_CANONICAL_STORE.

Run 20260624T021359: a worker campaign read os.environ['F3DASM_CANONICAL_STORE']
(empty by default), defaulted to the wrong namespace, and overwrote another
delegation's scratch data. The Claude backend injected F3DASM_DELEGATION_ID +
F3DASM_RUN_CONFIG but not the store path; get_evaluator() reads store_dir from
run_config, but the worker's own scripts read the env var directly.
"""
from __future__ import annotations

import json

from a3dasm._src.backends.base import (
    set_delegation_id,
    set_run_config_path,
)
from a3dasm._src.backends.claude import _build_session_env


def _with_run_config(tmp_path, store_dir):
    rc = tmp_path / "debug" / "run_config.json"
    rc.parent.mkdir(parents=True, exist_ok=True)
    rc.write_text(json.dumps({"store_dir": str(store_dir), "study_dir": "x"}))
    return rc


def test_session_env_includes_canonical_store(tmp_path):
    store = tmp_path / "runs" / "T0" / "experiment_data"
    rc = _with_run_config(tmp_path, store)
    try:
        set_delegation_id("D003")
        set_run_config_path(str(rc))
        env = _build_session_env()
    finally:
        set_delegation_id(None)
        set_run_config_path(None)
    assert env["F3DASM_DELEGATION_ID"] == "D003"
    assert env["F3DASM_RUN_CONFIG"] == str(rc)
    assert env["F3DASM_CANONICAL_STORE"] == str(store)


def test_session_env_tolerates_missing_run_config(tmp_path):
    # No run_config bound → no store key, no crash (best-effort).
    try:
        set_delegation_id("D001")
        set_run_config_path(None)
        env = _build_session_env()
    finally:
        set_delegation_id(None)
    assert env == {"F3DASM_DELEGATION_ID": "D001"}
    assert "F3DASM_CANONICAL_STORE" not in env
