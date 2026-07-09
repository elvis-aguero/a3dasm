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
    assert env["F3DASM_DELEGATION_ID"] == "D001"
    assert "F3DASM_CANONICAL_STORE" not in env
    # PATH is always injected (run interpreter prepended); see the dedicated
    # test below. No other keys beyond delegation id + PATH here.
    assert set(env) <= {"F3DASM_DELEGATION_ID", "PATH"}


def test_session_env_prepends_run_interpreter_to_path():
    """The agent's shell must use the SAME interpreter as the agent loop, so
    `python` in Bash can import the framework + study deps. Regression for the
    e2e finding where bash `python` resolved to a system Python without the
    package, forcing off-ledger workarounds. The run interpreter's bin dir must
    be prepended to PATH (not replace the inherited PATH)."""
    import os
    import sys

    env = _build_session_env()
    run_bin = os.path.dirname(sys.executable)
    assert "PATH" in env
    parts = env["PATH"].split(os.pathsep)
    assert parts[0] == run_bin, f"run interpreter bin must be first: {parts[:2]}"
    # inherited PATH preserved (not clobbered)
    assert env["PATH"].endswith(os.environ.get("PATH", "")) or \
        os.environ.get("PATH", "") in env["PATH"]
