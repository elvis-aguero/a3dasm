"""AgenticRun._maybe_start_slurm_llm — the execute() wiring, fully headless.

Every SLURM/HTTP call in slurm_llm is stubbed, so this needs no cluster. Covers:
disabled -> no-op/None; enabled -> submit + wait + publish VLLM_BASE_URL +
persist jobid + return it for teardown.
"""
from __future__ import annotations

import logging
from pathlib import Path

from a3dasm._src import slurm_llm
from a3dasm._src.agent_runtime import DEFAULT_MODEL, AgenticRun


def _bare_run(tmp_path: Path, backend: str = "vllm") -> AgenticRun:
    run = AgenticRun.__new__(AgenticRun)
    run.study_dir = tmp_path
    run._model = DEFAULT_MODEL
    run._backend = backend
    return run


def _log():
    return logging.getLogger("test.slurm_llm_wiring")


def test_disabled_is_a_noop(tmp_path, monkeypatch):
    monkeypatch.delenv("VLLM_BASE_URL", raising=False)
    run = _bare_run(tmp_path)
    debug = tmp_path / "debug"
    debug.mkdir()
    # No llm_slurm block, and an explicitly-disabled one, both -> None.
    assert run._maybe_start_slurm_llm({}, debug, _log()) is None
    assert run._maybe_start_slurm_llm(
        {"llm_slurm": {"enabled": False}}, debug, _log()) is None
    assert "VLLM_BASE_URL" not in __import__("os").environ


def test_enabled_submits_waits_and_publishes(tmp_path, monkeypatch):
    monkeypatch.delenv("VLLM_BASE_URL", raising=False)
    calls = {}
    def _submit(p):
        calls["script"] = p
        return "77"
    monkeypatch.setattr(slurm_llm, "submit_serve_job", _submit)
    monkeypatch.setattr(slurm_llm, "wait_until_running",
                        lambda jid, t, **k: "gpu007")
    monkeypatch.setattr(slurm_llm, "wait_until_ready",
                        lambda url, t, **k: calls.setdefault("ready_url", url))

    run = _bare_run(tmp_path, backend="vllm")
    debug = tmp_path / "debug"
    debug.mkdir()
    cfg = {"llm_slurm": {"enabled": True, "model": "gemma-4-9b",
                         "params_b": 9, "gpu_model": "a100",
                         "cluster": {"partition": "gpu", "account": "acct"}}}

    jobid = run._maybe_start_slurm_llm(cfg, debug, _log())

    assert jobid == "77"
    # published endpoint points at the granted node + profile port (8000)
    import os
    assert os.environ["VLLM_BASE_URL"] == "http://gpu007:8000/v1"
    assert calls["ready_url"] == "http://gpu007:8000/v1"
    # jobid persisted for the watchdog reaper; script written
    assert (debug / "serve_job.jobid").read_text() == "77"
    assert (debug / "vllm_serve.sh").exists()
    assert "--model gemma-4-9b" in (debug / "vllm_serve.sh").read_text()
    monkeypatch.delenv("VLLM_BASE_URL", raising=False)


def test_submit_failure_propagates_no_silent_fallback(tmp_path, monkeypatch):
    def boom(_p):
        raise RuntimeError("sbatch: no such partition")
    monkeypatch.setattr(slurm_llm, "submit_serve_job", boom)
    run = _bare_run(tmp_path)
    debug = tmp_path / "debug"
    debug.mkdir()
    cfg = {"llm_slurm": {"enabled": True, "model": "gemma-4-9b"}}
    try:
        run._maybe_start_slurm_llm(cfg, debug, _log())
    except RuntimeError as e:
        assert "sbatch" in str(e)
    else:
        raise AssertionError("expected the serve-start failure to propagate")
