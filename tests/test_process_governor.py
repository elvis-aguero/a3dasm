"""Commit 4: at the oracle entry a campaign process applies the hard memory cap
and registers its PID (so the run's watcher can find + reap its tree regardless of
how the agent's Bash launched it). Idempotent per process; best-effort."""
from __future__ import annotations

import json
import os

import a3dasm._src.instrumented as inst


def _reset(monkeypatch, spy):
    monkeypatch.setattr(inst, "_GOVERNOR_PID_APPLIED", False)
    monkeypatch.setattr(
        "a3dasm._src.resource_backend.get_resource_backend",
        lambda: spy,
    )


class _SpyBackend:
    def __init__(self):
        self.cap_calls = []
    def set_self_limit(self, cap):
        self.cap_calls.append(cap); return False  # no self-cap; watcher enforces
    def read_rss(self, pids):
        return 0
    def kill(self, pids):
        return 0
    def proc_start_time(self, pid):
        return 12345.0  # recorded into governor_pids for the ownership check


def test_applies_cap_and_registers_pid(tmp_path, monkeypatch):
    spy = _SpyBackend()
    _reset(monkeypatch, spy)
    store_dir = tmp_path / "experiment_data"
    (tmp_path / "debug").mkdir(parents=True)
    inst._apply_process_governor(
        {"mem_cap_bytes": 4 * 1024 ** 3}, store_dir, "D002")
    # hard cap applied to this process
    assert spy.cap_calls == [4 * 1024 ** 3]
    # PID registered under the delegation
    reg = tmp_path / "debug" / "governor_pids.jsonl"
    rec = json.loads(reg.read_text().splitlines()[-1])
    assert rec["delegation_id"] == "D002" and rec["pid"] == os.getpid()
    assert rec["start_time"] == 12345.0  # recorded for the ownership check


def test_idempotent_per_process(tmp_path, monkeypatch):
    spy = _SpyBackend()
    _reset(monkeypatch, spy)
    store_dir = tmp_path / "experiment_data"
    (tmp_path / "debug").mkdir(parents=True)
    for _ in range(3):
        inst._apply_process_governor({"mem_cap_bytes": 1}, store_dir, "D002")
    assert len(spy.cap_calls) == 1  # only the first call acts
    reg = tmp_path / "debug" / "governor_pids.jsonl"
    assert len(reg.read_text().splitlines()) == 1


def test_no_cap_no_setlimit_but_still_registers(tmp_path, monkeypatch):
    spy = _SpyBackend()
    _reset(monkeypatch, spy)
    store_dir = tmp_path / "experiment_data"
    (tmp_path / "debug").mkdir(parents=True)
    inst._apply_process_governor({}, store_dir, "D003")  # no mem_cap_bytes
    assert spy.cap_calls == []  # nothing to cap
    reg = tmp_path / "debug" / "governor_pids.jsonl"
    assert json.loads(reg.read_text().splitlines()[-1])["delegation_id"] == "D003"


def test_never_raises_without_debug_dir(tmp_path, monkeypatch):
    _reset(monkeypatch, _SpyBackend())
    # no debug/ dir → registration skipped, no raise
    inst._apply_process_governor({"mem_cap_bytes": 1}, tmp_path / "experiment_data", "D001")
