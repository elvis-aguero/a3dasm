"""Watchdog cleanup helpers (BACKLOG #11, #12): reap leftover background jobs and
leave a synthetic post-mortem retrospective on a watchdog kill.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time

from a3dasm._src.watchdog_cleanup import (
    check_memory_and_kill,
    delegation_rss,
    read_governor_pids,
    reap_governor_pids,
    reap_process_group,
    seconds_since_last_activity,
    write_watchdog_retrospective,
)


# ── stall detection: the watchdog kills a HANG, never a slow/parallel run ──────

def test_seconds_since_last_activity_tracks_writes(tmp_path):
    run_dir = tmp_path / "runs" / "T"
    (run_dir / "debug").mkdir(parents=True)
    # Empty/absent run → inf (nothing written yet, not a stall).
    assert seconds_since_last_activity(tmp_path / "nope") == float("inf")
    # A just-written file → ~0s idle (the run is live).
    (run_dir / "debug" / "diagnostics.jsonl").write_text("{}\n")
    assert seconds_since_last_activity(run_dir) < 5
    # Backdate every file → looks idle (a stall the watchdog would catch).
    old = time.time() - 9999
    import os as _os
    for p in run_dir.rglob("*"):
        _os.utime(p, (old, old))
    assert seconds_since_last_activity(run_dir) > 9000


# ── L2b/#14: memory watch + recursive reap over self-registered campaign PIDs ──

class _StubBackend:
    """rss_by_pid maps pid→bytes; start_by_pid maps pid→creation time (the live
    value the ownership check reads). Records killed pids."""
    def __init__(self, rss_by_pid=None, start_by_pid=None):
        self.rss_by_pid = rss_by_pid or {}
        self.start_by_pid = start_by_pid or {}
        self.killed = []
    def set_self_limit(self, cap):
        return False
    def read_rss(self, pids):
        return sum(self.rss_by_pid.get(p, 0) for p in pids)
    def kill(self, pids):
        self.killed.extend(pids); return len(pids)
    def proc_start_time(self, pid):
        return self.start_by_pid.get(pid, 1000.0)  # default matches recorded


def _write_pids(run_dir, mapping, start=1000.0):
    debug = run_dir / "debug"; debug.mkdir(parents=True, exist_ok=True)
    with (debug / "governor_pids.jsonl").open("w") as f:
        for did, pids in mapping.items():
            for p in pids:
                f.write(json.dumps(
                    {"delegation_id": did, "pid": p, "start_time": start}) + "\n")


def test_read_governor_pids_groups_by_delegation(tmp_path):
    _write_pids(tmp_path, {"D001": [10, 11], "D002": [20]})
    assert read_governor_pids(tmp_path) == {
        "D001": [(10, 1000.0), (11, 1000.0)], "D002": [(20, 1000.0)]}
    assert read_governor_pids(tmp_path / "nope") == {}  # missing → empty


def test_recycled_pid_is_not_killed(tmp_path):
    # Registered D001 pid 10 at start_time 1000; the LIVE pid 10 now reports a
    # DIFFERENT start time (it's a recycled, unrelated process) → must be spared.
    _write_pids(tmp_path, {"D001": [10]}, start=1000.0)
    be = _StubBackend(rss_by_pid={10: 9_000_000_000},   # "over cap"
                      start_by_pid={10: 5000.0})         # live start ≠ recorded
    killed = check_memory_and_kill(tmp_path, cap_bytes=1024 ** 3, backend=be)
    assert killed == [] and be.killed == []  # innocent bystander spared
    # And a normally-matching pid still gets killed (kill path not broken):
    be2 = _StubBackend(rss_by_pid={10: 9_000_000_000}, start_by_pid={10: 1000.0})
    assert check_memory_and_kill(tmp_path, cap_bytes=1024 ** 3, backend=be2) == ["D001"]


def test_check_memory_kills_only_the_over_cap_delegation(tmp_path):
    _write_pids(tmp_path, {"D001": [10], "D002": [20, 21]})
    # D002's tree is over a 1 GB cap; D001 is under.
    be = _StubBackend(rss_by_pid={10: 500, 20: 800_000_000, 21: 800_000_000})
    killed = check_memory_and_kill(tmp_path, cap_bytes=1024 ** 3, backend=be)
    assert killed == ["D002"]
    assert set(be.killed) == {20, 21}        # the whole D002 tree
    # audit diagnostic written
    diag = json.loads((tmp_path / "debug" / "diagnostics.jsonl").read_text().splitlines()[-1])
    assert diag["error_type"] == "MEMORY_CAP_KILL"


def test_check_memory_under_cap_kills_nothing(tmp_path):
    _write_pids(tmp_path, {"D001": [10]})
    be = _StubBackend(rss_by_pid={10: 100})
    assert check_memory_and_kill(tmp_path, cap_bytes=1024 ** 3, backend=be) == []
    assert be.killed == []


def test_reap_governor_pids_kills_all_registered(tmp_path):
    _write_pids(tmp_path, {"D001": [10, 11], "D002": [20]})
    be = _StubBackend()
    n = reap_governor_pids(tmp_path, backend=be)
    assert n == 3 and set(be.killed) == {10, 11, 20}


def test_memory_cap_actually_kills_a_real_over_cap_process(tmp_path):
    """End-to-end proof of the RSS-based hard cap: a real subprocess that holds
    ~300 MB resident, registered with its true start time, is killed by
    check_memory_and_kill under a 100 MB cap — via the real psutil backend."""
    import pytest
    psutil = pytest.importorskip("psutil")
    # Hold ~300 MB resident (bytearray is zero-filled → resident), then idle.
    child = subprocess.Popen([
        sys.executable, "-c",
        "x=bytearray(300*1024*1024); import time; time.sleep(120)",
    ])
    try:
        time.sleep(1.5)  # let it allocate
        start = psutil.Process(child.pid).create_time()  # the TRUE ownership token
        debug = tmp_path / "debug"; debug.mkdir(parents=True)
        (debug / "governor_pids.jsonl").write_text(json.dumps(
            {"delegation_id": "D001", "pid": child.pid, "start_time": start}) + "\n")
        killed = check_memory_and_kill(tmp_path, cap_bytes=100 * 1024 ** 2)  # real backend
        assert killed == ["D001"]
        for _ in range(40):
            if child.poll() is not None:
                break
            time.sleep(0.1)
        assert child.poll() is not None, "over-cap process was not killed"
    finally:
        if child.poll() is None:
            child.kill()


def test_delegation_rss_sums_the_delegations_tree(tmp_path):
    _write_pids(tmp_path, {"D001": [10, 11], "D002": [20]})
    be = _StubBackend(rss_by_pid={10: 100, 11: 200, 20: 999})
    assert delegation_rss(tmp_path, "D001", backend=be) == 300  # 10+11 only
    assert delegation_rss(tmp_path, "D002", backend=be) == 999
    assert delegation_rss(tmp_path, "D999", backend=be) == 0   # unknown → 0


# ── #11: reap leftover background processes ──────────────────────────────────

def test_reap_kills_a_detached_background_process():
    # start_new_session=True puts the child in its OWN session/group (pgid==pid),
    # mirroring a leftover detached campaign — and keeps the test runner's group
    # untouched.
    child = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(300)"],
        start_new_session=True,
    )
    try:
        assert child.poll() is None  # running
        reap_process_group(child.pid)  # pid == pgid for a session leader
        for _ in range(40):  # up to ~4s for SIGTERM to land
            if child.poll() is not None:
                break
            time.sleep(0.1)
        assert child.poll() is not None, "background process was not reaped"
    finally:
        if child.poll() is None:
            child.kill()


def test_reap_unknown_pgid_is_silent():
    # A pgid with no live processes must not raise.
    reap_process_group(999_999)


# ── #12: synthetic watchdog retrospective ────────────────────────────────────

def test_write_watchdog_retrospective_from_disk_state(tmp_path):
    debug = tmp_path / "debug"
    debug.mkdir()
    (debug / "delegation_log.jsonl").write_text(
        json.dumps({"id": "D001", "to_node": "datagenerator", "status": "DONE"}) + "\n"
        + json.dumps({"id": "D004", "to_node": "implementer", "status": "RUNNING"}) + "\n"
    )
    (debug / "diagnostics.jsonl").write_text(
        json.dumps({"error_type": "MILESTONE_BLOCK"}) + "\n"
        + json.dumps({"error_type": "ERROR_RETURN"}) + "\n"
    )
    write_watchdog_retrospective(tmp_path, 3600)
    lines = [l for l in (debug / "retrospectives.jsonl").read_text().splitlines() if l.strip()]
    assert len(lines) == 1
    rec = json.loads(lines[0])
    assert rec["role"] == "watchdog" and rec["source_id"] == "WATCHDOG"
    assert "WATCHDOG POST-MORTEM" in rec["text"]
    assert "D004" in rec["text"] and "implementer:RUNNING" in rec["text"]
    assert "MILESTONE_BLOCK" in rec["text"]
    assert "strategizer" in rec["text"]  # points to the transcript


def test_write_watchdog_retrospective_appends_not_overwrites(tmp_path):
    debug = tmp_path / "debug"
    debug.mkdir()
    (debug / "retrospectives.jsonl").write_text(
        json.dumps({"source_id": "D001", "role": "datagenerator", "text": "x"}) + "\n"
    )
    write_watchdog_retrospective(tmp_path, 3600)
    lines = [l for l in (debug / "retrospectives.jsonl").read_text().splitlines() if l.strip()]
    assert len(lines) == 2  # original kept, watchdog appended
    assert json.loads(lines[-1])["role"] == "watchdog"


def test_write_watchdog_retrospective_never_raises_on_missing_dir(tmp_path):
    # No debug/ artifacts yet — must still write a (sparse) entry without raising.
    write_watchdog_retrospective(tmp_path, 3600)
    rec = json.loads((tmp_path / "debug" / "retrospectives.jsonl").read_text().splitlines()[0])
    assert rec["role"] == "watchdog" and "(none)" in rec["text"]


# ── resource AWARENESS (telemetry): peak RSS high-water + static envelope ─────

def test_peak_rss_high_water_rides_the_existing_poll(tmp_path):
    """check_memory_and_kill records each delegation's peak tree-RSS as a free
    byproduct of its enforcement read — max across ticks, never decreasing."""
    from a3dasm._src import watchdog_cleanup as wc
    wc._PEAK_RSS.clear()
    _write_pids(tmp_path, {"D001": [10]})
    cap = 8 * 1024 ** 3  # 8 GB cap — under it, so nothing is killed
    # tick 1: 100 MB
    wc.check_memory_and_kill(tmp_path, cap, backend=_StubBackend({10: 100 * 1024**2}))
    assert wc.delegation_peak_rss("D001") == 100 * 1024**2
    # tick 2: 300 MB → peak rises
    wc.check_memory_and_kill(tmp_path, cap, backend=_StubBackend({10: 300 * 1024**2}))
    assert wc.delegation_peak_rss("D001") == 300 * 1024**2
    # tick 3: 50 MB → peak HOLDS (high-water, not current)
    wc.check_memory_and_kill(tmp_path, cap, backend=_StubBackend({10: 50 * 1024**2}))
    assert wc.delegation_peak_rss("D001") == 300 * 1024**2
    assert wc.delegation_peak_rss("D999") == 0  # unknown delegation


def test_resource_envelope_is_O1_no_directory_walk(tmp_path, monkeypatch):
    """resource_envelope must be a single statvfs (shutil.disk_usage) + cpu_count,
    NEVER a recursive walk — this is the explicit footprint requirement."""
    import shutil as _sh
    from a3dasm._src import watchdog_cleanup as wc

    calls = {"disk_usage": 0}
    real = _sh.disk_usage

    def _spy(path):
        calls["disk_usage"] += 1
        return real(path)
    monkeypatch.setattr(wc.shutil, "disk_usage", _spy)

    env = wc.resource_envelope(tmp_path, 4 * 1024 ** 3)
    assert calls["disk_usage"] == 1, "disk must be ONE statvfs, not a walk"
    assert env["cores"] and env["cores"] > 0
    assert env["ram_cap_bytes"] == 4 * 1024 ** 3
    assert env["disk_free_bytes"] is not None and env["disk_free_bytes"] > 0


def test_resource_envelope_never_raises_on_bad_path():
    from a3dasm._src.watchdog_cleanup import resource_envelope
    env = resource_envelope("/nonexistent/path/xyz", None)
    assert env["ram_cap_bytes"] is None  # cap unset → None, no crash
    assert "cores" in env


def test_usable_cores_respects_cpu_affinity(monkeypatch):
    """The cores fact given to agents must be what THIS process may use (a
    SLURM/cgroup cpuset), not the host total — os.sched_getaffinity, not the
    old os.cpu_count() that reported 48 while the job was pinned to 1."""
    import a3dasm._src.watchdog_cleanup as wc
    monkeypatch.setattr(wc.os, "sched_getaffinity", lambda pid: {0, 1, 2},
                        raising=False)
    assert wc._usable_cores() == 3          # cpuset size, not the host's total


def test_usable_cores_falls_back_without_affinity(monkeypatch):
    """Off-Linux (no sched_getaffinity) falls back to os.cpu_count()."""
    import a3dasm._src.watchdog_cleanup as wc
    monkeypatch.delattr(wc.os, "sched_getaffinity", raising=False)
    monkeypatch.setattr(wc.os, "cpu_count", lambda: 8)
    assert wc._usable_cores() == 8
