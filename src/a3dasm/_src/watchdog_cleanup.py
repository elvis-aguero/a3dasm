"""Best-effort cleanup a study's run watchdog performs before ``os._exit``.

Two jobs, both observed-necessary on the n=5 watchdog post-mortems:
  1. ``reap_process_group`` — kill leftover background jobs the run spawned (e.g.
     an implementer's detached optimisation campaign) so they don't outlive the
     watchdog and keep burning CPU.
  2. ``write_watchdog_retrospective`` — a watchdog kill is abrupt, so the
     strategizer never writes its own end-of-run retrospective and §1 Step 1 is
     blind. Leave a synthetic, clearly-labelled post-mortem in
     ``retrospectives.jsonl`` from disk state.

Both are best-effort and never raise — a watchdog must still exit.
"""
from __future__ import annotations

import json
import os
import shutil
import signal
import threading
from datetime import datetime, timezone
from pathlib import Path

# ── resource AWARENESS (telemetry, not enforcement) ──────────────────────────
# Per-delegation peak process-tree RSS, recorded as a free byproduct of the
# memory watcher's existing 5s poll (check_memory_and_kill already computes each
# delegation's tree RSS to enforce the cap). A high-water max() per tick — no new
# poll, no I/O, no thread. The watcher and the readers (delegation footer,
# GetStatus) live in the SAME process, so this in-memory dict bridges them
# without a file. Keyed by delegation_id (one run per process).
_PEAK_RSS: dict[str, int] = {}
_PEAK_LOCK = threading.Lock()


def reap_process_group(pgid: int) -> None:
    """SIGTERM every process in ``pgid`` (the run's process group). run.py makes
    itself a group leader at startup, so its descendants — the SDK ``claude`` CLI,
    the agent's Bash, any backgrounded script — share its group. The caller should
    ``signal.signal(SIGTERM, SIG_IGN)`` in itself FIRST so it survives this group
    signal and still reaches ``os._exit`` with the intended code. Best-effort.

    Residual limit: a descendant that started a NEW session (setsid) escapes the
    group and is not reached; default ``subprocess``/``&`` do not, so the common
    leftover (a detached campaign) is covered.
    """
    try:
        os.killpg(pgid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError, OSError):
        pass


# ── per-delegation memory watch + recursive reap (resource-governance L2b/#14) ──
# These read the campaign PIDs that get_evaluator self-registered
# (<run_dir>/debug/governor_pids.jsonl) and use the ResourceBackend's RECURSIVE
# tree kill — which catches the new-session/detached campaigns that os.killpg
# misses (the #14 escape) regardless of how the agent's Bash launched them.

def read_governor_pids(run_dir) -> dict[str, list[tuple[int, float | None]]]:
    """Parse governor_pids.jsonl into ``{delegation_id: [(pid, start_time), ...]}``.
    ``start_time`` is the process creation time recorded at registration (or None);
    it is the ownership token the kill path verifies. Empty on any error."""
    out: dict[str, list[tuple[int, float | None]]] = {}
    try:
        reg = Path(run_dir) / "debug" / "governor_pids.jsonl"
        if not reg.exists():
            return out
        for ln in reg.read_text().splitlines():
            ln = ln.strip()
            if not ln:
                continue
            try:
                r = json.loads(ln)
            except Exception:
                continue
            did, pid = r.get("delegation_id"), r.get("pid")
            if did is not None and isinstance(pid, int):
                out.setdefault(did, []).append((pid, r.get("start_time")))
    except Exception:
        pass
    return out


def _owned_pids(entries, be) -> list[int]:
    """From ``[(pid, recorded_start), ...]`` return only the pids that are STILL
    our campaign: the live process's start time matches what we recorded (±1s). A
    RECYCLED pid (a different program that inherited the number), a dead pid, or
    one we can't verify is EXCLUDED — so we never signal an innocent bystander.
    Fail-safe: when in doubt, don't kill."""
    owned: list[int] = []
    for pid, recorded in entries:
        cur = be.proc_start_time(pid)
        if cur is None:
            continue  # gone / unverifiable → not ours
        if recorded is None or abs(cur - recorded) < 1.0:
            owned.append(pid)
    return owned


def check_memory_and_kill(run_dir, cap_bytes: int, backend=None) -> list[str]:
    """Sample each registered delegation's process-tree RSS and kill any tree over
    ``cap_bytes`` — the authoritative, REAL-USAGE memory enforcer. Verifies
    ownership (start-time match) before reading/killing, so a recycled pid is
    never touched. Returns the delegation IDs killed. Best-effort — never raises."""
    killed: list[str] = []
    if not cap_bytes or cap_bytes <= 0:
        return killed
    try:
        from .resource_backend import get_resource_backend
        be = backend or get_resource_backend()
        for did, entries in read_governor_pids(run_dir).items():
            try:
                owned = _owned_pids(entries, be)
                if not owned:
                    continue
                rss = be.read_rss(owned)
                # Record the peak as a free byproduct of this enforcement read.
                with _PEAK_LOCK:
                    if rss > _PEAK_RSS.get(did, 0):
                        _PEAK_RSS[did] = rss
                if rss > cap_bytes:
                    be.kill(owned)
                    killed.append(did)
            except Exception:
                pass
        if killed:
            _log_memory_kill(run_dir, killed, cap_bytes)
    except Exception:
        pass
    return killed


def reap_governor_pids(run_dir, backend=None) -> int:
    """Recursively kill every registered campaign's process tree — the watchdog's
    catch-all for detached/new-session campaigns the process-group kill misses.
    Verifies ownership first (never signals a recycled pid). Returns processes
    signalled. Best-effort."""
    try:
        from .resource_backend import get_resource_backend
        be = backend or get_resource_backend()
        owned = [
            p for entries in read_governor_pids(run_dir).values()
            for p in _owned_pids(entries, be)
        ]
        return be.kill(owned) if owned else 0
    except Exception:
        return 0


def seconds_since_last_activity(run_dir) -> float:
    """Wall-seconds since the run last wrote ANYTHING under ``run_dir`` — the
    run's liveness signal.

    A live run constantly writes: a strategizer turn streams transcripts, the
    canonical ledger flushes evaluations, the delegation log and diagnostics
    advance. A genuinely hung run (a frozen LLM/CLI call, a zombie subprocess)
    writes nothing. So "no file has changed for a long window" is a true STALL,
    whereas a slow-but-busy or heavily-parallel run keeps this number small —
    which is why a stall watchdog never penalises parallelism the way a flat
    wall-clock deadline does.

    Returns ``float('inf')`` if the run dir is empty/absent (nothing written
    yet). Best-effort — never raises.
    """
    run_dir = Path(run_dir)
    latest = 0.0
    try:
        for p in run_dir.rglob("*"):
            try:
                m = p.stat().st_mtime
                if m > latest:
                    latest = m
            except OSError:
                continue
    except OSError:
        pass
    if latest == 0.0:
        return float("inf")
    import time as _t
    return max(0.0, _t.time() - latest)


def delegation_rss(run_dir, delegation_id: str, backend=None) -> int:
    """Total process-tree RSS (bytes) of one delegation's OWNED campaign(s), for
    GetStatus telemetry so the strategizer can SEE a fat delegation. 0 if none /
    unknown. Best-effort — never raises."""
    try:
        from .resource_backend import get_resource_backend
        be = backend or get_resource_backend()
        owned = _owned_pids(read_governor_pids(run_dir).get(delegation_id, []), be)
        return be.read_rss(owned) if owned else 0
    except Exception:
        return 0


def delegation_peak_rss(delegation_id: str) -> int:
    """Peak process-tree RSS (bytes) seen for one delegation across the memory
    watcher's 5s ticks — its high-water memory footprint, for the KPI footer and
    GetStatus. 0 if the watcher never sampled it (a sub-tick or off-ledger
    delegation). Lower bound: a spike between two ticks is missed."""
    with _PEAK_LOCK:
        return _PEAK_RSS.get(str(delegation_id), 0)


def _usable_cores() -> int | None:
    """CPUs this PROCESS may actually use — not the host's total. Respects a
    SLURM/cgroup cpuset via ``os.sched_getaffinity`` (Linux); falls back to
    ``os.cpu_count()`` where affinity is unavailable (macOS/Windows dev). The old
    ``os.cpu_count()`` reported host cores (e.g. 48) even when the job was pinned
    to 1, mis-priming agents to oversubscribe parallel evaluations."""
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count()


def resource_envelope(run_dir, mem_cap_bytes: int | None) -> dict:
    """The static resource envelope surfaced to an agent at delegation start —
    "what you HAVE": ``{cores, ram_cap_bytes, disk_free_bytes}``.

    All O(1): the usable-core count (``os.sched_getaffinity`` — the CPUs THIS
    process may use under a SLURM/cgroup cpuset — falling back to
    ``os.cpu_count()`` off-Linux), the passed cap (no syscall), and
    ``shutil.disk_usage(run_dir).free`` (a single ``statvfs`` — NOT a recursive
    walk). ``disk_free_bytes`` is the host filesystem's free space, not a per-run
    quota. Never raises; fields fall back to None on error."""
    cores = _usable_cores()
    disk_free = None
    try:
        disk_free = shutil.disk_usage(str(run_dir)).free
    except OSError:
        pass
    return {
        "cores": cores,
        "ram_cap_bytes": int(mem_cap_bytes) if mem_cap_bytes else None,
        "disk_free_bytes": disk_free,
    }


def _log_memory_kill(run_dir, killed: list[str], cap_bytes: int) -> None:
    """Append a MEMORY_CAP_KILL diagnostic so the kill is auditable. Best-effort."""
    try:
        debug = Path(run_dir) / "debug"
        if not debug.exists():
            return
        rec = {
            "ts": datetime.now(tz=timezone.utc).isoformat(timespec="seconds"),
            "node": ",".join(killed),
            "error_type": "MEMORY_CAP_KILL",
            "message": f"killed {killed} over hard mem cap {cap_bytes} bytes",
        }
        with (debug / "diagnostics.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
    except Exception:
        pass


def write_watchdog_retrospective(run_dir, watchdog_seconds: int) -> None:
    """Append a synthetic post-mortem to ``<run_dir>/debug/retrospectives.jsonl``.

    The record uses the same shape as a real node retrospective (ts, source_id,
    role, flagged, text) but is honestly labelled ``role="watchdog"`` — it is NOT
    fabricated first-person text. It surfaces the last delegation state and the
    diagnostics tally (disk-only) plus a pointer to the strategizer transcript, so
    a watchdog-killed run gives §1 Step 1 a breadcrumb instead of silence.
    """
    try:
        debug = Path(run_dir) / "debug"
        delg: dict = {}
        dl = debug / "delegation_log.jsonl"
        if dl.exists():
            for ln in dl.read_text().splitlines():
                ln = ln.strip()
                if not ln:
                    continue
                try:
                    r = json.loads(ln)
                except Exception:
                    continue
                delg[r.get("id")] = f"{r.get('to_node')}:{r.get('status')}"

        diag: dict = {}
        dg = debug / "diagnostics.jsonl"
        if dg.exists():
            for ln in dg.read_text().splitlines():
                ln = ln.strip()
                if not ln:
                    continue
                try:
                    r = json.loads(ln)
                except Exception:
                    continue
                k = r.get("error_type") or r.get("type") or "other"
                diag[k] = diag.get(k, 0) + 1

        text = (
            "## WATCHDOG POST-MORTEM\n"
            f"Run force-killed at {watchdog_seconds}s — no clean close, so the "
            "strategizer wrote no first-person retrospective. Reconstruct its "
            "reasoning from debug/transcripts/strategizer/ (CLAUDE.md §1 Step 4).\n"
            f"Last delegation state: {delg or '(none)'}\n"
            f"Diagnostics: {diag or '(none)'}"
        )
        rec = {
            "ts": datetime.now(tz=timezone.utc).isoformat(timespec="seconds"),
            "source_id": "WATCHDOG",
            "role": "watchdog",
            "flagged": False,
            "text": text,
        }
        debug.mkdir(parents=True, exist_ok=True)
        with (debug / "retrospectives.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
    except Exception:
        pass
