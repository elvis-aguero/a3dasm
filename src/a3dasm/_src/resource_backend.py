"""Single abstraction for OS-specific resource governance — so memory limits,
RSS telemetry, and process-tree reaping live in ONE place instead of being
sprinkled as ``if darwin / elif linux`` across instrumented.py and run.py.

The rest of the codebase uses ``get_resource_backend()`` and the three-method
interface only; swapping the implementation (e.g. a cgroup-native backend on a
Linux/HPC deploy) is a one-line change here, not a sweep through call sites.

Backends:
  - ``PsutilBackend`` (default, cross-platform: Linux / macOS / Windows) — RSS via
    psutil, kill via the recursive process tree (enforcement is RSS-based, no
    RLIMIT_AS — see ResourceBackend below).
  - ``StdlibBackend`` (degraded fallback if psutil is absent) — best-effort.
  - (future) a cgroup-native ``LinuxBackend`` for HPC — reads ``memory.current`` /
    sets ``memory.max`` and cooperates with SLURM's job cgroup. The interface
    below is the seam; not built until there's a cluster to verify it on.
"""
from __future__ import annotations

import abc
import os
import signal
from collections.abc import Sequence


class ResourceBackend(abc.ABC):
    """Governs the memory + lifetime of a delegation's process tree.

    The authoritative memory enforcer is the RSS watcher (``read_rss`` →
    ``kill``), which measures REAL resident memory. We deliberately do NOT impose
    a per-process ``RLIMIT_AS`` ceiling: that caps RESERVED virtual address space,
    which scientific libs (numpy/BLAS/multiprocessing) routinely over-reserve
    without using — so it false-kills legitimate campaigns. A real per-process
    hard cap on RESIDENT memory is a cgroup-backend capability (memory.max),
    added at HPC deploy behind this same interface.
    """

    @abc.abstractmethod
    def set_self_limit(self, cap_bytes: int) -> bool:
        """Apply a per-process hard cap to the CURRENT process. Returns True only
        if a backend actually self-enforces (a future cgroup backend will).
        psutil/stdlib backends do NOT self-cap — they return False, and the RSS
        watcher is the sole, real-usage-based enforcer."""

    @abc.abstractmethod
    def read_rss(self, pids: Sequence[int]) -> int:
        """Total RESIDENT memory (bytes) of the given root pids PLUS their
        descendants. 0 if unknowable (degraded backend) or all gone."""

    @abc.abstractmethod
    def kill(self, pids: Sequence[int]) -> int:
        """Terminate the given root pids AND all descendants (SIGTERM→SIGKILL),
        regardless of session/new-session (the reap the process-group kill
        misses). Returns the number of processes signalled."""

    @abc.abstractmethod
    def proc_start_time(self, pid: int) -> float | None:
        """Process creation time (epoch seconds) for ownership verification —
        the watcher compares this against the value recorded at registration so a
        RECYCLED pid (a different program that inherited the number) is never
        killed. None if the process is gone or the start time is unreadable."""


class PsutilBackend(ResourceBackend):
    """Cross-platform backend backed by psutil (a declared agentic dep)."""

    def __init__(self) -> None:
        import psutil  # noqa: F401 — fail loud if selected without psutil
        self._psutil = psutil

    def set_self_limit(self, cap_bytes: int) -> bool:
        # No per-process self-cap: RLIMIT_AS would cap reserved virtual space (the
        # wrong metric → false-kills). The RSS watcher is the enforcer.
        return False

    def proc_start_time(self, pid: int) -> float | None:
        try:
            return self._psutil.Process(pid).create_time()
        except self._psutil.Error:
            return None

    def _tree(self, pids: Sequence[int]):
        """Yield live psutil.Process for each root pid + its recursive children,
        de-duplicated."""
        seen: set[int] = set()
        for pid in pids:
            try:
                proc = self._psutil.Process(pid)
            except self._psutil.Error:
                continue
            for p in (proc, *proc.children(recursive=True)):
                if p.pid not in seen:
                    seen.add(p.pid)
                    yield p

    def read_rss(self, pids: Sequence[int]) -> int:
        total = 0
        for p in self._tree(pids):
            try:
                total += p.memory_info().rss
            except self._psutil.Error:
                pass
        return total

    def kill(self, pids: Sequence[int]) -> int:
        procs = list(self._tree(pids))
        for p in procs:
            try:
                p.terminate()
            except self._psutil.Error:
                pass
        _gone, alive = self._psutil.wait_procs(procs, timeout=3)
        for p in alive:
            try:
                p.kill()
            except self._psutil.Error:
                pass
        return len(procs)


class StdlibBackend(ResourceBackend):
    """Degraded fallback when psutil is unavailable. No self-cap; RSS telemetry
    unavailable (returns 0, so the watcher can't enforce); kill is shallow (no
    recursive descendant discovery) and ownership cannot be verified."""

    def set_self_limit(self, cap_bytes: int) -> bool:
        return False  # no self-cap; the RSS watcher (psutil) is the enforcer

    def read_rss(self, pids: Sequence[int]) -> int:
        return 0  # no portable stdlib way to sum a process tree's RSS

    def kill(self, pids: Sequence[int]) -> int:
        n = 0
        for pid in pids:
            try:
                os.kill(pid, signal.SIGTERM)
                n += 1
            except OSError:
                pass
        return n

    def proc_start_time(self, pid: int) -> float | None:
        return None  # ownership verification needs psutil; unavailable here


_BACKEND: ResourceBackend | None = None


def get_resource_backend() -> ResourceBackend:
    """The process-wide resource backend, selected ONCE: psutil if importable
    (the normal path — it's a declared agentic dep), else the degraded stdlib
    fallback. Callers use only this + the three-method interface."""
    global _BACKEND
    if _BACKEND is None:
        try:
            _BACKEND = PsutilBackend()
        except Exception:  # noqa: BLE001 — psutil missing/broken → degrade
            _BACKEND = StdlibBackend()
    return _BACKEND
