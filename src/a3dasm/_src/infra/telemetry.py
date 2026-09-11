"""Separable, analysis-ready LLM-call telemetry.

This subsystem is deliberately *additive* and *off the decision path*: it
records one row per LLM call and merges them into a ``summary.json`` for
post-hoc analysis (ablations — "where did the budget go; does the machinery
improve outcomes").  A telemetry write must NEVER break a run, so every
``record_call`` swallows its own errors.

Layout (under ``<run>/debug/telemetry/``):
  - ``calls.<pid>.jsonl``  — one JSON row per LLM call, one file per process.
    Per-PID files keep parallel worker *processes* from corrupting a shared
    file; within a process an instance lock serialises the daemon worker
    threads' appends.
  - ``summary.json``       — written by :meth:`Telemetry.merge`, with totals
    and breakdowns by role / phase / model.

A row carries the same token fields the run already accumulates
(``adapter.last_usage``) plus ``role`` / ``model`` / ``phase`` /
``delegation_id`` / ``ts`` so each call is attributable.
"""
from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Optional

# Token fields copied verbatim from adapter.last_usage.
_TOKEN_FIELDS = (
    "input_tokens",
    "output_tokens",
    "cache_read_input_tokens",
    "cache_creation_input_tokens",
)


class Telemetry:
    """Per-run telemetry writer.  One instance per :class:`StrategizerNode`."""

    def __init__(self, debug_dir: Any) -> None:
        self._dir = Path(debug_dir) / "telemetry"
        self._lock = threading.Lock()
        self._path = self._dir / f"calls.{os.getpid()}.jsonl"

    # -- recording -----------------------------------------------------------

    def record_call(
        self,
        *,
        role: Optional[str],
        model: Optional[str],
        phase: Optional[str],
        delegation_id: Optional[str],
        usage: Optional[dict],
    ) -> None:
        """Append one row for a single LLM call.

        Never raises into the caller: a telemetry failure must not break the
        run.  ``usage`` is ``adapter.last_usage`` (may be empty / partial /
        have ``total_cost_usd=None`` under ollama — all tolerated).
        """
        try:
            usage = usage or {}
            row = {
                "role": role,
                "model": model,
                "phase": phase,
                "delegation_id": delegation_id,
                "ts": time.time(),
            }
            for f in _TOKEN_FIELDS:
                row[f] = usage.get(f, 0) or 0
            # cost is the one field that stays None under ollama (never faked)
            row["total_cost_usd"] = usage.get("total_cost_usd")
            self._append_row(row)
        except Exception:  # noqa: BLE001 — telemetry is best-effort
            pass

    def _append_row(self, row: dict) -> None:
        self._dir.mkdir(parents=True, exist_ok=True)
        line = json.dumps(row) + "\n"
        with self._lock:
            with open(self._path, "a", encoding="utf-8") as fh:
                fh.write(line)

    # -- aggregation ---------------------------------------------------------

    @staticmethod
    def merge(debug_dir: Any) -> dict:
        """Union all ``calls.*.jsonl`` files → ``summary.json``; return it.

        Robust to a missing telemetry dir (returns an empty summary) and to
        malformed lines (skipped).  Breakdowns by role / phase / model each
        partition the totals.
        """
        tdir = Path(debug_dir) / "telemetry"
        rows: list[dict] = []
        if tdir.is_dir():
            for f in sorted(tdir.glob("calls.*.jsonl")):
                try:
                    for line in f.read_text(encoding="utf-8").splitlines():
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            rows.append(json.loads(line))
                        except (ValueError, json.JSONDecodeError):
                            continue
                except OSError:
                    continue

        def _bucket() -> dict:
            return {
                "calls": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "cache_read_input_tokens": 0,
                "cache_creation_input_tokens": 0,
                "total_cost_usd": 0.0,
            }

        def _add(b: dict, r: dict) -> None:
            b["calls"] += 1
            for f in _TOKEN_FIELDS:
                b[f] += int(r.get(f, 0) or 0)
            cost = r.get("total_cost_usd")
            if cost is not None:
                b["total_cost_usd"] += cost

        totals = _bucket()
        by_role: dict = {}
        by_phase: dict = {}
        by_model: dict = {}
        tss = []
        for r in rows:
            _add(totals, r)
            _add(by_role.setdefault(r.get("role"), _bucket()), r)
            _add(by_phase.setdefault(r.get("phase"), _bucket()), r)
            _add(by_model.setdefault(r.get("model"), _bucket()), r)
            ts = r.get("ts")
            if isinstance(ts, (int, float)):
                tss.append(ts)

        totals["total_tokens"] = (
            totals["input_tokens"] + totals["output_tokens"]
        )
        totals["wall_time_s"] = (max(tss) - min(tss)) if len(tss) >= 2 else 0.0

        summary = {
            "totals": totals,
            "by_role": by_role,
            "by_phase": by_phase,
            "by_model": by_model,
        }
        try:
            tdir.mkdir(parents=True, exist_ok=True)
            (tdir / "summary.json").write_text(
                json.dumps(summary, indent=2), encoding="utf-8"
            )
        except OSError:
            pass
        return summary
