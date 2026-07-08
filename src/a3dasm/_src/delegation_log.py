"""Graph-wide append-only delegation log for agentic runs.

Stores all inter-node delegations in ``debug/delegation_log.jsonl``.
Thread-safe via a single lock.
"""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

__all__ = ["DelegationLog"]


def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat(timespec="seconds")


class DelegationLog:
    """Graph-wide append-only log at debug/delegation_log.jsonl.

    Every inter-node delegation — regardless of which node fired it — is
    recorded here. Stores FULL task text and FULL deliverable text (not truncated).
    Thread-safe via a single lock.
    """

    def __init__(self, log_path: Path) -> None:
        self._path = Path(log_path)
        self._lock = threading.Lock()
        # Monotonic sequence counter for globally-unique delegation IDs.
        # Shared across all orchestrating nodes that hold a reference to
        # this log — guarantees D### uniqueness even when datagenerator
        # and implementer both delegate to literature_reviewer.
        self._seq: int = 0
        # Ensure parent directory exists
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def next_id(self) -> str:
        """Return the next globally-unique delegation ID (e.g. ``"D001"``).

        Thread-safe: increments the shared monotonic counter under
        ``self._lock``.
        """
        with self._lock:
            self._seq += 1
            return f"D{self._seq:03d}"

    def record(
        self,
        *,
        id: str,
        from_node: str,
        to_node: str,
        task: str,
        deliverable: str,
        hypothesis_ids: list[str],
        started_at: str,
        completed_at: str,
        status: str,
        tokens_in: int = 0,
        tokens_out: int = 0,
        cost_usd: float | None = None,
        is_falsification_attempt: bool = False,
        evals: int = 0,
        phase: str | None = None,
    ) -> None:
        """Append one delegation record.

        task and deliverable are stored in full. ``phase`` is the optional
        f3dasm process phase this delegation belongs to (DoE / DataGeneration /
        ML / Optimization / …); additive, default None for old/untagged records.
        """
        record: dict[str, Any] = {
            "id": id,
            "from_node": from_node,
            "to_node": to_node,
            "task": task,
            "deliverable": deliverable,
            "hypothesis_ids": hypothesis_ids,
            "started_at": started_at,
            "completed_at": completed_at,
            "status": status,
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "cost_usd": cost_usd,
            "is_falsification_attempt": is_falsification_attempt,
            "evals": evals,
            "phase": phase,
        }
        with self._lock:
            with self._path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")

    def record_started(
        self,
        *,
        id: str,
        from_node: str,
        to_node: str,
        task: str,
        hypothesis_ids: list[str],
        started_at: str,
        is_falsification_attempt: bool = False,
        phase: str | None = None,
    ) -> None:
        """Append a RUNNING entry at DISPATCH, before the worker runs.

        A delegation flushes provenance-stamped rows into the canonical ledger
        DURING execution, but record() (the terminal DONE/FAILED entry) is only
        written at COMPLETION. A delegation cancelled or killed mid-flight (e.g.
        by the wall/eval budget) would otherwise leave ledgered rows with NO log
        entry — orphan evals that break the provenance audit trail and escape
        the eval-budget counter. This RUNNING entry guarantees every dispatched
        delegation is traceable; the terminal record (same id) supersedes it via
        the last-wins collapse in _load_all.
        """
        record: dict[str, Any] = {
            "id": id,
            "from_node": from_node,
            "to_node": to_node,
            "task": task,
            "deliverable": "",
            "hypothesis_ids": hypothesis_ids,
            "started_at": started_at,
            "completed_at": None,
            "status": "RUNNING",
            "tokens_in": 0,
            "tokens_out": 0,
            "cost_usd": None,
            "is_falsification_attempt": is_falsification_attempt,
            "evals": 0,
            "phase": phase,
        }
        with self._lock:
            with self._path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")

    def query_received(
        self, node_name: str, n: int | None = None
    ) -> list[dict]:
        """Return last n records where to_node == node_name, oldest-first.

        Returns all matching records if n is None.
        """
        with self._lock:
            records = self._load_all()

        matching = [r for r in records if r.get("to_node") == node_name]
        if n is not None:
            # MCP string-in tools may pass n as "6"; `matching[-n:]` would raise
            # "bad operand type for unary -: 'str'". Coerce here so every caller
            # (routing.py and worker.py RecallHistory) is covered at once.
            try:
                n = int(n)
            except (TypeError, ValueError):
                return matching
            # Return the last n, oldest-first
            matching = matching[-n:]
        return matching

    def last_completed_id(self, from_node: str) -> str | None:
        """Return the ID of the most recently completed (DONE) delegation
        sent BY from_node. Used for triggered_by injection in HypothesisUpdate."""
        with self._lock:
            records = self._load_all()

        # Filter for DONE delegations from this node, return last one's ID
        done = [
            r for r in records
            if r.get("from_node") == from_node and r.get("status") == "DONE"
        ]
        if not done:
            return None
        return done[-1]["id"]

    def mark_attempt(self, delegation_id: str, hypothesis_id: str) -> bool:
        """Retroactively flag an existing record as a falsification ATTEMPT of
        hypothesis_id (the read-time post-hoc link).

        Sets is_falsification_attempt=True, adds hypothesis_id to its
        hypothesis_ids, and stamps attempt_linked_post_hoc=True so the critic
        scrutinises adequacy harder than a flag declared up front at delegate
        time. Rewrites the (small) append-only log in place under the lock.
        Returns True iff a matching record was updated.
        """
        with self._lock:
            records = self._load_all()
            updated = False
            for r in records:
                if r.get("id") == delegation_id:
                    r["is_falsification_attempt"] = True
                    hids = r.get("hypothesis_ids") or []
                    if hypothesis_id not in hids:
                        hids = [*hids, hypothesis_id]
                    r["hypothesis_ids"] = hids
                    r["attempt_linked_post_hoc"] = True
                    updated = True
            if updated:
                with self._path.open("w", encoding="utf-8") as f:
                    for r in records:
                        f.write(json.dumps(r) + "\n")
            return updated

    def query_all(self) -> list[dict]:
        """Return every record, oldest-first."""
        with self._lock:
            return self._load_all()

    def _load_all(self) -> list[dict]:
        """Load all records, COLLAPSED to one per delegation id (last write wins).

        A delegation is logged with a RUNNING entry at dispatch and a terminal
        (DONE/FAILED) entry at completion. Collapsing last-wins gives every
        consumer one record per id — the terminal record if it exists, else the
        RUNNING entry that keeps a cancelled/killed delegation's ledger rows
        traceable. Ordered by the position of each id's latest write so
        completion order is preserved (last_completed_id stays correct).

        Must be called under lock or in a read-only context.
        """
        if not self._path.exists():
            return []
        lines = self._path.read_text(encoding="utf-8").strip().splitlines()
        latest: dict[str, dict] = {}
        order: dict[str, int] = {}
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            rid = r.get("id")
            if rid is None:
                continue
            latest[rid] = r
            order[rid] = i
        return [latest[k] for k in sorted(latest, key=lambda k: order[k])]
