"""Science monitor: only the UNLEDGERED_EVALS runtime rule remains.

Hard invariants (EVIDENCE_DELEGATION_EXISTS, SUPPORTED_WITHOUT_ATTACK)
have been moved to the data boundary — HypothesisUpdate tool in
strategizer.py returns an error synchronously when violated.

Soft guidance rules (STALE_OPEN, UNANCHORED_DELEGATION, POSTERIOR_INERTIA)
have been removed from the runtime. They are covered by:
  - delegation-finish chores prompt (routing.py)
  - prompt guidance in the strategizer system prompt

EVIDENCE_NUMBERS_MATCH removed: too brittle — derived/reformatted values
legitimately appear in evidence without matching the report verbatim.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

__all__ = ["ScienceMonitor", "Violation"]

MAX_INJECT_PER_TURN = 2
ESCALATION_CAP = 2
ESCALATE_AFTER_VIOLATIONS = 3
ESCALATE_AFTER_ERROR_REPEATS = 2

# DUPLICATE_EVALUATION — new duplicate rows since the last nudge that trip
# the counter, the per-delegation nudge cap, and the minimum gap between
# nudges. See backlog #24 / specs/07-duplicate-evaluation-detection.md.
DUP_NUDGE_THRESHOLD = 3
DUP_MAX_NUDGES = 2
DUP_COOLDOWN_S = 60.0


@dataclass(frozen=True)
class Violation:
    rule: str
    severity: str          # "error" | "warn"
    h_id: str | None
    message: str

    @property
    def key(self) -> tuple:
        return (self.rule, self.h_id)


class ScienceMonitor:
    """Evaluates UNLEDGERED_EVALS; formats bounded corrective injections."""

    def __init__(
        self,
        ledger,
        delegation_log,
        diagnostics_writer: Callable[[dict], None] | None = None,
        stale_k: int = 3,       # kept for API compat, unused
        max_inject: int = MAX_INJECT_PER_TURN,
        escalation_cap: int = ESCALATION_CAP,
        store_dir: str | None = None,
        role_of: Callable[[str], str] | None = None,
    ) -> None:
        self._ledger = ledger
        self._dlog = delegation_log
        self._diag = diagnostics_writer
        # Maps a node name -> its role, so UNLEDGERED_EVALS can exempt roles
        # whose contract forbids get_evaluator() (the datagenerator).
        self._role_of = role_of
        self._max_inject = max_inject
        self._escalation_cap = escalation_cap
        self._lock = threading.Lock()
        self._logged_keys: set[tuple] = set()
        self._h_rule_seen: dict[str, set[str]] = {}
        self._error_streak: dict[tuple, int] = {}
        self._escalations = 0
        self._pending_escalation: list[str] | None = None
        self.store_dir: str | None = store_dir
        # DUPLICATE_EVALUATION bookkeeping, per delegation id.
        self._now: Callable[[], float] = time.time
        self._dup_baseline: dict[str, int] = {}
        self._dup_counter: dict[str, int] = {}
        self._dup_nudge_count: dict[str, int] = {}
        self._dup_last_nudge_ts: dict[str, float] = {}

    def evaluate(self) -> list[Violation]:
        """Run UNLEDGERED_EVALS + UNSTAMPED_ROWS + DUPLICATE_EVALUATION
        against current state."""
        records = self._dlog.query_all()
        return (
            self._check_unledgered(records)
            + self._check_unstamped_rows()
            + self._check_duplicate_evaluations()
        )

    def _check_unstamped_rows(self) -> list[Violation]:
        """UNSTAMPED_ROWS: the experiment stores gained rows that carry no
        provenance stamp — appended outside get_evaluator() (the public
        ExperimentData.store() write-door). Principle: a counted eval must be
        attributable to a source; a stamp-less append is neither counted nor
        reproducible, so surface it instead of letting it pass silently. This is
        the reverse direction of UNLEDGERED_EVALS (which asks 'did a delegation's
        evals reach the store?'); this asks 'do all the store's rows have an
        owner?'. Warn-only — it never blocks a run. No-ops when store_dir is None.
        """
        if self.store_dir is None:
            return []
        try:
            from .instrumented import unstamped_row_count as _unstamped
            n = _unstamped(self.store_dir)
        except Exception:  # noqa: BLE001
            return []
        if n <= 0:
            return []
        return [Violation(
            "UNSTAMPED_ROWS", "warn", None,
            f"{n} row(s) in the experiment store carry no provenance stamp — "
            "they were written outside get_evaluator() (e.g. a direct "
            "ExperimentData.store() append). Such rows are NOT counted as "
            "evaluations and CANNOT be attributed to a delegation or "
            "reproduced. Any number backing the headline must come from rows "
            "written through get_evaluator(). Re-run those evaluations via "
            "get_evaluator() so they are ledgered and attributable.",
        )]

    def _check_duplicate_evaluations(self) -> list[Violation]:
        """DUPLICATE_EVALUATION: a delegation re-evaluated a design point
        already FINISHED, unchanged, in the ledger — wasted budget, no new
        evidence. Real incident (run example_study/20260713T221841): three
        separate scripts under one delegation independently re-sampled an
        identical seed=42 LHS design, so 122 of its 160 rows (76%) were exact
        repeats of 38 unique points (backlog #24).

        Counter-based, not level-triggered like UNLEDGERED_EVALS/
        UNSTAMPED_ROWS: fires once DUP_NUDGE_THRESHOLD NEW duplicate rows have
        landed since the last check for that delegation, then resets to 0 —
        so it fires again on the NEXT batch of new duplicates, not
        continuously while the condition persists. Capped at DUP_MAX_NUDGES
        per delegation and rate-limited to one per DUP_COOLDOWN_S, so a burst
        of duplicate rows can't spam the strategizer. No-ops when store_dir is
        None.
        """
        if self.store_dir is None:
            return []
        try:
            from .instrumented import duplicate_eval_stats as _dupstats
            stats = _dupstats(self.store_dir)
        except Exception:  # noqa: BLE001
            return []
        out: list[Violation] = []
        with self._lock:
            for d_id, s in stats.items():
                total_dup = s["duplicate_rows"]
                baseline = self._dup_baseline.get(d_id, 0)
                new_dups = total_dup - baseline
                if new_dups <= 0:
                    continue
                self._dup_baseline[d_id] = total_dup
                if self._dup_nudge_count.get(d_id, 0) >= DUP_MAX_NUDGES:
                    continue
                self._dup_counter[d_id] = (
                    self._dup_counter.get(d_id, 0) + new_dups)
                if self._dup_counter[d_id] < DUP_NUDGE_THRESHOLD:
                    continue
                now = self._now()
                last_ts = self._dup_last_nudge_ts.get(d_id)
                if last_ts is not None and (now - last_ts) < DUP_COOLDOWN_S:
                    continue
                self._dup_counter[d_id] = 0
                self._dup_nudge_count[d_id] = (
                    self._dup_nudge_count.get(d_id, 0) + 1)
                self._dup_last_nudge_ts[d_id] = now
                worst = s["worst"]
                worst_txt = (
                    f" e.g. {worst[0]} was evaluated {worst[1]}x."
                    if worst else ""
                )
                out.append(Violation(
                    "DUPLICATE_EVALUATION", "warn", d_id,
                    f"Delegation {d_id} has re-evaluated design points "
                    f"already in the ledger — {total_dup} duplicate row(s) "
                    f"so far.{worst_txt} Check QueryStore for existing rows "
                    "at a candidate before re-evaluating it; re-sampling the "
                    "same points wastes eval budget without new evidence.",
                ))
        return out

    def _check_unledgered(self, all_records: list[dict]) -> list[Violation]:
        """UNLEDGERED_EVALS: DONE delegations that reported evals but wrote no
        rows to ANY experiment store. No-ops when store_dir is None.

        Counts a delegation's stamped rows by provenance across every experiment
        store (not just the default one) — so a worker that reached the oracle
        via get_evaluator(namespace=...) at the call site is NOT false-flagged as
        having bypassed it (run 20260627T203327 D003 wrote to the 'polar' store
        and was wrongly warned by the old canonical-only check).
        """
        if self.store_dir is None:
            return []
        from pathlib import Path as _Path
        sd = _Path(self.store_dir)
        try:
            from .instrumented import delegation_evals as _devals
        except Exception:  # noqa: BLE001
            return []

        out: list[Violation] = []
        for r in all_records:
            if r.get("status") != "DONE":
                continue
            evals = r.get("evals", 0) or 0
            if evals <= 0:
                continue
            # The datagenerator validates ONE sample by calling its generator
            # directly (gen.call), NOT get_evaluator() — its source is not yet
            # registered at validation time, so by spec it CANNOT ledger that
            # eval. Flagging it as unledgered drift is a guaranteed false
            # positive against the role contract, so exempt the role.
            if (self._role_of is not None
                    and self._role_of(r.get("to_node", "")) == "datagenerator"):
                continue
            d_id = r.get("id", "")
            if _devals(sd, d_id) > 0:
                continue
            # Key on the delegation id (not None) so multiple
            # simultaneously-unledgered delegations each get their own
            # message + dedupe bookkeeping instead of collapsing to one.
            out.append(Violation(
                "UNLEDGERED_EVALS", "warn", d_id,
                f"Delegation {d_id} reported {evals} evaluation(s) but "
                "wrote none to the canonical ledger — it bypassed "
                "get_evaluator(). Fine for throwaway exploration, but any "
                "number that feeds the HEADLINE must come from ledgered "
                "rows or pipeline.ipynb cannot reproduce it. If this "
                "delegation's results back a conclusion, re-run them via "
                "get_evaluator().",
            ))
        return out

    def on_delegation_complete(self, delegation_id: str) -> None:
        """Hook after a delegation finishes; updates bookkeeping."""
        self._bookkeep(self.evaluate(), count_streaks=False)

    def drain(self) -> str:
        """Re-validate, dedupe, cap, digest. Returns injection text
        (possibly empty). Call from the node's _drain_notifications."""
        live = self.evaluate()
        self._bookkeep(live, count_streaks=True)
        if not live:
            return ""
        seen: dict[tuple, Violation] = {}
        for v in sorted(live, key=lambda v: v.severity != "error"):
            seen.setdefault(v.key, v)
        ordered = list(seen.values())
        head = ordered[: self._max_inject]
        rest = ordered[self._max_inject:]
        lines = [
            f"[SCIENCE MONITOR — {v.rule}] {v.message}" for v in head
        ]
        if rest:
            digest = ", ".join(
                f"{v.rule}({v.h_id})" for v in rest)
            lines.append(
                f"[SCIENCE MONITOR] +{len(rest)} more: {digest}")
        return "\n".join(lines) + "\n"

    def _bookkeep(
        self,
        live: list[Violation],
        count_streaks: bool = False,
    ) -> None:
        """Update diagnostics log, h_rule_seen, and (optionally) streaks.

        Streak counting is intentionally restricted to drain() calls so
        that a persistent error is not double-counted within a single
        agent turn (on_hypothesis_update + drain would both increment).
        Semantics: "persisting across N drain calls".
        """
        with self._lock:
            live_keys = {v.key for v in live}
            for v in live:
                if v.key not in self._logged_keys:
                    self._logged_keys.add(v.key)
                    if self._diag is not None:
                        self._diag({
                            "rule": v.rule, "severity": v.severity,
                            "hypothesis_id": v.h_id,
                            "message": v.message,
                        })
                if v.h_id is not None:
                    self._h_rule_seen.setdefault(
                        v.h_id, set()).add(v.rule)
                if count_streaks and v.severity == "error":
                    self._error_streak[v.key] = \
                        self._error_streak.get(v.key, 0) + 1
            # resolved keys may re-fire later → re-log + reset streaks
            for key in list(self._logged_keys):
                if key not in live_keys:
                    self._logged_keys.discard(key)
                    self._error_streak.pop(key, None)
            if self._escalations >= self._escalation_cap:
                self._pending_escalation = None
                return
            offenders = sorted(
                h for h, rules in self._h_rule_seen.items()
                if len(rules) >= ESCALATE_AFTER_VIOLATIONS)
            offenders += sorted(
                {key[1] for key, n in self._error_streak.items()
                 if n >= ESCALATE_AFTER_ERROR_REPEATS
                 and key[1] is not None
                 and key[1] not in offenders})
            self._pending_escalation = offenders or None

    def escalation_due(self) -> list[str] | None:
        with self._lock:
            return self._pending_escalation

    def note_escalated(self) -> None:
        with self._lock:
            self._escalations += 1
            self._pending_escalation = None
            self._h_rule_seen.clear()
            self._error_streak.clear()
