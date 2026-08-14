"""Single source of truth for a run's resource-constraint state.

Every node-node interaction in this system is a delegation — human ->
strategizer (the initial problem-statement handoff), strategizer -> worker,
and strategizer -> critic (both GATE and FEEDBACK mode). Every one of them
should see the SAME live constraint facts (eval budget, wall-clock budget)
automatically, in-band, rather than something latent an agent has to go dig
for. This module computes that snapshot fresh at any delegation boundary;
both the text injected into an agent's context and the structured fields
persisted to DelegationLog come from calling this SAME function, so they can
never drift apart the way four independent, partial computations previously
did (a wall-clock-elapsed-only dispatch banner, a wall-clock-remaining-only
completion footer that only fired when eval rows were written, an eval-only
budget block hand-built inside the GATE critic call, and nothing at all in
FEEDBACK mode or the initial human handoff).
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ConstraintSnapshot:
    """A point-in-time read of the run's soft budgets. Always advisory —
    these numbers nudge; nothing here hard-stops a run."""

    eval_budget: int | None
    evals_used: int
    wall_budget_s: float | None
    wall_elapsed_s: float | None

    @property
    def evals_remaining(self) -> int | None:
        if self.eval_budget is None:
            return None
        return max(0, self.eval_budget - self.evals_used)

    @property
    def wall_remaining_s(self) -> float | None:
        if self.wall_budget_s is None or self.wall_elapsed_s is None:
            return None
        return max(0.0, self.wall_budget_s - self.wall_elapsed_s)

    @property
    def eval_exhausted(self) -> bool:
        return self.eval_budget is not None and self.evals_used >= self.eval_budget

    @property
    def wall_exhausted(self) -> bool:
        return (
            self.wall_budget_s is not None
            and self.wall_elapsed_s is not None
            and self.wall_elapsed_s >= self.wall_budget_s
        )

    @staticmethod
    def _dur(s: float) -> str:
        if s < 90:
            return f"{s:.0f}s"
        if s < 5400:
            return f"{s / 60:.1f}min"
        return f"{s / 3600:.2f}h"

    def as_text(self) -> str:
        """Compact block for injection into any agent-facing message —
        dispatch, completion report, or critic task message alike."""
        lines = ["<constraints>"]
        if self.eval_budget is not None:
            lines.append(
                f"Evaluation budget: {self.evals_used}/{self.eval_budget} "
                "evals used"
                + (" (EXHAUSTED)" if self.eval_exhausted else "") + "."
            )
        else:
            lines.append(
                f"Evaluation budget: unspecified "
                f"({self.evals_used} evals used so far)."
            )
        if self.wall_budget_s is not None and self.wall_elapsed_s is not None:
            _el = self._dur(self.wall_elapsed_s)
            _tot = self._dur(self.wall_budget_s)
            _pct = (self.wall_elapsed_s / self.wall_budget_s) * 100
            lines.append(
                f"Wall-clock budget: {_el}/{_tot} used ({_pct:.0f}%)"
                + (" (EXHAUSTED)" if self.wall_exhausted else "") + "."
            )
        else:
            lines.append("Wall-clock budget: unspecified.")
        lines.append("</constraints>")
        return "\n".join(lines)

    def as_dict(self) -> dict[str, Any]:
        """Structured fields for DelegationLog persistence — the same
        numbers as as_text(), never computed independently."""
        return {
            "eval_budget": self.eval_budget,
            "evals_used": self.evals_used,
            "evals_remaining": self.evals_remaining,
            "wall_budget_s": self.wall_budget_s,
            "wall_elapsed_s": self.wall_elapsed_s,
            "wall_remaining_s": self.wall_remaining_s,
        }


def _evals_used(experiment_data_dir: Path | None, delegation_log: Any) -> int:
    """Canonical evaluation count so far.

    Ledger rows (summed across every design namespace) are the single
    source of truth; falls back to summing the delegation log's
    self-reported counts only for lookup-direct studies that wrote no
    store — mirrors the rationale already established at the Done() gate
    (a delegation killed mid-flight flushes rows whose evals never reach
    the log, so the log-sum alone would under-count).
    """
    if experiment_data_dir is not None:
        try:
            from .instrumented import total_ledgered_evals
            n = int(total_ledgered_evals(experiment_data_dir))
            if n:
                return n
        except Exception:  # noqa: BLE001
            pass
    if delegation_log is not None:
        try:
            return sum(
                (r.get("evals") or 0) for r in delegation_log.query_all()
            )
        except Exception:  # noqa: BLE001
            pass
    return 0


def compute_constraint_snapshot(
    *,
    eval_budget: int | None,
    budget_seconds: float | None,
    run_start: float | None,
    experiment_data_dir: Path | None,
    delegation_log: Any = None,
) -> ConstraintSnapshot:
    """Compute a fresh constraint snapshot NOW.

    Call this at every delegation boundary (dispatch and completion) rather
    than caching a value from earlier — it is cheap (one ledger read plus
    arithmetic), and its entire purpose depends on being current, not on
    being fast to skip recomputing.
    """
    wall_elapsed = (
        time.time() - run_start if run_start is not None else None
    )
    evals_used = _evals_used(experiment_data_dir, delegation_log)
    return ConstraintSnapshot(
        eval_budget=eval_budget,
        evals_used=evals_used,
        wall_budget_s=budget_seconds,
        wall_elapsed_s=wall_elapsed,
    )


def snapshot_for_node(node: Any) -> ConstraintSnapshot:
    """Compute a fresh snapshot from a strategizer/worker node's own live
    attributes — the one place that knows how to derive
    experiment_data_dir (run_dir/experiment_data, via
    node._current_notes_dir.parent.parent, the same derivation already used
    at the Done() gate and by ScienceMonitor's store_dir wiring), so every
    call site shares this instead of re-deriving the path independently.
    """
    notes_dir = getattr(node, "_current_notes_dir", None)
    experiment_data_dir = (
        notes_dir.parent.parent / "experiment_data"
        if notes_dir is not None else None
    )
    return compute_constraint_snapshot(
        eval_budget=getattr(node, "_eval_budget", None),
        budget_seconds=getattr(node, "_budget_seconds", None),
        run_start=getattr(node, "_run_start", None),
        experiment_data_dir=experiment_data_dir,
        delegation_log=getattr(node, "_delegation_log", None),
    )
