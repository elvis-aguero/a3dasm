"""Run lifecycle: unrecoverable-condition detectors (USD budget, repeated errors,
time backstop) and the resumable checkpoint-halt. A mixin on the strategizer."""
from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ._constants import backstop_enabled, run_backstop_multiple

if TYPE_CHECKING:
    from langgraph.types import Command


class LifecycleMixin:
    def _halt_resumable(
        self,
        state: Any,
        *,
        reason: str,
        status: str = "halted",
        extra_update: dict | None = None,
    ):
        """Checkpoint-and-halt cleanly on an unrecoverable condition.

        Instead of crashing, write ``debug/run_status.json`` (so tooling and
        the human can see the run is resumable) and return a ``Command`` to
        ``END`` whose ``last_report`` is prefixed with a HALTED banner.  The
        durable SqliteSaver checkpoint + the persisted ``thread_id`` are what
        make the run resumable via ``AgenticRun(resume_from=...)`` — no new
        serialized state is introduced here.
        """
        import json as _json

        from langchain_core.messages import AIMessage
        from langgraph.graph import END
        from langgraph.types import Command

        run_dir = state.get("run_dir")
        thread_id = None
        if run_dir:
            debug_dir = Path(run_dir) / "debug"
            tid_path = debug_dir / "thread_id"
            try:
                if tid_path.exists():
                    thread_id = tid_path.read_text().strip()
            except OSError:
                thread_id = None
            try:
                debug_dir.mkdir(parents=True, exist_ok=True)
                (debug_dir / "run_status.json").write_text(
                    _json.dumps(
                        {
                            "status": status,
                            "reason": reason,
                            "resumable": True,
                            "thread_id": thread_id,
                        },
                        indent=2,
                    ),
                    encoding="utf-8",
                )
            except OSError:
                pass

        # Preserve the latest conclusion below the banner.
        prior_text = ""
        for _m in reversed(state["messages"]):
            if isinstance(_m, AIMessage):
                prior_text = str(_m.content)
                break

        banner = f"## ⚠ HALTED (resumable) — {reason}\n\n"
        update = {
            "messages": [],
            "done": True,
            "last_report": banner + (prior_text or "(no prior report)"),
            "token_totals": dict(self._token_totals),
            "error_counts": dict(self._error_counts),
        }
        if extra_update:
            update.update(extra_update)
        return Command(goto=END, update=update)

    def _check_unrecoverable(self, state: Any, budget: float | None, start: float | None) -> Command | None:
        """Return a halt Command if an unrecoverable condition is met, else None.
        Extracted verbatim from __call__ (USD ceiling → repeated errors → time
        backstop)."""

        def _halt_tallies() -> dict:
            with self._registry_lock:
                _total_new = len(self._registry)
                _evals_new = sum(
                    e["evals"] for e in self._registry.values()
                )
            return {
                "total_delegations": (
                    state["total_delegations"] + _total_new
                ),
                "evals_used": state.get("evals_used", 0) + _evals_new,
            }

        # (1) USD cost ceiling. Hard, resumable (raise budget_usd and resume).
        # Inactive under ollama (no cost data): warn once, never halt.
        _budget_usd = self._budget_usd
        if _budget_usd is not None and _budget_usd > 0:
            _spent = self._token_totals.get("total_cost_usd") or 0.0
            if not self._cost_observed:
                if (
                    getattr(self, "_turn_count", 0) >= 1
                    and not self._usd_inactive_warned
                ):
                    self._usd_inactive_warned = True
                    self._record_science_drift({
                        "error_type": "USD_BUDGET_INACTIVE",
                        "budget_usd": _budget_usd,
                        "note": "no per-call cost reported (e.g. ollama); "
                                "USD ceiling treated as inactive",
                    })
            elif _spent >= _budget_usd:
                self._record_science_drift({
                    "error_type": "USD_BACKSTOP",
                    "spent_usd": _spent,
                    "budget_usd": _budget_usd,
                })
                return self._halt_resumable(
                    state,
                    reason=(
                        f"USD budget exhausted "
                        f"(${_spent:.4f} / ${_budget_usd:.4f})"
                    ),
                    extra_update=_halt_tallies(),
                )

        # (2) Repeated errors: a target failing N times in a row (genuine
        # worker EXCEPTIONS, not REVISE loops or poor results — those reset the
        # streak on any success) is not going to self-heal by looping the
        # strategizer at it again. The default is deliberately CONSERVATIVE: a
        # legitimate run "stuck" in the scientific process loops on critic
        # verdicts and slow delegations, none of which count here — only hard
        # consecutive crashes do. Knob: max_consecutive_errors (config.yaml
        # runtime block; F3DASM_MAX_CONSECUTIVE_ERRORS overrides); 0 disables.
        from ..settings import get_int
        _max_err = get_int("max_consecutive_errors", 12)
        if _max_err > 0:
            with self._registry_lock:
                _stuck = [
                    (t, n) for t, n in self._consecutive_errors.items()
                    if n >= _max_err
                ]
            if _stuck:
                _t, _n = _stuck[0]
                self._record_science_drift({
                    "error_type": "REPEATED_ERRORS",
                    "target": _t,
                    "consecutive": _n,
                })
                return self._halt_resumable(
                    state,
                    reason=(
                        f"repeated errors: {_t} failed {_n}x consecutively"
                    ),
                    extra_update=_halt_tallies(),
                )

        # (3) Time backstop: past run_backstop_multiple x the (soft) time
        # budget, bound runaway cost. Now resumable (raise budget + resume).
        _backstop_mult = run_backstop_multiple()
        if backstop_enabled() and budget is not None and start is not None:
            _elapsed_now = time.time() - start
            if _elapsed_now > budget * _backstop_mult:
                with self._registry_lock:
                    _abandoned = [
                        d for d, e in self._registry.items()
                        if e["status"] in ("Working", "FollowUp")
                    ]
                self._record_science_drift({
                    "error_type": "RUN_BACKSTOP",
                    "elapsed": _elapsed_now,
                    "budget": budget,
                    "multiple": _backstop_mult,
                    "abandoned": _abandoned,
                })
                return self._halt_resumable(
                    state,
                    reason=(
                        f"time backstop: {int(_backstop_mult)}x budget "
                        f"exceeded ({_elapsed_now:.0f}s / {budget:.0f}s)"
                    ),
                    extra_update=_halt_tallies(),
                )

        return None
