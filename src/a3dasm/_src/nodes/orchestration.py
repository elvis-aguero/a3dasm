"""The behaviour of a node that HAS outgoing edges: it delegates.

One of the two behaviours :class:`~.node.Node` dispatches between, and the
only one that runs a multi-turn loop: it reads Reports, decides the next
delegation, and owns the run's gates (budget, critic, reproduction). A node
has this behaviour because of its TOPOLOGY, not its role — any node with an
outgoing edge orchestrates, whatever it is called.

``inspect.getsource(Node._orchestrate)`` reads the full routing logic.
"""

from __future__ import annotations

import threading
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..runtime.graph_state import AgenticState

from ..epistemics.hypothesis_ledger import HypothesisLedger
from ..epistemics.science_monitor import ScienceMonitor
from ..infra.delegation_log import DelegationLog
from ._constants import run_backstop_multiple
from .notices import wrap_notice
from .parsing import _to_adapter_messages


class OrchestrationMixin:
    """Delegating behaviour, engaged when a node has outgoing edges."""

    def _init_orchestration(
        self,
        *,
        name: str,
        outgoing: list[str],
        spec: Any,
        study_dir: Any = None,
        interactive: bool = False,
        max_ask: int = 1,
        worker_adapters: dict | None = None,
        notes_dir: Any = None,
        workspace_dir: Any = None,
        delegation_log: DelegationLog | None = None,
    ) -> None:
        """Set up the delegation registry, the ledgers and the routing tools."""
        self._route: dict = {}
        self._study_dir = study_dir
        self._workspace_dir = Path(workspace_dir) if workspace_dir is not None else None
        self._interactive = interactive
        self._max_ask = max_ask
        self._ask_count = 0
        self._current_notes_dir: Path | None = None
        # Parallel delegation registry: id → {"status", "result", "evals", "hypothesis_ids", "started_at"}
        self._worker_adapters: dict[str, Any] = worker_adapters or {}
        self._registry: dict[str, dict] = {}
        self._registry_lock = threading.Lock()
        self._threads: dict[str, threading.Thread] = {}
        # Push notifications: background threads append here; tool calls drain it.
        self._notifications: list[str] = []
        self._notifications_lock = threading.Lock()
        # Budget state — set at the start of each __call__ from AgenticState
        self._budget_seconds: float | None = None
        self._run_start: float | None = None
        # Hard USD cost ceiling (None = inactive). Set each __call__ from state.
        self._budget_usd: float | None = None
        # True once any LLM call reports a real cost (claude). Stays False under
        # ollama (cost is None) → the USD ceiling is treated as inactive.
        self._cost_observed: bool = False
        self._usd_inactive_warned: bool = False
        # Consecutive Errored delegations per target (reset on that target's
        # next success). Drives the repeated-errors resumable halt.
        self._consecutive_errors: dict[str, int] = {}
        # Per-delegation pending messages (budget warnings) to prepend to
        # worker tool results.  Keyed by delegation_id; drained on next call.
        self._pending_worker_msgs: dict[str, list[str]] = {}
        self._pending_worker_msgs_lock = threading.Lock()
        # Tracks which budget % thresholds (80, 90, 100, 110 …) have already
        # been broadcast to workers so each is sent exactly once.
        self._budget_notified_pcts: set[int] = set()
        # Confer messaging: async inter-node messages keyed by TARGET node name.
        # A node's messages are delivered when it next drains (orchestrator: each
        # turn via _drain_notifications; worker: collect-on-send when it next
        # calls Confer). Faithful port of the stashed Confer design (audit).
        self._confer_seq: int = 0
        self._confer_inbox: dict[str, list[str]] = {}
        self._confer_inbox_lock = threading.Lock()
        # Hypothesis ledger — persists hypotheses.json
        self._ledger: HypothesisLedger | None = (
            HypothesisLedger(Path(notes_dir)) if notes_dir is not None else None
        )
        # Milestone ledger (process policy) — persists milestones.json. Seeded
        # with the config default gates unless disabled. DISTINCT from the
        # hypothesis ledger (epistemics): process vs what's-true.
        from ..epistemics.milestones import MilestoneLedger
        from ..runtime.settings import get_bool
        self._milestones: MilestoneLedger | None = None
        if notes_dir is not None and get_bool("milestones_enabled", True):
            self._milestones = MilestoneLedger(Path(notes_dir))
            # C3 switchable: the draft-pipeline gate seeds only when the
            # pipeline-deliverable knob is on (off = byte-identical to today).
            self._milestones.seed_defaults(
                include_pipeline=get_bool("pipeline_deliverable", True))
        # Graph-wide delegation log (demand-driven episodic memory)
        self._delegation_log: DelegationLog | None = delegation_log
        # Science drift monitor — active when both ledger and log present
        self._science_monitor: ScienceMonitor | None = None
        if self._ledger is not None and delegation_log is not None:
            self._science_monitor = ScienceMonitor(
                self._ledger,
                delegation_log,
                diagnostics_writer=self._record_science_drift,
                role_of=self._role_of,
            )
        # Running total of delegations at the START of the current __call__
        # Used as a seed for the delegation sequence counter.
        self._state_total_delegations: int = 0
        # Snapshot of _delegation_seq at the start of the current turn, so
        # the terminal Command counts only THIS turn's new delegations. Set
        # again by _absorb_state every turn; initialised here so the node's
        # attribute set is complete before any turn has run.
        self._seq_at_turn_start: int = 0
        # Monotonic per-node delegation counter — never reset within a
        # run.  Seeded from _state_total_delegations on first __call__
        # so checkpoint-resumed runs continue from the correct offset.
        # Because it never resets, it avoids the ID collision that
        # occurs when completed delegations are pruned from the registry
        # but _state_total_delegations has not yet accumulated them.
        self._delegation_seq: int = 0
        # Two-shot Done() gate: first call warns, second call closes.
        # Resets to False whenever a new Delegate() fires.
        self._done_warned: bool = False
        # Science monitor fires once per turn; reset at __call__ start.
        self._science_injected_this_turn: bool = False
        # Post-Done exit interview: set after the critic accepts; the next
        # Done() carries only the retrospective. _final_summary holds the real
        # conclusion so the recorded summary is the science, not the interview.
        self._awaiting_retro: bool = False
        self._final_summary: str | None = None
        # Consecutive non-PASS critic verdicts; after 3, the gate closes
        # gracefully UNGATED (bounded escape) instead of looping forever.
        self._revise_count: int = 0
        # Eval budget for this run (stashed each turn from state).
        self._eval_budget: int | None = None
        # Cumulative cap on the "no canonical source registered" nudge (soft).
        self._no_source_nudges: int = 0
        # Bounded re-prompt counter: incremented each time the node loops back
        # due to an unaccepted termination (no Done or refused Done).  NOT reset
        # in the A1/A2 per-turn block — it persists across loopbacks within one
        # run.  After 3 loopbacks the run terminates UNGATED.
        self._finish_attempts: int = 0
        # Accumulated token usage across strategizer + all workers this run.
        self._token_totals: dict = {
            "input_tokens": 0,
            "output_tokens": 0,
            "cache_read_input_tokens": 0,
            "cache_creation_input_tokens": 0,
            "total_cost_usd": 0.0,
        }
        # Per-node raw tool-call error count: any ERROR: return or raised
        # exception from any injected closure counts as one error for that node.
        self._error_counts: dict[str, int] = {}
        # Separable per-call telemetry — additive, off the decision path. Lives
        # under debug/telemetry/ (notes_dir is debug/strategizer_notes).
        from ..infra.telemetry import Telemetry
        self._telemetry: Telemetry | None = (
            Telemetry(Path(notes_dir).parent) if notes_dir is not None else None
        )
        self.adapter.closure_tools.update(self._build_routing_closures())
        self.adapter.route_watcher = lambda: self._route.get("kind") == "done"

    # ── Authoritative delegation status (audit BF-0) ─────────────────────────
    # The persistent delegation_log owns existence + terminal status: it
    # survives node reconstruction and background threads write their terminal
    # DONE/FAILED record to it. The in-memory _registry is ONLY a cache of live
    # execution state (threads, streamed results) and can lag or be rebuilt
    # empty — so any "does D exist / is it terminal" question reads the log.
    def _log_status(self, delegation_id: str) -> tuple[str | None, str]:
        """Return (status, deliverable) for *delegation_id* from the persistent
        log, or (None, "") if the log has no such delegation."""
        if self._delegation_log is None:
            return None, ""
        for r in self._delegation_log.query_all():
            if r.get("id") == delegation_id:
                return r.get("status"), (r.get("deliverable") or "")
        return None, ""

    def _pending_delegations(self) -> list[str]:
        """In-flight delegations, reconciled against the authoritative log.

        A delegation the log shows terminal (DONE/FAILED) is never reported
        pending, even if the in-memory cache still says "Working". That stale
        state is what made Done()'s liveness gate refuse forever and kill run4
        by watchdog after it had already found the optimum (audit BF-0/BF-2).
        """
        with self._registry_lock:
            pending = [
                d for d, e in self._registry.items()
                if e.get("status") == "Working"
            ]
        if self._delegation_log is not None:
            terminal = {
                r["id"] for r in self._delegation_log.query_all()
                if r.get("status") in ("DONE", "FAILED")
            }
            pending = [d for d in pending if d not in terminal]
        return pending

    def _find_datagenerator_name(self) -> str | None:
        """Name of the first connected datagenerator worker, or None.

        Its presence means a canonical ground-truth source CAN be authored
        and registered for this study.
        """
        spec = self._spec
        if spec is None or not hasattr(spec, "nodes"):
            return None
        for target in self._outgoing:
            agent = spec.nodes.get(target)
            if (
                agent is not None
                and getattr(agent, "role", None) == "datagenerator"
                and target in self._worker_adapters
            ):
                return target
        return None

    def _canonical_source_registered(self) -> bool:
        """True if a canonical ground-truth source is resolvable.

        Reads run_config.json: an evaluator entrypoint OR a lookup pool counts
        as a registered source. Best-effort — on any read failure, assume not
        registered (the nudge is soft, so a false 'no' just costs one notice).
        """
        notes = self._current_notes_dir
        if notes is None:
            return False
        try:
            import json as _json
            cfg = _json.loads(
                (Path(notes).parent / "run_config.json").read_text())
            return bool(
                cfg.get("evaluator_entrypoint")
                or cfg.get("evaluator_lookup")
            )
        except Exception:  # noqa: BLE001
            return False

    @property
    def _current_run_dir(self):
        """This run's root, derived from the notes dir the graph state sets.

        ``run_dir`` arrives on the state, not on the node, and only
        ``_current_notes_dir`` (``<run>/debug/strategizer_notes``) is kept
        from it — so the run root is that path's grandparent. ``None``
        before the first invoke, which callers must tolerate.
        """
        notes = self._current_notes_dir
        return None if notes is None else notes.parent.parent

    def _drain_notifications(self) -> str:
        """Return and clear any pending push notifications, or empty
        string."""
        with self._notifications_lock:
            if not self._notifications:
                text = ""
            else:
                msgs = list(self._notifications)
                self._notifications.clear()
                text = "\n".join(msgs) + "\n\n"
        # Confer inbox: messages other nodes addressed to THIS node (async
        # mailbox). Drained here so the orchestrator receives them on its next
        # turn / next tool call, prepended to any push notifications.
        with self._confer_inbox_lock:
            _confer = self._confer_inbox.pop(self._name, [])
        if _confer:
            text = "\n\n".join(_confer) + "\n\n" + text
        # Operator notes: messages a human queued in the viewer while the run
        # was working. Delivered here, on the same path as Confer, so a note
        # reaches the agent at its next tool call rather than interrupting a
        # turn in progress. Marked as coming from the operator because the
        # agent should weigh it differently from another agent's message —
        # it is the one voice in the run that is not itself an agent.
        run_dir = self._current_run_dir
        if run_dir is not None:
            from ..infra.operator_channel import drain_note_rows
            rows = drain_note_rows(run_dir)
            mine: list[str] = []
            for _row in rows:
                _to = _row.get("to_node") or ""
                _note = (
                    "[OPERATOR NOTE — from the human running this study. "
                    "Weigh it as a briefing correction, not as another "
                    f"agent's opinion: {_row['text']}]"
                )
                # A note addressed to a RUNNING delegation goes to that
                # worker, on the same per-delegation queue Confer and the
                # budget warnings use — so a human can correct work already
                # in flight instead of waiting for a wrong result. The queue
                # is claimed destructively, so this is the only place that
                # may drain it: routing here is what keeps an addressed note
                # from being swallowed by the orchestrator's own delivery.
                if _to:
                    with self._registry_lock:
                        _entry = self._registry.get(_to)
                        _live = bool(_entry) and _entry.get("status") in (
                            "Working", "FollowUp")
                    if _live:
                        with self._pending_worker_msgs_lock:
                            self._pending_worker_msgs.setdefault(
                                _to, []).append(_note)
                        continue
                    # Addressed to something not running: the human still
                    # said it, so it must not vanish — hand it to the
                    # orchestrator with the intended recipient named.
                    _note = (
                        f"[OPERATOR NOTE addressed to {_to}, which is not "
                        f"running — delivered to you instead: "
                        f"{_row['text']}]"
                    )
                mine.append(_note)
            if mine:
                text = "\n\n".join(mine) + "\n\n" + text
        if self._science_monitor is not None:
            offenders = self._science_monitor.escalation_due()
            _critic_name = self._find_critic_name()
            if offenders and _critic_name is not None:
                # Escalation fires: perform bookkeeping-only drain (discard
                # text) so the critic findings are the sole corrective
                # payload — regular drift messages would pollute context.
                self._science_monitor.drain()
                task_msg = self._build_feedback_task_msg(offenders)
                findings = self._invoke_critic(task_msg)
                self._science_monitor.note_escalated()
                if self._delegation_log is not None:
                    _fb_id = (
                        "FB"
                        + datetime.now(
                            tz=timezone.utc
                        ).strftime("%H%M%S")
                    )
                    self._delegation_log.record(
                        id=_fb_id,
                        from_node=self._name,
                        to_node=_critic_name,
                        task="ScienceMonitor escalation audit",
                        deliverable=findings,
                        hypothesis_ids=offenders,
                        workspace_sha=self._commit_workspace(
                            f"{_fb_id} {self._name} -> {_critic_name} "
                            "[ESCALATION]"),
                        started_at=datetime.now(
                            tz=timezone.utc
                        ).isoformat(timespec="seconds"),
                        completed_at=datetime.now(
                            tz=timezone.utc
                        ).isoformat(timespec="seconds"),
                        status="FEEDBACK",
                        tokens_in=0,
                        tokens_out=0,
                        cost_usd=None,
                    )
                text += (
                    "[SCIENCE MONITOR — ESCALATION] Repeated drift "
                    f"on {', '.join(offenders)}. Critic audit "
                    f"findings:\n{findings}\n"
                )
            else:
                # No escalation: inject at most once per strategizer turn
                # to avoid the same warning appearing on every tool call.
                if not self._science_injected_this_turn:
                    drift = self._science_monitor.drain()
                    if drift:
                        text += drift
                        self._science_injected_this_turn = True
        # Everything accumulated above is a3dasm speaking to the agent, not
        # a tool's output — mark it so both the agent and the viewer can
        # tell the difference (see nodes/notices.py).
        return wrap_notice(text)

    def _next_confer_seq(self) -> int:
        """Monotonic per-run Confer message sequence number."""
        with self._confer_inbox_lock:
            self._confer_seq += 1
            return self._confer_seq

    def _build_routing_closures(self) -> dict:
        from .tools.routing import build_routing_tools
        return build_routing_tools(self)

    def _wrap_closure(self, fn: Any, node_name: str) -> Any:
        """Return a version of *fn* that records ERROR returns and exceptions.

        Uses functools.wraps so inspect.signature() follows __wrapped__ to the
        original function — _infer_schema_from_callable must see the real
        parameter names, not (*args, **kwargs).

        Also coerces string-typed arguments to int/float/bool when the
        function annotation requests it (handles Ollama passing "5" for
        an int parameter).
        """
        import functools as _functools
        import inspect as _inspect
        import typing as _typing

        node = self
        tool_name = getattr(fn, "__name__", repr(fn))

        # Resolve type hints once; fall back to {} if any forward ref
        # cannot be resolved (e.g. "DelegationLog | None").
        try:
            _hints = _typing.get_type_hints(fn)
        except Exception:  # noqa: BLE001
            _hints = {}
        _COERCIBLE = {int, float, bool}

        def _coerce(name: str, value: Any) -> Any:
            target = _hints.get(name)
            if target not in _COERCIBLE or not isinstance(value, str):
                return value
            if target is bool:
                low = value.strip().lower()
                if low in ("true", "1", "yes"):
                    return True
                if low in ("false", "0", "no"):
                    return False
                return value
            try:
                return target(value)
            except ValueError:
                return value

        @_functools.wraps(fn)
        def _wrapped(*args, **kwargs):
            # Coerce string args before calling the real function.
            try:
                bound = _inspect.signature(fn).bind_partial(
                    *args, **kwargs
                )
                for pname in list(bound.arguments):
                    bound.arguments[pname] = _coerce(
                        pname, bound.arguments[pname]
                    )
                args, kwargs = bound.args, bound.kwargs
            except TypeError:
                pass  # signature mismatch: let fn raise its own error

            try:
                result = fn(*args, **kwargs)
                if (
                    isinstance(result, str)
                    and result.lstrip().startswith("ERROR:")
                ):
                    node._record_tool_error(
                        node_name,
                        tool_name,
                        "ERROR_RETURN",
                        result[:300],
                    )
                return result
            except Exception as exc:
                node._record_tool_error(
                    node_name,
                    tool_name,
                    type(exc).__name__,
                    str(exc)[:300],
                    tb=traceback.format_exc(),
                )
                raise

        return _wrapped

    def _orchestrate(self, state: AgenticState) -> Any:
        """One orchestration turn, in the order it happens.

        Absorb the run state onto the node, work out what the agent must be
        TOLD this turn, invoke it once, then route on what it did. Every step
        is a method below, named for its step; a turn ends in exactly one of
        three ways, which :meth:`_route_turn` states.
        """
        self._absorb_state(state)
        budget_warnings = self._budget_warnings(state)
        halt = self._check_unrecoverable(
            state, self._budget_seconds, self._run_start)
        if halt is not None:
            return halt
        pending_notifs = self._reset_for_turn()
        messages = self._compose_messages(state, budget_warnings, pending_notifs)
        ai_msg = self._invoke_turn(messages)
        return self._route_turn(state, ai_msg)

    # ── Before the turn ──────────────────────────────────────────────────────

    def _absorb_state(self, state: AgenticState) -> None:
        """Copy the run state the node's TOOLS read onto the node itself.

        The closures reach their context through ``node.…``, not through
        AgenticState, so anything a tool needs has to land here first.
        """
        # Update notes_dir from current state run_dir
        run_dir = state.get("run_dir")
        if run_dir:
            self._current_notes_dir = (
                Path(run_dir) / "debug" / "strategizer_notes"
            )
            if self._ledger is None:
                self._ledger = HypothesisLedger(self._current_notes_dir)
            # Wire canonical store dir into ScienceMonitor lazily.
            # store_dir is the ExperimentData *project_dir* (run_dir/
            # experiment_data), NOT the folder holding the CSVs. ExperimentData
            # appends its own EXPERIMENTDATA_SUBFOLDER ("experiment_data"), so
            # the rows live one level deeper at
            # run_dir/experiment_data/experiment_data/output.csv — hence the
            # apparent double directory is correct, not a typo.
            if self._science_monitor is not None:
                self._science_monitor.store_dir = (
                    self._current_notes_dir.parent.parent
                    / "experiment_data"
                )

        # Eval budget → available to the Done() critic gate for budget-aware
        # framing (judge the best honest conclusion within evals spent).
        self._eval_budget = state.get("eval_budget")

        # Required aux deliverables (config.yaml) → on the node so WriteDeliverable
        # may write them: the gate REQUIRES them, so the writing tool must accept
        # them (else gate-vs-tool deadlock — audit run 20260624T021359).
        self._required_deliverables = state.get("required_deliverables") or []

        # Store on node so GetStatus() can compute delegation timeout
        self._budget_seconds = state.get("budget_seconds")
        self._run_start = state.get("start_time")
        self._budget_usd = state.get("budget_usd")

        # Capture total_delegations so Delegate() can seed the counter.
        self._state_total_delegations = state.get("total_delegations", 0)
        # Seed the monotonic counter from state on first turn (or after a
        # checkpoint rebuild).  Never decremented — ensures IDs are unique
        # even when the registry is pruned between turns.
        if self._delegation_seq < self._state_total_delegations:
            self._delegation_seq = self._state_total_delegations
        # Snapshot seq at turn start so total_new counts only THIS turn.
        self._seq_at_turn_start: int = self._delegation_seq

    def _ledgered_eval_total(self, floor: int) -> int:
        """The run's eval count, preferring the ledger over an accumulator.

        The canonical ledger is the source of truth: a killed/cancelled
        delegation flushes rows the state accumulator never sees, so the
        accumulator undercounts. Summed across the canonical store AND every
        design namespace — namespace evals were invisible to the run total and
        to the soft budget. ``floor`` keeps the accumulator for lookup-direct
        studies with no instrumented store.
        """
        try:
            from ..evaluation.ledger_summary import total_ledgered_evals
            _nd = getattr(self, "_current_notes_dir", None)
            if _nd is not None:
                return max(
                    floor,
                    int(total_ledgered_evals(
                        _nd.parent.parent / "experiment_data")),
                )
        except Exception:  # noqa: BLE001
            pass
        return floor

    def _budget_warnings(self, state: AgenticState) -> list[dict]:
        """Advisory budget messages for this turn.

        The time budget is a SOFT constraint — warnings only; the run is never
        force-terminated for exceeding it. A separate run-level backstop
        (RUN_BACKSTOP_MULTIPLE x budget) bounds runaway cost.
        """
        import time
        warnings: list[dict] = []
        budget, start = self._budget_seconds, self._run_start
        if budget is not None and start is not None:
            elapsed = time.time() - start
            pct = elapsed / budget
            if pct >= 1.0:
                warnings.append({
                    "role": "user",
                    "content": (
                        f"Time budget fully consumed "
                        f"({elapsed:.0f}s / {budget:.0f}s). This is an "
                        "advisory soft limit — the run is NOT terminated. "
                        "Wind down: finish the experiment in flight, then "
                        "wrap up and call Done(); avoid starting new "
                        "delegations. A hard cost backstop applies only at "
                        f"{int(run_backstop_multiple())}x budget. Do NOT cancel "
                        "a delegation that is still progressing to save time — "
                        "its ledgered evals already persist, so cancelling only "
                        "throws away its report; let it finish and read it."
                    ),
                })
            elif pct >= 0.95:
                warnings.append({
                    "role": "user",
                    "content": (
                        f"Warning: time budget at {pct*100:.0f}% "
                        f"({elapsed:.0f}s / {budget:.0f}s). "
                        "Begin wrapping up — call Done() soon. Don't cancel a "
                        "progressing delegation under time pressure; its evals "
                        "are already ledgered and cancelling only loses its "
                        "report (GetStatus shows whether it's progressing)."
                    ),
                })

        eval_budget = state.get("eval_budget")
        evals_used = self._ledgered_eval_total(state.get("evals_used", 0))
        if eval_budget is not None and evals_used >= eval_budget:
            warnings.append({
                "role": "user",
                "content": (
                    f"Warning: eval budget exceeded"
                    f" ({evals_used} used / {eval_budget} budget)."
                    f" Do not run further evaluations."
                ),
            })
        return warnings

    def _reset_for_turn(self) -> list[str]:
        """A1/A2: clear per-turn state; return the notifications to deliver.

        Working/FollowUp entries are preserved so loopbacks don't orphan live
        delegations whose background threads are still running.
        """
        self._route.clear()
        self._ask_count = 0
        self._done_warned = False
        self._science_injected_this_turn = False
        with self._registry_lock:
            self._registry = {
                d: e for d, e in self._registry.items()
                if e["status"] in ("Working", "FollowUp", "Done")
            }
            self._threads = {
                d: t for d, t in self._threads.items()
                if d in self._registry
            }
        with self._notifications_lock:
            pending = list(self._notifications)
            self._notifications.clear()
        return pending

    # ── What the agent is told this turn ─────────────────────────────────────

    def _compose_messages(
        self,
        state: AgenticState,
        budget_warnings: list[dict],
        pending_notifs: list[str],
    ) -> list[dict]:
        """The conversation plus everything injected in-band this turn."""
        return (
            _to_adapter_messages(state["messages"])
            + budget_warnings
            + self._no_source_nudge()
            + self._backlog_announcement()
            + [{"role": "user", "content": n} for n in pending_notifs]
        )

    def _no_source_nudge(self) -> list[dict]:
        """Recommend registering a canonical source (soft, ≤3×).

        If this graph has a datagenerator (so a canonical ground-truth source
        CAN be authored) but none is registered, recommend delegating to it.
        Without a registered source every evaluation lands off-ledger and
        nothing is reproducible from the canonical store. Soft and capped —
        never blocks; the strategizer may ignore it for a genuinely
        source-free study.
        """
        if (
            self._no_source_nudges >= 3
            or self._find_datagenerator_name() is None
            or self._canonical_source_registered()
        ):
            return []
        self._no_source_nudges += 1
        _dg = self._find_datagenerator_name()
        self._record_intervention(
            "NO_SOURCE_NUDGE", self._name,
            "No canonical source registered; recommended delegating to "
            f"'{_dg}'.",
            notice=self._no_source_nudges, cap=3,
        )
        return [{
            "role": "user",
            "content": (
                "[SETUP] No canonical ground-truth source is registered "
                "for this study (no evaluator entrypoint or lookup pool). "
                f"A '{_dg}' agent is available — delegate to it to author "
                "and register the source, so evaluations flow through "
                "get_evaluator(), land in the canonical store, and the "
                "result is reproducible. If this is intentionally a "
                "source-free (surrogate-only) study, disregard this. "
                f"(notice {self._no_source_nudges}/3)"
            ),
        }]

    def _backlog_announcement(self) -> list[dict]:
        """Announce the process backlog ONCE, at the start of the run.

        As a conversation message, so the agent cannot claim it didn't know
        these gate the implementer. Injected the first time this node runs.
        """
        if self._milestones is None or getattr(self, "_backlog_announced", False):
            return []
        from ..epistemics.milestones import render_backlog
        _bl = render_backlog(self._milestones)
        self._backlog_announced = True
        return [{"role": "user", "content": _bl}] if _bl else []

    def _invoke_turn(self, messages: list[dict]) -> Any:
        """Run one model turn and account for what it spent."""
        from langchain_core.messages import AIMessage

        # DEBUG: stream this turn's full reasoning + tool-calls to
        # debug/transcripts/strategizer/turn_NNN.jsonl.
        from ..backends.base import (
            debug_enabled as _dbg,
        )
        from ..backends.base import (
            set_transcript_sink as _set_sink,
        )
        self._turn_count = getattr(self, "_turn_count", 0) + 1
        if _dbg() and self._current_notes_dir is not None:
            _set_sink(str(
                self._current_notes_dir.parent / "transcripts"
                / "strategizer" / f"turn_{self._turn_count:03d}.jsonl"))
        text = self.adapter.invoke(messages)
        # Accumulate this node's own token usage.
        self._record_usage(
            getattr(self.adapter, "last_usage", {}) or {},
            role=self._role_of(self._name),
            model=getattr(self.adapter, "model", None),
            phase="strategizer_turn",
            delegation_id=None,
        )
        return AIMessage(content=text)

    # ── How the turn ends ────────────────────────────────────────────────────

    def _route_turn(self, state: AgenticState, ai_msg: Any) -> Any:
        """A turn ends in exactly one of three ways.

        1. Work is still in flight     → re-prompt, free (no attempt spent)
        2. It stopped without closing  → re-prompt, bounded (3 attempts)
        3. Otherwise                   → the run ends

        Reproduction is owned entirely by the Done() gate (it runs the
        controlled gate before any close and declares a FAILED run after a
        bounded number of sighted attempts — see CheckDeliverable), so there is
        no separate post-accept repro check here; this handles only deliverable
        presence and un-accepted termination.
        """
        accepted = self._route.get("kind") == "done"
        missing = self._missing_deliverables(state)
        for router in (self._reprompt_while_working,
                       self._reprompt_unfinished):
            held = router(ai_msg, accepted, missing)
            if held is not None:
                return held
        return self._terminate_run(state, ai_msg, accepted, missing)

    def _reprompt_while_working(
        self, ai_msg: Any, accepted: bool, missing: list
    ) -> Any | None:
        """Delegations still running: re-prompt WITHOUT spending an attempt.

        A healthy delegation still in flight is WORK IN PROGRESS, not a failed
        finish: the deliverables usually depend on its result, and it WILL
        report. Spending a bounded finish-attempt on it means a slow-but-healthy
        delegation (run-4: D004 at ~2.5 evals/s, ~100s from done, with wall
        budget to spare) burns 3 "finish attempts" across turns and force-
        terminates the run UNGATED. The run's time backstop
        (run_backstop_multiple x budget, checked each turn) bounds a delegation
        that truly hangs.
        """
        from langchain_core.messages import HumanMessage
        from langgraph.types import Command

        if accepted:
            return None
        working = self._working_delegations()
        if not working:
            return None
        msg = (
            f"Delegations still running: {working}. They are"
            " progressing — poll with GetStatus() and call Done() only"
            " once they report (then write any remaining deliverables"
            " from their results). Do NOT close early. This wait does"
            " NOT count against your finish attempts; the run's time"
            " budget is the backstop."
        )
        if missing:
            msg += "\n\nStill to write AFTER they finish: " + ", ".join(missing)
        return Command(
            goto=self._name,
            update={"messages": [ai_msg, HumanMessage(content=msg)]},
        )

    def _reprompt_unfinished(
        self, ai_msg: Any, accepted: bool, missing: list
    ) -> Any | None:
        """Bounded re-prompt when the turn ended without an accepted close."""
        from langchain_core.messages import HumanMessage
        from langgraph.types import Command

        if (accepted and not missing) or self._finish_attempts >= 3:
            return None
        self._finish_attempts += 1
        problems: list[str] = []
        if missing:
            missing_list = "\n".join(f"- {p}" for p in missing)
            problems.append(
                "Required deliverables are missing from the"
                f" study directory:\n{missing_list}\n"
                "Write them via WriteDeliverable() before"
                " calling Done()."
            )
        if not accepted:
            working = self._working_delegations()
            if working:
                problems.append(
                    f"Delegations still running: {working}."
                    " Poll them with GetStatus() and call Done()"
                    " once they finish."
                )
            else:
                problems.append(
                    "You ended your turn without an accepted"
                    " Done(). If Done() was refused (critic"
                    " verdict, two-shot confirmation, or another"
                    " gate), address the refusal and call Done()"
                    " again. A run only closes through an"
                    " accepted Done()."
                )
        return Command(
            goto=self._name,
            update={
                "messages": [
                    ai_msg,
                    HumanMessage(content=(
                        "Run cannot complete"
                        f" (attempt {self._finish_attempts}/3):\n"
                        + "\n\n".join(problems)
                    )),
                ],
            },
        )

    def _working_delegations(self) -> list[str]:
        """Delegation ids still in flight right now."""
        with self._registry_lock:
            return [
                d for d, e in self._registry.items()
                if e["status"] in ("Working", "FollowUp")
            ]

    def _terminate_run(
        self, state: AgenticState, ai_msg: Any, accepted: bool, missing: list
    ) -> Any:
        """Close the run: final counts, banner, ghost flush, terminal Command."""
        from langgraph.graph import END
        from langgraph.types import Command

        # total_new: only delegations created THIS turn (seq delta vs the
        # snapshot taken at turn start), not Done entries from prior turns.
        with self._registry_lock:
            total_new = self._delegation_seq - self._seq_at_turn_start
            evals_new = sum(e["evals"] for e in self._registry.values())

        summary = self._banner(
            self._route.get("summary") or ai_msg.content, accepted, missing)
        self._flush_ghost_delegations()

        # The persisted (reported) eval total prefers the ledger aggregate over
        # the accumulator: the accumulator can drop evals a namespace-blind
        # guard mis-flagged as off-ledger, and never saw namespace stores at
        # all. The ledger across all namespaces is authoritative → run_status.
        _evals_persist = self._ledgered_eval_total(
            state.get("evals_used", 0) + evals_new)
        return Command(
            goto=END,
            update={
                "messages": [ai_msg],
                "done": True,
                "last_report": summary,
                "total_delegations": state["total_delegations"] + total_new,
                "evals_used": _evals_persist,
                "token_totals": dict(self._token_totals),
                "error_counts": dict(self._error_counts),
            },
        )

    def _banner(self, summary: str, accepted: bool, missing: list) -> str:
        """Prepend the UNGATED banner when the run ends without an accepted Done().

        A FAILED-reproduction close carries its own ⛔ banner in the route
        summary and IS accepted=done, so it is not re-banner'd here.
        """
        if accepted and not missing:
            return summary
        flags = []
        if not accepted:
            flags.append(
                "the run terminated WITHOUT an accepted Done() —"
                " the final conclusions did NOT pass the"
                " adversarial critic gate"
            )
        if missing:
            flags.append(f"required deliverables missing: {missing}")
        return (
            "## ⚠ UNGATED RUN\n\n"
            "This run is NOT validated: " + "; ".join(flags) +
            ".\nTreat all conclusions below as unaudited.\n\n---\n\n"
            + summary
        )

    def _flush_ghost_delegations(self) -> None:
        """Close out delegations whose threads die with the interpreter.

        Daemon threads still alive when the run closes are killed at process
        exit — their run() never reaches the DONE/FAILED record write, leaving
        orphan RUNNING entries in the log. Write an INTERRUPTED terminal record
        for each so query_all() (last-wins) collapses to a closed state instead
        of RUNNING.
        """
        with self._registry_lock:
            live = [
                (did, dict(entry))
                for did, entry in self._registry.items()
                if entry.get("status") in ("Working", "FollowUp")
            ]
        if not live or self._delegation_log is None:
            return
        _now = datetime.now(tz=timezone.utc).isoformat(timespec="seconds")
        for _did, _entry in live:
            self._delegation_log.record(
                id=_did,
                from_node=self._name,
                to_node=_entry.get("target", "unknown"),
                # An interrupted delegation never reached _finish_ok/_error,
                # so its partial writes are still uncommitted. Commit them
                # HERE, against the delegation that made them, or the next
                # delegation to commit absorbs them and the history says the
                # wrong worker wrote those files.
                workspace_sha=self._commit_workspace(
                    f"{_did} {self._name} -> "
                    f"{_entry.get('target', 'unknown')} [INTERRUPTED]"),
                task="",
                deliverable=(
                    "INTERRUPTED: run closed while this delegation was "
                    "still running (background thread killed at process exit)"
                ),
                hypothesis_ids=_entry.get("hypothesis_ids") or [],
                started_at=_entry.get("started_at") or "",
                completed_at=_now,
                status="INTERRUPTED",
                tokens_in=0,
                tokens_out=0,
                cost_usd=None,
                is_falsification_attempt=bool(
                    _entry.get("is_falsification_attempt")
                ),
                evals=_entry.get("evals", 0),
                phase=_entry.get("phase"),
            )
