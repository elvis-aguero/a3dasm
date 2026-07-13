"""StrategizerNode: orchestrator node for the f3dasm agentic runtime.

``inspect.getsource(StrategizerNode.__call__)`` reads the full routing logic.
"""

from __future__ import annotations

import threading
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..graph_state import AgenticState

from ..delegation_log import DelegationLog
from ..hypothesis_ledger import HypothesisLedger
from ..science_monitor import ScienceMonitor
from ._constants import run_backstop_multiple
from .base import AgentNode
from .critic_gate import CriticGateMixin
from .lifecycle import LifecycleMixin
from .parsing import _to_adapter_messages
from .recording import RecordingMixin


def _headline_consistency(stdout: str) -> str | None:
    """Cross-check the notebook's STATED answer against its COMPUTED one.

    If the deliverable prints BOTH a freshly-computed ``REPRODUCED: <v>`` and a
    ``CLAIMED_HEADLINE: <v>`` (the value its write-up states), assert they
    agree within a relative tolerance. Returns an error string on mismatch,
    else None. LENIENT by design: if either marker is absent it returns None,
    so it adds no new failure mode (and no wait) when the convention isn't used
    — it only catches an internally self-contradicting deliverable (run
    20260705T181941: prose said 0.3644 while an idxmax cell printed a 0.3648
    noise row, and the gate waved it through for 4 rounds).
    """
    import re as _re

    def _grab(tag: str):
        m = _re.search(
            tag + r":\s*([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)",
            stdout or "")
        return float(m.group(1)) if m else None

    rep, claim = _grab("REPRODUCED"), _grab("CLAIMED_HEADLINE")
    if rep is None or claim is None:
        return None
    if abs(rep - claim) > 1e-9 + 1e-3 * abs(claim):
        return (
            f"Headline inconsistency: the write-up states CLAIMED_HEADLINE="
            f"{claim} but the notebook's own computation prints REPRODUCED="
            f"{rep}. The reported answer must be what the notebook computes — "
            "fix the selection (e.g. an idxmax picking a noise/near-duplicate "
            "row) or the prose so the stated and computed headlines agree.")
    return None


class StrategizerNode(RecordingMixin, CriticGateMixin, LifecycleMixin, AgentNode):
    """Orchestrator: reads Reports, decides next Delegation or Done/Ask."""

    def __init__(
        self,
        adapter: Any,
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
        super().__init__(adapter)
        self._name = name
        self._outgoing = list(outgoing)
        self._spec = spec
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
        from ..milestones import MilestoneLedger
        from ..settings import get_bool
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
        from ..telemetry import Telemetry
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
        return text

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

    def _build_hypothesis_closures(self) -> dict:
        """Build HypothesisPropose/Update/List/Get closures."""
        node = self

        from ..tool_catalog import tool_examples

        @tool_examples(
            "HypothesisPropose('Optimal t/L is near 0.08 — thin walls maximise "
            "buckling', 'a point with t/L in [0.10,0.14] beats "
            "buckling_load_norm 1.47', 'best at t/L ~ 0.08 ± 0.02', 0.55)",
        )
        def HypothesisPropose(
            statement: str,
            falsification_criterion: str,
            prediction: str,
            prior: float,
        ) -> str:
            """Propose a hypothesis. Returns its ID (H1, H2, …) or ERROR.

            statement: ONE falsifiable claim (no compound claims).
            falsification_criterion: what observation would kill it.
            prediction: the measurable outcome you expect.
            prior: plausibility in (0, 1). Past 3 OPEN you are asked to
            confirm (re-submit the same proposal) rather than blocked —
            tracking several at once is fine, e.g. one per design."""
            if node._ledger is None:
                return (
                    "ERROR: hypothesis ledger not available in this run."
                )
            return node._ledger.propose(
                statement=statement,
                falsification_criterion=falsification_criterion,
                prediction=prediction,
                prior=prior,
                proposed_by=node._name,
            )

        @tool_examples(
            "HypothesisUpdate('H1', 'SUPPORTED', 'sweep top t/L=0.09 beats "
            "threshold', 0.80, evidence={'delegation': 'D001', 'numbers': "
            "{'top_tL': 0.09, 'buckling_load_norm': 1.47}})",
        )
        def HypothesisUpdate(
            hypothesis_id: str,
            status: str,
            comment: str,
            posterior: float,
            evidence: dict | None = None,
        ) -> str:
            """Update hypothesis status with evidence and updated belief.

            status: OPEN | SUPPORTED | FALSIFIED | INCONCLUSIVE.
            posterior: your updated belief in [0, 1] — always required.
            evidence: {"delegation": "D###", "numbers": {key: value}}
              required for closing statuses; numbers should cite values
              from that delegation's report.
            SUPPORTED requires a completed falsification attempt targeting
              this hypothesis first — Delegate(..., is_falsification_attempt=True,
              hypothesis_ids=[hypothesis_id]) and wait for it to complete.
            RETRACTING SUPPORTED/INCONCLUSIVE back to OPEN (e.g. you marked it
              SUPPORTED but no falsification ATTEMPT was made — Charter §2) needs
              NO new evidence: pass evidence=None and explain in the comment; the
              evidence the verdict was based on is carried forward. Use this to
              fix your own premature close instead of leaving a contradiction.
              (Un-falsifying a FALSIFIED hypothesis still needs new evidence — it
              is a new claim, not a retraction.)
            triggered_by is auto-injected from last completed
            delegation."""
            if node._ledger is None:
                return (
                    "ERROR: hypothesis ledger not available in this run."
                )
            if isinstance(evidence, str):
                import json as _json
                try:
                    evidence = _json.loads(evidence)
                except _json.JSONDecodeError:
                    return (
                        "ERROR: evidence must be a JSON object like "
                        '{"delegation": "D004", "numbers": {...}}.'
                    )
            # SUPPORTED without a completed falsification attempt: a TWO-SHOT
            # CONFIRM, not a hard block (§4, user-approved this session). A verdict
            # is reversible when justified in writing, so this is a deliberate
            # pause — not an impossible action. First call nudges; a re-call with
            # a written justification in `comment` confirms. The verdict validator
            # and the gate critic remain the downstream falsification floor.
            if status == "SUPPORTED" and node._delegation_log is not None:
                completed = [
                    r for r in node._delegation_log.query_all()
                    if r.get("status") == "DONE"
                    and r.get("is_falsification_attempt")
                    and hypothesis_id in (r.get("hypothesis_ids") or [])
                ]
                if not completed:
                    if not hasattr(node, "_supported_confirm_pending"):
                        node._supported_confirm_pending = set()
                    h = node._ledger.get(hypothesis_id) if node._ledger else {}
                    crit = (h or {}).get("falsification_criterion", "(none set)")
                    _justified = len((comment or "").strip()) >= 30
                    if (hypothesis_id not in node._supported_confirm_pending
                            or not _justified):
                        node._supported_confirm_pending.add(hypothesis_id)
                        return (
                            f"[CONFIRM] You are marking {hypothesis_id} SUPPORTED "
                            "without a completed falsification attempt on record. "
                            "The Popperian charter asks that a hypothesis be "
                            "challenged before it is accepted — the clean path is "
                            "to delegate a refutation test "
                            "(is_falsification_attempt=True, "
                            f"hypothesis_ids=['{hypothesis_id}']), or "
                            "LinkFalsificationAttempt if one already ran. If you "
                            "have genuine grounds to accept it WITHOUT that, re-call "
                            "HypothesisUpdate with the same status and a written "
                            "justification in `comment` (a sentence on why SUPPORTED "
                            "holds and how it could still be refuted) — that "
                            f"confirms. Falsification criterion: {crit!r}"
                        )
                    node._supported_confirm_pending.discard(hypothesis_id)
            # Single-source attribution (the principle behind what used to be a
            # prompt format-rule): a closing verdict cites the ONE delegation whose
            # report contains the cited numbers. A list / comma-joined value can't
            # be attributed to a single source, so reject it here (the tool is the
            # right place for this, not the prompt — CLAUDE.md §2).
            ev = evidence or {}
            d_cited = ev.get("delegation")
            if isinstance(d_cited, (list, tuple)) or (
                isinstance(d_cited, str) and "," in d_cited
            ):
                return (
                    f"ERROR: evidence['delegation'] for {hypothesis_id} must name "
                    "ONE delegation — the one whose report contains the cited "
                    f"numbers — not several ({d_cited!r}). A verdict has to be "
                    "attributable to a single source; cite the authoritative one "
                    "and mention the others in `comment` or the numbers dict."
                )
            if (
                d_cited is not None
                and d_cited != "D000"
                and node._delegation_log is not None
            ):
                completed_ids = {
                    r["id"] for r in node._delegation_log.query_all()
                    if r.get("status") == "DONE"
                }
                if d_cited not in completed_ids:
                    return (
                        f"ERROR: {hypothesis_id} cites evidence from {d_cited!r}, "
                        "which is not a completed delegation. Only cite completed "
                        "delegations (status DONE). Check GetStatus or the "
                        "delegation log — if the delegation hasn't finished, wait "
                        "for it."
                    )
            # Provenance: a verdict is attributed to the source it RESTS ON, so
            # `triggered_by` is the delegation the agent CITED as evidence
            # (validated above as a single completed delegation / the D000
            # ground-truth anchor). Only when no evidence delegation was cited
            # (non-closing updates) fall back to the most-recently-completed
            # delegation. Previously this always used last_completed_id, which
            # mislabelled the audit trail whenever an unrelated delegation
            # finished after the cited one (run 20260706T204732: H5 cited D011
            # but recorded triggered_by=D013).
            triggered_by: str | None = None
            if isinstance(d_cited, str) and d_cited:
                triggered_by = d_cited
            elif node._delegation_log is not None:
                triggered_by = node._delegation_log.last_completed_id(
                    node._name)
            if triggered_by is None:
                with node._registry_lock:
                    done_entries = [
                        (d_id, entry)
                        for d_id, entry in node._registry.items()
                        if entry.get("status") in ("Done", "Errored")
                    ]
                if done_entries:
                    triggered_by = done_entries[-1][0]
            result = node._ledger.update(
                hypothesis_id,
                status,
                comment,
                evidence,
                posterior,
                triggered_by,
            )
            # #9: advisory live verdict-substance check — closing verdicts only,
            # and only when a NEW entry was actually appended ("Updated …"); not
            # on ERROR/SETTLED no-ops. Non-blocking: it appends a charter critique
            # to what the agent sees this turn, but never changes the update.
            from ..verdict_validator import CLOSING_STATUSES as _CLOSING
            if status in _CLOSING and result.startswith("Updated "):
                advisory = node._run_verdict_validator(
                    hypothesis_id, status, comment, evidence,
                )
                if advisory:
                    result = f"{result}\n{advisory}"
            return result

        def LinkFalsificationAttempt(
            delegation_id: str, hypothesis_id: str
        ) -> str:
            """Retroactively mark a completed delegation as a falsification
            ATTEMPT of a registered hypothesis (the read-time safety net for
            when the attempt was not declared up front at Delegate time).

            Links ONLY — it does NOT record a verdict and CANNOT change the
            hypothesis's pre-registered prediction. You must still call
            HypothesisUpdate to record the verdict, judged against that
            immutable prediction. Link only if the delegation genuinely tested
            the prediction — never retrofit an exploratory result."""
            if node._ledger is None:
                return (
                    "ERROR: hypothesis ledger not available in this run."
                )
            h_entry = node._ledger.get(hypothesis_id)
            if h_entry is None:
                return f"ERROR: hypothesis {hypothesis_id!r} not found."
            with node._registry_lock:
                entry = node._registry.get(delegation_id)
                if entry is None:
                    return (
                        f"ERROR: unknown delegation {delegation_id!r}. "
                        f"Known: {list(node._registry)}"
                    )
                if entry.get("status") != "Done":
                    return (
                        f"ERROR: {delegation_id} is not a completed (Done) "
                        "delegation; cannot link it as a falsification "
                        "attempt."
                    )
                entry["is_falsification_attempt"] = True
                hids = entry.get("hypothesis_ids") or []
                if hypothesis_id not in hids:
                    hids = [*hids, hypothesis_id]
                entry["hypothesis_ids"] = hids
                entry["reconciled"] = True
            if node._delegation_log is not None:
                node._delegation_log.mark_attempt(
                    delegation_id, hypothesis_id)
            pred = (
                h_entry.get("prediction")
                or h_entry.get("falsification_criterion")
                or "(no prediction on record)"
            )
            return (
                f"Linked {delegation_id} as a falsification attempt of "
                f"{hypothesis_id} (post-hoc). Pre-registered prediction: "
                f"\"{pred}\". Now record the VERDICT: "
                f"HypothesisUpdate('{hypothesis_id}', "
                "status=SUPPORTED|FALSIFIED|INCONCLUSIVE, posterior=…, "
                f"evidence={{'delegation': '{delegation_id}', "
                "'numbers': {…}}), judging THIS report against that "
                "prediction. Linking does NOT record a verdict."
            )

        # HypothesisList / HypothesisGet (read-only) live in the shared,
        # declaration-gated builder (build_declared_shared_closures) so leaf
        # workers can be granted them too. This builder returns only the
        # MUTATE tools, which stay orchestrator-only.
        return {
            "HypothesisPropose": HypothesisPropose,
            "HypothesisUpdate": HypothesisUpdate,
            "LinkFalsificationAttempt": LinkFalsificationAttempt,
        }

    def _build_milestone_closures(self) -> dict:
        """Build MilestoneList/Propose/Complete/Skip closures (process policy)."""
        node = self

        def MilestoneList() -> str:
            """List process milestones with id, status, description.

            Milestones are PROCESS steps (do X before Y; get Z ready), distinct
            from hypotheses (epistemics). Default milestones self-resolve when
            their condition is met; you author your own with MilestonePropose."""
            if node._milestones is None:
                return "Milestone ledger not available in this run."
            return node._milestones.format()

        def MilestonePropose(description: str) -> str:
            """Add your own process milestone. Returns its id (M1, M2, …).

            While pending, it joins the backlog that gates delegating to the
            implementer (exactly like the default milestones) — so use it to
            hold yourself to a process step you don't want to skip."""
            if node._milestones is None:
                return "ERROR: milestone ledger not available in this run."
            return node._milestones.propose(description)

        def MilestoneComplete(milestone_id: str, note: str) -> str:
            """Mark a milestone DONE. A brief `note` (one line on WHY it's
            satisfied — what was done / which delegation) is REQUIRED, so
            ticking is a deliberate, auditable act, not a rubber stamp."""
            if node._milestones is None:
                return "ERROR: milestone ledger not available in this run."
            if not note or not note.strip():
                return (
                    "ERROR: a brief note is required to complete a milestone — "
                    "one line on why it's satisfied (what you did / which "
                    "delegation). This keeps ticking honest and auditable."
                )
            return node._milestones.complete(milestone_id, note)

        def MilestoneSkip(milestone_id: str, reason: str) -> str:
            """Skip a milestone this study legitimately doesn't need (give a
            reason). The escape hatch so soft gates never deadlock you."""
            if node._milestones is None:
                return "ERROR: milestone ledger not available in this run."
            return node._milestones.skip(milestone_id, reason)

        return {
            "MilestoneList": MilestoneList,
            "MilestonePropose": MilestonePropose,
            "MilestoneComplete": MilestoneComplete,
            "MilestoneSkip": MilestoneSkip,
        }

    def _missing_deliverables(self, state: AgenticState) -> list[str]:
        """Return required deliverable paths not present at study_dir yet.

        The single deliverable (pipeline.ipynb) is always required, authored
        before Done() is accepted. It is the human-readable recipe AND the
        reproduction in one notebook: the runtime executes it
        lazily (see _reproduction_gate) to verify the headline re-derives from the
        ledger with zero new evals. Additional paths can be declared in
        state['required_deliverables'].
        """
        from ..notebook_exec import required_deliverable_name
        study_dir = Path(state.get("study_dir", "."))
        # WriteDeliverable writes BARE names to study_dir/ (it rejects path
        # separators). Normalise any configured path to its basename so a stray
        # 'workspace/…' prefix in a study config can't spuriously flag a present
        # deliverable as missing.
        required = [required_deliverable_name()] + list(
            state.get("required_deliverables") or [])
        seen: set[str] = set()
        missing: list[str] = []
        for p in required:
            name = Path(p).name
            if name in seen:
                continue
            seen.add(name)
            if not (study_dir / name).exists():
                missing.append(name)
        return missing

    def _reproduction_gate(self, state: AgenticState | None = None) -> str | None:
        """Execute pipeline.ipynb under a CONTROLLED reproduction gate.

        The binding reproducibility check. The pipeline must:
          (a) finish cleanly within a time ceiling (no heavy from-scratch run);
          (b) add ZERO new oracle rows (lazy: skip FINISHED evals);
          (c) NOT modify/delete existing ledger rows (integrity — no faking the
              zero-delta by delete+re-add or value rewrite);
          (d) print ``REPRODUCED: <value>`` — an informational headline marker
              for the critic/human; the runtime does NOT gate on it. Headline
              grounding (the value traces to a real ledger row) is the critic's
              HEADLINE PROVENANCE check, not an independent runtime extremum
              match (which wrongly rejected constrained optima).
        Returns None on PASS (and stashes ``self._repro_ok_detail``), else a
        problem string. Skips silently when there is no run context. Callable
        without ``state`` — study dir comes from ``self._study_dir``.

        On PASS the deliverable is a faithful, lightweight, lazy reproduction —
        not a script doing "sneaky stuff" unrelated to validating the pipeline.
        """
        import json as _json
        import re
        import shutil
        import subprocess
        import tempfile

        study_dir = (
            Path(self._study_dir) if getattr(self, "_study_dir", None) is not None
            else Path((state or {}).get("study_dir", "."))
        )
        # The deliverable is pipeline.ipynb. (A .py is still executable by the
        # executor-agnostic gate, kept only as a fallback for gate-logic tests;
        # the notebook is preferred when both are present.) Absence is left to
        # _missing_deliverables.
        deliverable = next(
            (study_dir / n for n in ("pipeline.ipynb", "pipeline.py")
             if (study_dir / n).exists()),
            None,
        )
        if deliverable is None:
            return None  # absence is handled by _missing_deliverables
        notes = self._current_notes_dir
        if notes is None:
            return None  # no run dir context (e.g. non-debug) — skip the gate
        run_dir = notes.parent.parent              # …/runs/<id>
        store_dir = run_dir / "experiment_data"
        run_config = run_dir / "debug" / "run_config.json"

        def _ledger_snapshot(store_root: Path) -> tuple[int, str]:
            """(row_count, content_hash) across EVERY store under store_root:
            the canonical/default store PLUS every design-namespace sibling
            (store_root/<namespace>/), via the same experiment_stores()
            aggregation LedgerBreakdown/ScienceMonitor already use. A single-
            store read here would miss a non-lazy write into a namespace
            store during "reproduction" — the sandbox copy this is called
            against is already namespace-complete (namespace stores nest
            under store_root), only the read needs to look past the default.

            content_hash is order-independent (sorted rounded values, tagged
            by store so identical values in two different stores can't
            false-collide) so a faithful lazy re-store doesn't false-trip it.
            """
            import hashlib

            from ..instrumented import experiment_stores

            total_rows = 0
            all_rows: list[tuple] = []
            for store in experiment_stores(store_root):
                try:
                    from f3dasm import ExperimentData
                    data = ExperimentData.from_file(project_dir=store)
                    _, out = data.to_pandas()
                except Exception:  # noqa: BLE001
                    continue
                total_rows += len(out)
                cols = [c for c in out.columns if not str(c).startswith("_")]
                if not cols:
                    continue
                vals = out[cols].round(10)
                all_rows.extend(
                    (store.name,) + tuple(r) for r in vals.to_numpy().tolist()
                )
            h = hashlib.sha256(repr(sorted(all_rows)).encode()).hexdigest()
            return total_rows, h

        # ── HERMETIC SANDBOX ──────────────────────────────────────────────────
        # CRITICAL: run the deliverable against a COPY of the canonical store, never
        # the live one. A faithful lazy pipeline adds nothing; a NON-lazy one
        # (re-evaluating) writes its evals into the THROWAWAY copy — we detect
        # that as "not lazy" while the real ledger stays pristine. Without this,
        # checking a non-lazy pipeline pollutes + inflates the canonical store
        # (and CheckDeliverable could be looped to balloon it without bound).
        before_n, before_hash = _ledger_snapshot(store_dir)
        if before_n == 0:
            return (
                "Canonical store has no rows — the campaign has not been "
                "evaluated yet. Run the delegation pipeline first so the "
                "ledger is populated, then the notebook can be reproduced "
                "lazily against those rows.")
        sandbox = Path(tempfile.mkdtemp(prefix="f3dasm_repro_"))
        try:
            sb_store = sandbox / "experiment_data"
            if store_dir.exists():
                shutil.copytree(store_dir, sb_store)
            else:
                sb_store.mkdir(parents=True, exist_ok=True)
            # A sandbox run_config so get_evaluator() also writes to the COPY
            # (it resolves the store from run_config["store_dir"], not the env).
            sb_run_config = sandbox / "run_config.json"
            if run_config.exists():
                _cfg = _json.loads(run_config.read_text())
            else:
                _cfg = {}
            _cfg["store_dir"] = str(sb_store)
            _cfg["lock_path"] = str(sb_store / "experiment_data" / ".lock")
            sb_run_config.write_text(_json.dumps(_cfg))

            from ..notebook_exec import sandbox_env
            env = sandbox_env(
                sb_store, sb_run_config, study_root=self._study_dir)
            _timeout = (
                max(0.1 * self._budget_seconds, 180.0)
                if self._budget_seconds else 300.0
            )
            try:
                # Executor-agnostic: a .ipynb runs via nbclient (in-env kernel),
                # a .py via subprocess — both return a CompletedProcess and raise
                # TimeoutExpired on timeout, so the asserts below are unchanged.
                from ..notebook_exec import run_deliverable
                proc = run_deliverable(
                    deliverable, cwd=sandbox, env=env, timeout=_timeout)
            except subprocess.TimeoutExpired:
                return (
                    f"{deliverable.name} did not finish within {_timeout:.0f}s. A "
                    "reproduction must be lightweight — load the ledger and skip "
                    "finished evals and heavy refits (cache-or-load surrogates). "
                    "Make it lazy.")
            after_n, after_hash = _ledger_snapshot(sb_store)
        finally:
            shutil.rmtree(sandbox, ignore_errors=True)

        # (a) clean exit — surface a generous stderr tail for sighted debugging.
        if proc.returncode != 0:
            return (
                f"{deliverable.name} FAILED to run (exit {proc.returncode}). It must "
                "load the ledger and derive the headline cleanly. Stderr:\n"
                + (proc.stderr or "")[-3000:]
                + ("\n\nStdout tail:\n" + proc.stdout[-800:]
                   if proc.stdout else ""))
        # (b) zero new evals (lazy).
        if after_n != before_n:
            return (
                f"{deliverable.name} is NOT lazy: re-running it changed the ledger row "
                f"count ({before_n} → {after_n}). It must LOAD the ledger "
                "(ExperimentData.from_file) and reach the oracle only via "
                "get_evaluator() so FINISHED rows are skipped — zero new evals.")
        # (c) integrity — existing rows unchanged.
        if before_hash and after_hash and before_hash != after_hash:
            return (
                f"{deliverable.name} MODIFIED existing ledger rows. A reproduction must "
                "read the ledger READ-ONLY (it may re-store identical rows, but "
                "must not rewrite values or delete+re-add). Do not tamper with "
                "the canonical store.")
        # (d) The printed ``REPRODUCED:`` line is an informational headline
        # marker for the critic / human reader — the runtime no longer gates on
        # it. Headline GROUNDING (the value traces to a real ledger row) is
        # owned by the critic's HEADLINE PROVENANCE check; an independent
        # runtime extremum match wrongly rejected legitimate CONSTRAINED optima
        # (a constrained best is, by definition, not an objective extremum), so
        # it forced studies to headline their infeasible unconstrained extremum
        # — see audit run 20260624T021359.
        # (e) internal consistency — if the notebook declares CLAIMED_HEADLINE
        # (the value its write-up states), it must equal the freshly-computed
        # REPRODUCED. Lenient: skips when the marker is absent.
        _hc = _headline_consistency(proc.stdout or "")
        if _hc is not None:
            return _hc
        m = re.search(r"REPRODUCED:\s*([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)",
                      proc.stdout or "")
        headline = f", REPRODUCED={m.group(1)}" if m else ""
        self._repro_ok_detail = (
            f"reproduced cleanly ({before_n} rows, unchanged, 0 new evals, "
            f"ran in <{_timeout:.0f}s{headline})")
        return None

    def __call__(self, state: AgenticState) -> Any:
        import time

        from langchain_core.messages import AIMessage, HumanMessage
        from langgraph.graph import END
        from langgraph.types import Command

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

        # Capture total_delegations so Delegate() can seed the counter.
        self._state_total_delegations = state.get("total_delegations", 0)
        # Seed the monotonic counter from state on first __call__ (or
        # after a checkpoint rebuild).  Never decremented — ensures IDs
        # are unique even when the registry is pruned between turns.
        if self._delegation_seq < self._state_total_delegations:
            self._delegation_seq = self._state_total_delegations
        # Snapshot seq at turn start so total_new counts only THIS call.
        self._seq_at_turn_start: int = self._delegation_seq

        # Time budget is a SOFT constraint — warnings only; the run is
        # never force-terminated for exceeding it. A separate run-level
        # backstop (RUN_BACKSTOP_MULTIPLE x budget) bounds runaway cost.
        budget_warnings: list[dict] = []
        budget = state.get("budget_seconds")
        start = state.get("start_time")
        # Store on node so GetStatus() can compute delegation timeout
        self._budget_seconds = budget
        self._run_start = start
        self._budget_usd = state.get("budget_usd")
        if budget is not None and start is not None:
            elapsed = time.time() - start
            pct = elapsed / budget
            if pct >= 1.0:
                budget_warnings.append({
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
                budget_warnings.append({
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
        evals_used = state.get("evals_used", 0)
        # The canonical ledger is the source of truth: a killed/cancelled
        # delegation flushes rows the state accumulator never sees, so the
        # accumulator undercounts and the budget warning never fires. Prefer the
        # ledger count (max() keeps the accumulator for lookup-direct studies
        # with no instrumented store).
        try:
            from ..instrumented import total_ledgered_evals
            _nd = getattr(self, "_current_notes_dir", None)
            if _nd is not None:
                # Sum across the canonical store AND every design namespace —
                # namespace evals were invisible to the run total + soft budget.
                _total = total_ledgered_evals(_nd.parent.parent / "experiment_data")
                evals_used = max(evals_used, int(_total))
        except Exception:  # noqa: BLE001
            pass
        if eval_budget is not None and evals_used >= eval_budget:
            budget_warnings.append({
                "role": "user",
                "content": (
                    f"Warning: eval budget exceeded"
                    f" ({evals_used} used / {eval_budget} budget)."
                    f" Do not run further evaluations."
                ),
            })

        _halt = self._check_unrecoverable(state, budget, start)
        if _halt is not None:
            return _halt

        # A1/A2: reset per-turn state so a reused node starts clean each call.
        # Working/FollowUp entries are preserved so loopbacks don't orphan live
        # delegations whose background threads are still running.
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
            _pending_notifs = list(self._notifications)
            self._notifications.clear()

        # ── No-canonical-source nudge (soft, ≤3×) ─────────────────────────
        # If this graph has a datagenerator (so a canonical ground-truth
        # source CAN be authored) but none is registered, recommend
        # delegating to it. Without a registered source every evaluation
        # lands off-ledger and nothing is reproducible from the canonical
        # store. Soft and capped — never blocks; the strategizer may ignore
        # it for a genuinely source-free study.
        registration_nudge: list[dict] = []
        if (
            self._no_source_nudges < 3
            and self._find_datagenerator_name() is not None
            and not self._canonical_source_registered()
        ):
            self._no_source_nudges += 1
            _dg = self._find_datagenerator_name()
            registration_nudge.append({
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
            })
            self._record_intervention(
                "NO_SOURCE_NUDGE", self._name,
                "No canonical source registered; recommended delegating to "
                f"'{_dg}'.",
                notice=self._no_source_nudges, cap=3,
            )

        # Announce the process backlog ONCE, at the start, as a conversation
        # message — so the agent cannot claim it didn't know these gate the
        # implementer. Injected the first time the strategizer is invoked.
        backlog_announce: list = []
        if (self._milestones is not None
                and not getattr(self, "_backlog_announced", False)):
            from ..milestones import render_backlog
            _bl = render_backlog(self._milestones)
            if _bl:
                backlog_announce = [{"role": "user", "content": _bl}]
            self._backlog_announced = True

        _notif_msgs = [{"role": "user", "content": n} for n in _pending_notifs]
        messages = (
            _to_adapter_messages(state["messages"])
            + budget_warnings + registration_nudge + backlog_announce
            + _notif_msgs
        )
        # DEBUG: stream this strategizer turn's full reasoning + tool-calls
        # to debug/transcripts/strategizer/turn_NNN.jsonl.
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
        # Accumulate strategizer's own token usage.
        self._record_usage(
            getattr(self.adapter, "last_usage", {}) or {},
            role=self._role_of(self._name),
            model=getattr(self.adapter, "model", None),
            phase="strategizer_turn",
            delegation_id=None,
        )
        ai_msg = AIMessage(content=text)

        route = self._route
        accepted = route.get("kind") == "done"
        missing = self._missing_deliverables(state)
        # Reproduction is owned entirely by the Done() gate now (it runs the
        # controlled gate before any close and declares a FAILED run after a
        # bounded number of sighted attempts — see CheckDeliverable). So there is
        # no separate post-accept repro check here; this branch handles only
        # deliverable presence and un-accepted termination.

        # ── Transient: delegations still running ──────────────────────────────
        # A healthy delegation still in flight is WORK IN PROGRESS, not a failed
        # finish: the deliverables usually depend on its result, and it WILL
        # report. Re-prompt to poll WITHOUT consuming the bounded finish-attempt
        # budget — otherwise a slow-but-healthy delegation (run-4: D004 at ~2.5
        # evals/s, ~100s from done, with wall budget to spare) burns 3 "finish
        # attempts" across turns and force-terminates the run UNGATED. The run's
        # time backstop (run_backstop_multiple x budget, checked each turn)
        # bounds a delegation that truly hangs.
        if not accepted:
            with self._registry_lock:
                _working_now = [
                    d for d, e in self._registry.items()
                    if e["status"] in ("Working", "FollowUp")
                ]
            if _working_now:
                msg = (
                    f"Delegations still running: {_working_now}. They are"
                    " progressing — poll with GetStatus() and call Done() only"
                    " once they report (then write any remaining deliverables"
                    " from their results). Do NOT close early. This wait does"
                    " NOT count against your finish attempts; the run's time"
                    " budget is the backstop."
                )
                if missing:
                    msg += (
                        "\n\nStill to write AFTER they finish: "
                        + ", ".join(missing)
                    )
                return Command(
                    goto=self._name,
                    update={"messages": [ai_msg, HumanMessage(content=msg)]},
                )

        # ── Bounded re-prompt on unaccepted termination ───────────────────────
        if (not accepted or missing) and self._finish_attempts < 3:
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
                with self._registry_lock:
                    working = [
                        d for d, e in self._registry.items()
                        if e["status"] in ("Working", "FollowUp")
                    ]
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

        # ── Terminal branch ───────────────────────────────────────────────────
        # Accumulate delegation counts and evals from registry.
        # total_new: only delegations created THIS call (seq delta vs
        # snapshot taken at __call__ start), not Done entries from prior turns.
        with self._registry_lock:
            total_new = self._delegation_seq - self._seq_at_turn_start
            evals_new = sum(e["evals"] for e in self._registry.values())

        summary = route.get("summary") or text

        # Prepend UNGATED banner if the run ends without an accepted Done().
        # (A FAILED-reproduction close carries its own ⛔ banner in route summary
        # and IS accepted=done, so it is not re-banner'd here.)
        if not accepted or missing:
            flags = []
            if not accepted:
                flags.append(
                    "the run terminated WITHOUT an accepted Done() —"
                    " the final conclusions did NOT pass the"
                    " adversarial critic gate"
                )
            if missing:
                flags.append(
                    f"required deliverables missing: {missing}"
                )
            summary = (
                "## ⚠ UNGATED RUN\n\n"
                "This run is NOT validated: " + "; ".join(flags) +
                ".\nTreat all conclusions below as unaudited.\n\n---\n\n"
                + summary
            )

        # Flush ghost delegations: daemon threads that are still alive when the
        # run closes are killed by the interpreter at process exit — their _run()
        # never reaches the DONE/FAILED record write, leaving orphan RUNNING
        # entries in the log. Write an INTERRUPTED terminal record for each so
        # query_all() (last-wins) collapses to a closed state instead of RUNNING.
        with self._registry_lock:
            _live = [
                (did, dict(entry))
                for did, entry in self._registry.items()
                if entry.get("status") in ("Working", "FollowUp")
            ]
        if _live and self._delegation_log is not None:
            _now = datetime.now(tz=timezone.utc).isoformat(timespec="seconds")
            for _did, _entry in _live:
                self._delegation_log.record(
                    id=_did,
                    from_node=self._name,
                    to_node=_entry.get("target", "unknown"),
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

        # The persisted (reported) eval total prefers the ledger aggregate over
        # the accumulator: the accumulator can drop evals a namespace-blind guard
        # mis-flagged as off-ledger, and never saw namespace stores at all. The
        # ledger across all namespaces is the authoritative count → run_status.
        _evals_persist = state.get("evals_used", 0) + evals_new
        try:
            from ..instrumented import total_ledgered_evals
            _nd = getattr(self, "_current_notes_dir", None)
            if _nd is not None:
                _evals_persist = max(
                    _evals_persist,
                    int(total_ledgered_evals(_nd.parent.parent / "experiment_data")),
                )
        except Exception:  # noqa: BLE001
            pass
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


