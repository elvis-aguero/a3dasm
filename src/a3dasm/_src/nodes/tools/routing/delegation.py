"""Delegation-family tools: Delegate/Wait/GetStatus/CancelDelegation/Reply/
FollowUp/Confer/RecallHistory.

Three objects, one per scope that used to be a level of closure nesting:

``DelegationTools``  the tools an orchestrating node is handed, bound to that
                     node (``self.node``).
``WorkerSession``    one dispatched delegation — the tools the worker itself is
                     handed, plus the thread body that runs it and records the
                     outcome.
``ConferTools``      Confer, bound to one sender. Identical for the
                     orchestrator and for every worker, which is why it is its
                     own object rather than a factory.

``build_delegation_closures(node)`` at the bottom is the registration table:
tool name -> bound method. Tools stay PascalCase methods so their docstrings
remain the model-facing description (``prompts.tool_catalog`` reads ``__doc__``)
and stay readable from source by the viewer's AST scan; ``inspect.signature``
drops ``self``, so the JSON schema the backends infer is unchanged.
"""
from __future__ import annotations

import functools
import re
import threading
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ....prompts.tool_catalog import tool_examples
from ..._constants import backstop_enabled, run_backstop_multiple
from ...notices import wrap_notice
from ...parsing import (
    _classify_response,
    _reconcile_delegation_evals,
    _stamped_eval_count,
)

# Roles whose delegations actually reach the ground-truth oracle and so are
# subject to the eval-ledger guards (raw-oracle nudge, unledgered bounce,
# off-ledger reconciliation). Role-based (not node-name-based) so it stays
# forward-compatible across node renames. The literature_reviewer (no oracle),
# datagenerator (legitimately builds/validates the oracle), and critic/
# strategizer are NOT evaluators and must be exempt — else they get bounced /
# nudged for work that never touches get_evaluator(). DebuggerAgent inherits
# role "implementer"; "debugger" is listed too for when it carries its own.
_LEDGER_GUARD_ROLES = frozenset({"implementer", "debugger"})

# MCP tool errors are infrastructure faults, not agent faults, and surface only
# as text in the worker's report — scanned for so they are recorded as system
# errors rather than counted against the agent.
_MCP_ERROR_PATTERNS = (
    r"(mcp__\w+__\w+)[^\n]*?(429|rate.?limit|timeout|timed.?out|unavailable|connection.?error)",
    r"(HTTP\s+(?:429|500|502|503))[^\n]*",
    r"(rate.?limit(?:ed|ing)?)[^\n]*",
)


# Forward-compatible delegation-target resolution. Agents repeatedly name a
# target by CAPABILITY rather than the exact graph node name — e.g.
# "pipeline"/"pipeline_executor" for the implementer (the "pipeline executor"),
# "data_generation" for the datagenerator — and bounce off "unknown target".
# Resolve in order: exact node name -> normalized name (case/separator-
# insensitive) -> normalized role. Resolution is by live node name or role
# only — there is NO hardcoded capability-synonym table. The strategizer prompt
# names targets by their hint/role, so it does not invent capability words like
# 'pipeline'/'oracle'; an unresolvable target returns None and the caller errors
# with the valid-target list (the agent then self-corrects). Resolving by role
# (not node name) keeps it forward-compatible across node renames.


def _norm_target(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


def resolve_target(
    requested: str, outgoing: list[str], roles: dict[str, str]
) -> str | None:
    """Map a requested delegation target to a valid outgoing node name, or None.

    ``roles`` maps node name -> its configured role. Resolution is exact-name →
    normalized-name → normalized-role. Only a unique, confident match resolves;
    anything else returns None (caller errors)."""
    if requested in outgoing:
        return requested
    rn = _norm_target(requested)
    if not rn:
        return None
    for t in outgoing:  # normalized node name (case/separator-insensitive)
        if _norm_target(t) == rn:
            return t
    for t in outgoing:  # normalized role
        if _norm_target(roles.get(t, "")) == rn:
            return t
    return None


def with_doc(fn, doc: str):
    """Re-present a bound tool method with a per-node docstring.

    ``Delegate``'s description embeds this node's own connected targets, so its
    text is per-node while its code is not. ``update_wrapper`` carries across
    ``__name__``/``__dict__`` (hence ``@tool_examples``), and the leading
    ``__wrapped__`` keeps ``inspect.signature`` — and so the inferred JSON
    schema — identical to the method's.
    """
    bound = functools.partial(fn)
    functools.update_wrapper(bound, fn)
    bound.__doc__ = doc
    return bound


class ConferTools:
    """Confer, bound to one sender node.

    Works identically for the orchestrating node, a worker, or a peer —
    ``sender_name`` is the only difference. Async: never blocks. A message is
    queued in the TARGET's inbox and delivered when the target next drains
    (orchestrator: each turn via _drain_notifications; worker: collect-on-send
    the next time IT calls Confer). Faithful port of the stashed Confer design.
    """

    def __init__(self, node: Any, sender_name: str) -> None:
        self.node = node
        self.sender_name = sender_name

    def Confer(self, target: str, message: str) -> str:
        """Send an async message to another node in the run.

        Returns immediately — neither side blocks. Use it to correct or
        steer a delegation that is ALREADY RUNNING, rather than waiting
        for a wrong result and re-delegating.

        target is a node name, a delegation id (D004 — address a
        specific delegation when two of one role are running), or the
        orchestrating node. A running delegation gets the message
        prefixed onto its next tool result; an idle node's message waits
        until that node itself Confers. The reply tells you which
        happened — read it, because "delivered" and "queued" are
        different outcomes.

        Reply by convention with Confer(sender_name, "re #N: <answer>").
        """
        node, sender_name = self.node, self.sender_name
        with node._registry_lock:
            # A delegation id is a legitimate address: when two
            # delegations of one role are running, the role name cannot
            # say which is meant, and the sender is reduced to
            # broadcasting "ignore this if you are D003".
            by_id = node._registry.get(target)
            live = [
                did for did, e in node._registry.items()
                if e.get("target") == target
                and e.get("status") in ("Working", "FollowUp")
            ]
            ever_woken = (
                target == node._name
                or by_id is not None
                or any(e.get("target") == target
                       for e in node._registry.values())
            )
        if by_id is not None:
            live = (
                [target]
                if by_id.get("status") in ("Working", "FollowUp")
                else []
            )
        if not ever_woken:
            return (
                f"ERROR: {target!r} is neither a node that has been "
                "delegated to this run nor a delegation id — cannot "
                "Confer with something that was never woken."
            )
        seq = node._next_confer_seq()
        envelope = (
            f"[Confer #{seq} from {sender_name} → {target}]: {message}\n"
            f"→ reply with Confer(\"{sender_name}\", \"re #{seq}: "
            "<answer>\")"
        )
        # Deliver on the path that actually reaches a BUSY worker: the
        # per-delegation queue whose contents are prefixed onto that
        # worker's next tool result (the same mechanism the budget and
        # backstop warnings ride). The name-keyed _confer_inbox alone is
        # drained only collect-on-send — i.e. only if the recipient
        # happens to call Confer itself — so a mid-flight correction to a
        # worker that never calls Confer was accepted, reported as
        # queued, and silently never delivered.
        if live:
            with node._pending_worker_msgs_lock:
                for did in live:
                    node._pending_worker_msgs.setdefault(
                        did, []).append(envelope)
        with node._confer_inbox_lock:
            if by_id is None:
                node._confer_inbox.setdefault(target, []).append(envelope)
            # Collect-on-send: drain any messages addressed to this sender
            # so replies arrive alongside the send confirmation.
            inbox = node._confer_inbox.pop(sender_name, [])
        inbox_text = ("\n\n".join(inbox) + "\n\n") if inbox else ""
        # Say which it was. "Queued" for an idle target and "delivered"
        # to a running one are different outcomes, and the sender's next
        # move depends on which happened.
        if live:
            where = (
                f"Delivered to {len(live)} running delegation"
                f"{'s' if len(live) != 1 else ''} of {target!r} "
                f"({', '.join(sorted(live))}); it appears on their next "
                "tool result."
            )
        else:
            where = (
                f"Queued for {target!r} (confer #{seq}) — no delegation "
                "of it is running right now, so it is delivered only if "
                "that node itself Confers later. Nothing is waiting on "
                "it; do not block."
            )
        return inbox_text + where


class WorkerSession:
    """One dispatched delegation: the worker's own tools and its thread body.

    Constructed by :meth:`DelegationTools.Delegate` once the delegation has an
    id and a task message, then run on its own daemon thread. Everything the
    run needs is an attribute here — what used to be twelve names captured from
    an enclosing scope.

    ``run`` is the thread body and is deliberately a sequence of named phases:
    bind the backend context, invoke (with one corrective retry), harvest what
    the invocation produced, then record the outcome. Any exception from any of
    them lands in :meth:`_finish_error`, so a worker crash is always recorded as
    a FAILED delegation rather than a lost thread.
    """

    def __init__(
        self,
        node: Any,
        *,
        worker: Any,
        delegation_id: str,
        target: str,
        intent: str,
        task_msg: str,
        hypothesis_ids: list[str],
        is_falsification_attempt: bool,
        phase: str | None,
        namespace: str | None,
        started_at: str,
    ) -> None:
        self.node = node
        self.worker = worker
        self.delegation_id = delegation_id
        self.target = target
        self.intent = intent
        self.task_msg = task_msg
        self.hypothesis_ids = hypothesis_ids
        self.is_falsification_attempt = is_falsification_attempt
        self.phase = phase
        self.namespace = namespace
        self.started_at = started_at
        # Honour-system eval count, set by ReportEvals; reconciled against the
        # provenance-stamped ledger rows before it is believed.
        self.claimed_evals: int = 0
        # Set by _bind_backend_context, read by the report-retry and the
        # eval reconciliation.
        self.guard_agent: Any = None
        self.enforce_ledger: bool = False

    # ── The tools the WORKER itself is handed ────────────────────────────────

    def ReportEvals(self, count: int) -> str:
        """Report the total ground-truth evaluations you performed this
        task. Call once per task, ALWAYS — even if 0. When you use
        get_evaluator() the canonical ledger is authoritative for the
        count, but this call also ARMS the unledgered-evals safety check,
        so never skip it."""
        node = self.node
        self.claimed_evals = int(count)
        # Drain any queued budget warnings for this delegation.
        with node._pending_worker_msgs_lock:
            msgs = node._pending_worker_msgs.pop(self.delegation_id, [])
        prefix = wrap_notice("\n".join(msgs))
        return prefix + f"Recorded {count} evaluations."

    def FollowUp(self, question: str) -> str:
        """Ask your delegating party one clarifying question before proceeding.

        Routes to whoever sent you this task: the agent that delegated
        to you.  One FollowUp per delegation.  The answer is injected
        directly into your context.  If no answer arrives, proceed with
        best judgment.
        """
        node, delegation_id = self.node, self.delegation_id
        with node._registry_lock:
            entry = node._registry.get(delegation_id, {})
            if entry.get("followup_count", 0) >= 1:
                return (
                    "FollowUp limit reached (1 per delegation). "
                    "Proceed with best judgment."
                )
            node._registry[delegation_id]["followup_question"] = question
            node._registry[delegation_id]["followup_count"] = 1
            node._registry[delegation_id]["status"] = "FollowUp"
            evt = node._registry[delegation_id]["followup_event"]
        with node._notifications_lock:
            node._notifications.append(
                f"[{delegation_id} FollowUp: {question!r} "
                f"→ call Reply('{delegation_id}', answer)]"
            )
        evt.wait(timeout=300)  # 5-minute patience; proceed if no reply
        with node._registry_lock:
            answer = node._registry[delegation_id].get("followup_answer")
            node._registry[delegation_id]["status"] = "Working"
        # Drain any queued budget warnings alongside the answer.
        with node._pending_worker_msgs_lock:
            msgs = node._pending_worker_msgs.pop(delegation_id, [])
        budget_prefix = wrap_notice("\n".join(msgs))
        base = answer or "No answer received. Proceed with best judgment."
        return budget_prefix + base

    def ReportProgress(self, note: str) -> str:
        """Leave a short progress note (<=200 chars) your delegator sees
        when it polls you. NON-BLOCKING — you keep working immediately;
        no answer comes back. Use it so the delegator can tell you are
        making progress rather than stuck (which prevents needless
        cancellation): e.g. 'LHS done, 250 evals; fitting GP next' or
        'BO round 3/10, best f=-0.81 so far'.
        """
        _n = (note or "").strip()[:200]
        with self.node._registry_lock:
            e = self.node._registry.get(self.delegation_id)
            if e is not None:
                e["progress_note"] = (_n, time.monotonic())
        return "Progress noted (your delegator will see it on poll)."

    def install_worker_tools(self) -> None:
        """Grant this delegation's worker its own per-delegation tools."""
        node, worker, target = self.node, self.worker, self.target
        worker.closure_tools["ReportEvals"] = self.ReportEvals  # never errors
        worker.closure_tools["ReportProgress"] = self.ReportProgress
        worker.closure_tools["FollowUp"] = node._wrap_closure(
            self.FollowUp, target)
        # Confer: async messaging to the orchestrator (or any woken peer).
        # The worker drains its OWN inbox collect-on-send (see ConferTools).
        worker.closure_tools["Confer"] = node._wrap_closure(
            ConferTools(node, target).Confer, target)
        # ConsultHandbook is injected universally at adapter construction
        # (agent_runtime._make_adapter) — every node gets it equally there.

    # ── The thread body ──────────────────────────────────────────────────────

    def run(self) -> None:
        """Run the delegation to completion and record its outcome."""
        try:
            self._bind_backend_context()
            text = self._invoke_with_report_retry()
            self._record_oracle_nudges()
            usage = getattr(self.worker, "last_usage", {}) or {}
            self.node._record_worker_usage(
                self.worker, self.target, self.delegation_id)
            self._flag_mcp_errors(text)
            evals, off_ledger, stamped = self._reconcile_evals()
            text = self._append_budget_report(text)
            self._finish_ok(text, evals, usage, off_ledger, stamped)
        except Exception:  # noqa: BLE001
            self._finish_error(traceback.format_exc())

    def _bind_backend_context(self) -> None:
        """Bind this worker thread's backend context before it is invoked.

        The backend reads these thread-locals when it builds the worker's
        session environment, so they must be set on the worker's OWN thread —
        which is this one.
        """
        from ....backends.base import (
            debug_enabled as _dbg,
        )
        from ....backends.base import (
            set_delegation_id as _set_did,
        )
        from ....backends.base import (
            set_namespace as _set_ns,
        )
        from ....backends.base import (
            set_oracle_registered as _set_oracle_reg,
        )
        from ....backends.base import (
            set_run_config_path as _set_rc,
        )
        from ....backends.base import (
            set_transcript_sink as _set_sink,
        )
        node = self.node

        # Bind the delegation id for this worker thread so the
        # backend can inject F3DASM_DELEGATION_ID into the session
        # env → get_evaluator() resolves without a mandatory cd
        # into D### (audit Finding 2).
        _set_did(self.delegation_id)

        # Scope this worker to its design namespace (Axis 3a/3b) so the
        # backend injects F3DASM_NAMESPACE → get_evaluator() resolves the
        # namespace's oracle + ledger. None → single-study canonical path.
        _set_ns(self.namespace or None)

        # Bind the run_config.json path too, so the backend injects
        # F3DASM_RUN_CONFIG → get_evaluator() resolves by explicit path
        # rather than walking up from the worker's cwd (study_dir),
        # which can never reach runs/<id>/debug/run_config.json.
        _notes_for_rc = node._current_notes_dir
        if _notes_for_rc is not None:
            _set_rc(str(_notes_for_rc.parent / "run_config.json"))

        # The eval-ledger guards apply ONLY to evaluator roles
        # (implementer/debugger) AND only once a canonical oracle is
        # registered. Non-evaluator roles (literature_reviewer,
        # datagenerator, critic) never reach get_evaluator(), so
        # nudging/bouncing them is a false positive. This one flag
        # gates all three guards below.
        self.guard_agent = (
            node._spec.nodes.get(self.target) if node._spec else None
        )
        _target_role = getattr(self.guard_agent, "role", None)
        self.enforce_ledger = (
            node._canonical_source_registered()
            and _target_role in _LEDGER_GUARD_ROLES
        )
        _set_oracle_reg(self.enforce_ledger)

        # DEBUG: stream this worker's full reasoning + tool-calls
        # to debug/transcripts/{delegation_id}.jsonl (thread-local;
        # run() is the worker's own thread).
        if _dbg() and node._current_notes_dir is not None:
            _set_sink(str(
                node._current_notes_dir.parent / "transcripts"
                / f"{self.delegation_id}.jsonl"))

    def _invoke_with_report_retry(self) -> str:
        """Invoke the worker; one corrective retry if its report is malformed.

        Validated against THIS agent's declared report_sections (audit
        Finding 4 — report_sections is the single source of truth, not a
        hardcoded list), so e.g. a missing ### Retrospective earns one
        corrective retry.
        """
        from ....prompts.agent_prompts import build_report_retry_prompt

        messages = [{"role": "user", "content": self.task_msg}]
        text = self.worker.invoke(messages)

        _req_sections = list(
            getattr(self.guard_agent, "report_sections", None) or []
        ) or None
        diagnosis = _classify_response(text, _req_sections)
        if diagnosis is not None:
            retry_messages = messages + [
                {"role": "ai", "content": text},
                {
                    "role": "user",
                    "content": (
                        f"{build_report_retry_prompt(_req_sections)}"
                        f"\n\nDiagnosis: {diagnosis}"
                    ),
                },
            ]
            text = self.worker.invoke(retry_messages)
        return text

    def _record_oracle_nudges(self) -> None:
        """Log any raw-oracle nudge firings from this delegation.

        Direct evidence, drained from the adapter's own budget.
        """
        _onb = getattr(self.worker, "_oracle_nudge", None)
        for _ev in list(getattr(_onb, "events", []) or []):
            self.node._record_intervention(
                "RAW_ORACLE_NUDGE", self.target,
                f"{self.delegation_id}: a {_ev.get('tool')} call "
                "reached the oracle directly; nudged toward "
                "get_evaluator().",
                snippet=_ev.get("snip", ""),
            )
        if _onb is not None:
            _onb.events = []

    def _flag_mcp_errors(self, text: str) -> None:
        """Record MCP tool errors named in the report as infrastructure faults.

        MCP errors appear as lines containing "error" near tool names in the
        report; they are the provider's fault, not the agent's, and
        _record_tool_error classifies them as such.
        """
        for _pat in _MCP_ERROR_PATTERNS:
            for _m in re.finditer(_pat, text, re.IGNORECASE):
                self.node._record_tool_error(
                    self.target,
                    _m.group(1) if _m.lastindex and _m.lastindex >= 1
                    else "mcp_tool",
                    "MCP_REPORTED",
                    _m.group(0)[:200],
                )
                break  # one log entry per pattern match per delegation

    def _run_experiment_root(self) -> Path | None:
        """The run's experiment_data root, holding every namespace's store."""
        _notes = self.node._current_notes_dir
        return (
            _notes.parent.parent / "experiment_data"
            if _notes is not None else None
        )

    def _reconcile_evals(self) -> tuple[int, bool, int]:
        """Believe the ledger, not the worker's self-report.

        Counts this delegation's provenance-stamped rows across EVERY
        experiment store; falls back to the honour-system count only when it
        wrote no rows anywhere. Reading the canonical store alone undercounts a
        namespaced delegation to 0 and falsely flags it off-ledger (run
        20260627T011059 D006/annular: 100 real evals logged as 0).

        Returns ``(evals, off_ledger, stamped)``.
        """
        return _reconcile_delegation_evals(
            self._run_experiment_root(),
            self.delegation_id,
            self.claimed_evals,
            self.enforce_ledger,
        )

    def _append_budget_report(self, text: str) -> str:
        """Append the completion constraint snapshot and per-eval KPI footer.

        The snapshot is ALWAYS appended (unlike the KPI footer, which only
        makes sense when this delegation actually wrote ledger rows), so every
        delegation's report is budget-aware — not just the ones that happened
        to evaluate something. Taken from the single source of truth
        (constraint_snapshot.py), shared with the RUNNING entry logged at
        dispatch and the terminal record below.

        Best-effort: a KPI footer must never fail a delegation.
        """
        from ....runtime.constraint_snapshot import snapshot_for_node
        text = text + "\n\n" + snapshot_for_node(self.node).as_text()

        _run_exp = self._run_experiment_root()
        try:
            from ....evaluation.ledger_summary import (
                RunStateSummary,
                experiment_stores,
            )
            _summary = None
            for _st in (experiment_stores(_run_exp)
                        if _run_exp is not None else []):
                _s = RunStateSummary.from_store(_st)
                if _s is not None and _s.n_per_delegation.get(
                        self.delegation_id, 0) > 0:
                    _summary = _s
                    break
            if _summary is not None:
                # Peak RAM this delegation reached (watcher high-water, free)
                # against the hard cap, so the strategizer learns the memory
                # footprint like it learns the time cost. (Wall/eval budget is
                # in the constraint snapshot above, not recomputed here.)
                from ....infra.watchdog_cleanup import delegation_peak_rss
                _footer = _summary.delegation_footer(
                    self.delegation_id,
                    peak_rss_bytes=delegation_peak_rss(self.delegation_id),
                    ram_cap_bytes=self._mem_cap_bytes(),
                )
                if _footer:
                    text = text + _footer
        except Exception:  # noqa: BLE001
            pass
        return text

    def _mem_cap_bytes(self) -> int | None:
        """The run's hard per-delegation memory cap, or None if unreadable."""
        _notes = self.node._current_notes_dir
        if _notes is None:
            return None
        try:
            import json as _json
            _cfgp = _notes.parent.parent / "debug" / "run_config.json"
            if _cfgp.exists():
                return _json.loads(_cfgp.read_text()).get("mem_cap_bytes")
        except Exception:  # noqa: BLE001
            return None
        return None

    def _finish_ok(
        self,
        text: str,
        evals: int,
        usage: dict,
        off_ledger: bool,
        stamped: int,
    ) -> None:
        """Record a completed delegation: registry, log, handoffs, retrospective."""
        node, delegation_id, target = self.node, self.delegation_id, self.target
        from ....runtime.constraint_snapshot import snapshot_for_node
        snapshot = snapshot_for_node(node)

        with node._registry_lock:
            _detached = (
                node._registry[delegation_id].get("status") == "Cancelled"
            )
            if _detached:
                # CancelDelegation detached this while it ran: keep it
                # Cancelled and DISCARD the deliverable. Usage was
                # already recorded above (the worker did spend tokens).
                node._registry[delegation_id]["evals"] = evals
            else:
                node._registry[delegation_id].update({
                    "status": "Done",
                    "result": text,
                    "evals": evals,
                    "usage": usage,
                })
                # This target made progress → clear its consecutive
                # error streak (the repeated-errors halt is for a
                # target stuck failing, not one that recovers).
                node._consecutive_errors[target] = 0
        with node._notifications_lock:
            node._notifications.append(
                f"[Delegation {delegation_id} "
                + ("completed after cancellation — result discarded]"
                   if _detached else "Done]")
            )
        # Loud, single record of the off-ledger condition — on the
        # cancel/detach path too (which the bounce never reaches), so a
        # cancelled delegation that evaluated off-ledger can no longer
        # vanish silently.
        #
        # Unledgered evals are a CORRECTIVE flag, not a re-run: re-running a
        # whole campaign to re-ledger is wasted wall-time (it helped blow the
        # watchdog in run 20260627T045747), and the critic's headline-
        # provenance check at the gate is the real floor. Cooperative agents
        # rarely bypass get_evaluator() on purpose; when they do, the tip
        # says so — once.
        if off_ledger:
            node._record_intervention(
                "OFF_LEDGER_EVALS", target,
                f"{delegation_id} claimed {self.claimed_evals} evals but none "
                "are provenance-stamped in any experiment store — counted "
                "as 0"
                + (" (delegation was cancelled/detached)"
                   if _detached else "")
                + ". These didn't go through get_evaluator(), so they "
                "cannot anchor a reproducible headline — that wasn't the "
                "right way to evaluate. If this delegation's numbers feed "
                "your conclusion, re-run it through get_evaluator(); and "
                "route evaluations through get_evaluator() from the start "
                "next time. (Not re-run for you: re-running a whole "
                "campaign to re-ledger wastes wall-time.)",
                claimed=self.claimed_evals,
                stamped=stamped,
                detached=_detached,
            )
        # Write delegation record to graph-wide delegation log
        if node._delegation_log is not None:
            node._delegation_log.record(
                id=delegation_id,
                from_node=node._name,
                to_node=target,
                task=self.intent,
                deliverable=text,
                hypothesis_ids=self.hypothesis_ids,
                started_at=self.started_at,
                completed_at=datetime.now(
                    tz=timezone.utc
                ).isoformat(timespec="seconds"),
                status="DONE",
                tokens_in=(usage.get("input_tokens", 0) or 0),
                tokens_out=(usage.get("output_tokens", 0) or 0),
                cost_usd=usage.get("total_cost_usd"),
                is_falsification_attempt=bool(self.is_falsification_attempt),
                evals=evals,
                phase=self.phase,
                constraints=snapshot.as_dict(),
            )
            if node._science_monitor is not None:
                try:
                    node._science_monitor.on_delegation_complete(delegation_id)
                except Exception:  # noqa: BLE001
                    pass

        self._register_authored_evaluator()

        # Worker retrospective (every node has a 'job done'
        # moment — see _record_retrospective).
        node._record_retrospective(
            node._role_of(target), delegation_id, text
        )

    def _register_authored_evaluator(self) -> None:
        """Point the canonical entrypoint at an oracle this delegation authored.

        When a datagenerator delegation authors an oracle, it drops a
        registration.json manifest in its workspace. Repoint the canonical
        entrypoint at it so the next get_evaluator() (re-reads config each
        call) resolves it — no manual config edit. Best-effort: never fail a
        delegation over registration.
        """
        node, delegation_id = self.node, self.delegation_id
        _notes = node._current_notes_dir
        try:
            _tgt = node._spec.nodes.get(self.target)
            if (
                _notes is None
                or getattr(_tgt, "role", None) != "datagenerator"
            ):
                return
            _run_dir = _notes.parent.parent
            _ws = (
                _run_dir / "debug" / "delegations"
                / delegation_id / "generators"
            )
            _manifest = _ws / "registration.json"
            if not _manifest.exists():
                return
            import json as _json

            from ....runtime.run_setup import register_evaluator_entrypoint
            _m = _json.loads(_manifest.read_text())
            _gf = _m["generator_file"]
            _gf_path = Path(_gf)
            if not _gf_path.is_absolute():
                # Resolve against the manifest dir, then the run dir;
                # take the first that exists.
                for _base in (_ws, _run_dir):
                    _cand = (_base / _gf).resolve()
                    if _cand.exists():
                        _gf_path = _cand
                        break
            # A datagenerator delegation scoped to a design namespace
            # registers that namespace's oracle (its own isolated store); the
            # manifest may also name one. The delegation's namespace is
            # authoritative.
            _ns = self.namespace or _m.get("namespace") or None
            # "I extended the file that is already canonical; do not repoint
            # anything." Without a way to SAY that, a delegation told to
            # extend the existing oracle faced three instructions it could not
            # jointly satisfy: its role contract makes the manifest mandatory,
            # dropping a manifest repoints the canonical entrypoint, and its
            # brief forbade repointing. The only escape was to encode the
            # intent in a path — name the already-canonical file by absolute
            # path so the repoint lands where it already pointed — which two
            # delegations had to invent independently. Stating the intent is
            # better than a path trick that happens to work.
            _in_place = bool(_m.get("extends_canonical"))
            if _in_place:
                _ep = _gf_path
            else:
                _ep = register_evaluator_entrypoint(
                    _run_dir / "debug" / "run_config.json",
                    _gf_path,
                    _m["attr"],
                    output_names=_m.get("output_names"),
                    namespace=_ns,
                )
            # Provenance is recorded either way: which delegation touched the
            # canonical source is part of the run record, and skipping the
            # REPOINT must not also skip the RECORD.
            with node._notifications_lock:
                _ns_tag = f" ns={_ns}" if _ns else ""
                _verb = "extended in place" if _in_place else "registered"
                node._notifications.append(
                    f"[Evaluator {_verb} by {delegation_id}: {_ep}{_ns_tag}]"
                )
        except Exception:  # noqa: BLE001
            pass

    def _finish_error(self, tb: str) -> None:
        """Record a delegation whose worker raised: registry + FAILED log row."""
        node, delegation_id, target = self.node, self.delegation_id, self.target
        usage = getattr(self.worker, "last_usage", {}) or {}
        node._record_worker_usage(self.worker, target, delegation_id)
        with node._registry_lock:
            node._registry[delegation_id].update({
                "status": "Errored",
                "result": tb,
                "evals": self.claimed_evals,
                "usage": usage,
            })
            node._consecutive_errors[target] = (
                node._consecutive_errors.get(target, 0) + 1
            )
        with node._notifications_lock:
            node._notifications.append(
                f"[Delegation {delegation_id} Errored]"
            )
        if node._delegation_log is not None:
            from ....runtime.constraint_snapshot import snapshot_for_node
            node._delegation_log.record(
                id=delegation_id,
                from_node=node._name,
                to_node=target,
                task=self.intent,
                # Keep the TAIL: the root exception is on
                # the last line of a traceback.
                deliverable="ERROR: " + tb[-2000:],
                hypothesis_ids=self.hypothesis_ids,
                started_at=self.started_at,
                completed_at=datetime.now(
                    tz=timezone.utc
                ).isoformat(timespec="seconds"),
                status="FAILED",
                tokens_in=(usage.get("input_tokens", 0) or 0),
                tokens_out=(usage.get("output_tokens", 0) or 0),
                cost_usd=usage.get("total_cost_usd"),
                is_falsification_attempt=bool(self.is_falsification_attempt),
                phase=self.phase,
                constraints=snapshot_for_node(node).as_dict(),
            )


class DelegationTools:
    """The delegation-family tools, bound to one orchestrating node."""

    def __init__(self, node: Any) -> None:
        self.node = node

    # ── Per-node prose ───────────────────────────────────────────────────────

    def delegate_doc(self) -> str:
        """Delegate's model-facing description, with THIS node's targets."""
        node = self.node
        # Build target hints from each connected agent's description.
        _target_hints = "\n  ".join(
            f"{t}: {node._spec.nodes[t].description}"
            for t in node._outgoing
            if t in node._spec.nodes
        )
        return (
            "Fire a task to a connected agent.\n\n"
            "CHOOSE THE MODE DELIBERATELY — neither is the default-good answer:\n"
            "  wait=False (async): returns a D### ID immediately and the worker\n"
            "    runs in the background. Multiple workers can then be alive at\n"
            "    once — which is the ONLY way Confer (live worker-to-worker\n"
            "    messaging) can do anything, and the only way the run's wall-clock\n"
            "    is the longest single chain rather than the sum of every\n"
            "    delegation. Collect them with a bare Wait() per worker — it\n"
            "    returns whichever finishes first, so a fan-out costs no polling.\n"
            "  wait=True (sync): blocks until the worker finishes and returns its\n"
            "    report directly, with zero polling. Simpler when this task must\n"
            "    fully finish before you can even decide the next one.\n"
            "  Ask yourself: could this run alongside other work, or might a peer\n"
            "  worker need to Confer with it mid-flight? If yes, async. If it is a\n"
            "  hard prerequisite for your very next decision, sync. Decide per\n"
            "  delegation; do not pick one mode reflexively for the whole run.\n\n"
            "CONTEXT PACKAGING: workers start each delegation with no memory of\n"
            "prior delegations. Include in the task message everything the worker\n"
            "needs: relevant paths, key findings from prior delegations, and the\n"
            "precise question to answer. You do NOT need to restate the\n"
            "hypotheses you name in hypothesis_ids — their statement, registered\n"
            "falsification_criterion and prediction are attached to the worker's\n"
            "task automatically, verbatim from the ledger. Spend the space on\n"
            "what the ledger does not already hold.\n\n"
            "hypothesis_ids must be non-empty when the ledger is active.\n"
            "The worker writes exclusively to {id}/ (relative to their workspace\n"
            "in debug/delegations/).\n\n"
            "Set is_falsification_attempt=True when this delegation attacks"
            " a hypothesis's stated falsification criterion.\n\n"
            "phase (optional): the f3dasm process stage this delegation advances —"
            " one of literature, doe, data_generation, ml, optimization, setup."
            " Tags the work's intent in the larger data-driven process; used by"
            " milestone gates, timing, and the critic.\n\n"
            "namespace (optional): open a NEW design parametrization as its own"
            " oracle + ledger. Leave it UNSET (the default) for the baseline study —"
            " that is most problems. Set namespace='some_name' only when the"
            " scientific question is a fundamentally different design REPRESENTATION"
            " (new variables / new geometry — e.g. 'elliptical_rings'): delegate a"
            " datagenerator with that namespace to build + register its oracle, then"
            " delegate implementers with the SAME namespace to evaluate in it. Each"
            " namespace keeps its own isolated ledger and the baseline is untouched;"
            " results compare across namespaces only insofar as they share the"
            " objective evaluator. A tool for creativity, not a requirement — open as"
            " many (or as few) as the science needs.\n\n"
            f"Available targets:\n  {_target_hints}"
        )

    # ── Hypothesis plumbing ──────────────────────────────────────────────────

    def _hypothesis_brief(
        self,
        hypothesis_ids: list | None,
        is_falsification_attempt: bool,
    ) -> str:
        """The registered hypotheses this delegation is meant to test, for
        the WORKER's task message.

        A hypothesis's falsification_criterion is immutable once registered
        and is the standard its verdict will be judged against — but until
        now the only party shown it was the delegator, and only at
        reconciliation time (``_falsification_checkpoint`` below, which
        fires on a Done report). The worker that actually produces the
        evidence never saw it: Delegate's contract put context packaging on
        the delegator ("Include in the task message everything the worker
        needs"), so the criterion reached the worker only if the delegator
        remembered to paste it.

        The measured cost of that gap: INCONCLUSIVE is the largest verdict
        class (100 of 295 hypotheses over 52 cluster runs) and the most
        expensive (median lifetime 3.15h vs 1.49h FALSIFIED, 0.91h
        SUPPORTED), and its verdict comments say why in so many words —
        "the registered H3 falsification criterion required a 50-iter
        constrained BO in the high-Ixx region. This BO was never executed";
        "Test is INADEQUATE relative to the registered 30-point LHS
        criterion". The work that ran was not the test that was registered,
        and nothing could notice until it was time to render a verdict.

        Injected in-band and automatically, on the same principle as the
        constraint snapshot: pre-registration the experimenter cannot read
        is not pre-registration.
        """
        node = self.node
        if node._ledger is None or not hypothesis_ids:
            return ""
        blocks = []
        for hid in hypothesis_ids:
            entry = node._ledger.get(str(hid)) or {}
            if not entry:
                continue
            stmt = (entry.get("statement") or "").strip()
            crit = (entry.get("falsification_criterion") or "").strip()
            pred = (entry.get("prediction") or "").strip()
            if not (stmt or crit or pred):
                continue
            # The statement is context and is capped; the criterion and the
            # prediction are the CONTRACT and go verbatim — truncating the
            # test a result will be judged against would reintroduce exactly
            # the mismatch this block exists to prevent.
            if len(stmt) > 700:
                stmt = stmt[:700].rstrip() + " […]"
            part = [f"**{hid}** — {stmt}" if stmt else f"**{hid}**"]
            if crit:
                part.append(f"  · REGISTERED FALSIFICATION CRITERION: {crit}")
            if pred:
                part.append(f"  · REGISTERED PREDICTION: {pred}")
            blocks.append("\n".join(part))
        if not blocks:
            return ""
        head = (
            "<registered_hypothesis>\n"
            "This delegation is a FALSIFICATION ATTEMPT on the hypotheses "
            "below. Their criteria were registered BEFORE this work and are "
            "immutable: your evidence will be judged against them exactly as "
            "written, so the test you run must be the test they specify "
            "(sampling plan, eval count, region, thresholds). If you cannot "
            "run that test, or find it cannot decide the criterion, say so in "
            "your report — an honest mismatch is usable; a different test "
            "reported as if it were this one is not."
            if is_falsification_attempt else
            "<registered_hypothesis>\n"
            "Context — the registered hypotheses this task is filed against. "
            "Their criteria are immutable and are what any verdict will be "
            "judged against; treat them as the standard your numbers have to "
            "speak to."
        )
        return head + "\n\n" + "\n\n".join(blocks) + "\n</registered_hypothesis>"

    def _falsification_checkpoint(self, delegation_id: str) -> str:
        """Read-time ritual text for a freshly-read Done report.

        Forces the strategizer to classify whether this delegation was a
        falsification ATTEMPT of a registered hypothesis and, if so, link it
        and record the verdict against the hypothesis's pre-registered
        (immutable) prediction. Returns "" when there is nothing to reconcile.
        Fires once per delegation (sets reconciled=True) to avoid nagging.
        """
        node = self.node
        if node._ledger is None:
            return ""
        try:
            hyps = node._ledger.list_all()
        except Exception:  # noqa: BLE001
            return ""
        if not hyps:
            return ""

        def _pred(hid: str) -> str:
            e = node._ledger.get(hid) or {}
            return (
                e.get("prediction")
                or e.get("falsification_criterion")
                or "(no prediction on record)"
            )

        with node._registry_lock:
            entry = node._registry.get(delegation_id)
            if (
                entry is None
                or entry.get("status") != "Done"
                or entry.get("reconciled")
            ):
                return ""
            is_fals = bool(entry.get("is_falsification_attempt"))
            linked = list(entry.get("hypothesis_ids") or [])
            entry["reconciled"] = True  # fire once

        if is_fals and linked:
            tested = "; ".join(
                f"{h} (prediction: \"{_pred(h)}\")" for h in linked)
            return (
                f"⚖ FALSIFICATION CHECKPOINT — {delegation_id} was declared "
                f"a falsification attempt of {tested}. Record the VERDICT now "
                "via HypothesisUpdate(<id>, "
                "status=SUPPORTED|FALSIFIED|INCONCLUSIVE, "
                f"evidence={{'delegation': '{delegation_id}', "
                "'numbers': {…}}), judging THIS report against that "
                "pre-registered prediction. Do not move on with the "
                "hypothesis left OPEN."
            )
        open_h = [h for h in hyps if h.get("current_status") == "OPEN"]
        if not open_h:
            return ""  # nothing open to test → don't nag exploration
        listing = "; ".join(
            f"{h['id']} (prediction: \"{_pred(h['id'])}\")" for h in open_h)
        return (
            f"⚖ FALSIFICATION CHECKPOINT — was {delegation_id} an attempt to "
            "test a registered hypothesis's pre-registered prediction? Open: "
            f"{listing}. If YES: LinkFalsificationAttempt('{delegation_id}', "
            "'<id>') then record the verdict with HypothesisUpdate against "
            f"that prediction. Link ONLY if {delegation_id} genuinely tested "
            "that prediction — do not retrofit an exploratory result onto a "
            "hypothesis. If it was exploration/setup, there is no hypothesis "
            "to attach — continue (no action needed)."
        )

    # ── Delegate, and the dispatch steps it runs in order ────────────────────

    @tool_examples(
        "Delegate('implementer', 'Run a 50-pt Latin sweep of t/L in "
        "[0.02,0.20]; evaluate via get_evaluator(); report top-5 by "
        "buckling_load_norm + the results CSV path + feasible count.', "
        "'top-5 t/L, their values, CSV path, n feasible', "
        "hypothesis_ids=['H1','H2'])",
        "Delegate('implementer', 'Falsification probe: dense grid n=20 of "
        "t/L in [0.10,0.14]; does any point beat buckling_load_norm 1.47?', "
        "'best value in range + pass/fail', hypothesis_ids=['H1'], "
        "is_falsification_attempt=True, phase='optimization')",
    )
    def Delegate(
        self,
        target: str,
        intent: str,
        expected_report: str,
        hypothesis_ids: list | None = None,
        wait: bool = False,
        is_falsification_attempt: bool = False,
        phase: str | None = None,
        namespace: str | None = None,
    ) -> str:
        """Fire a task to a connected agent.

        Replaced per node by :meth:`delegate_doc`, which appends this node's
        own connected targets. This text is the fallback when a node is built
        without a graph spec (tests).
        """
        node = self.node
        resolved = self._resolve_target(target)
        if resolved is None:
            return (
                f"ERROR: unknown target {target!r}."
                f" Valid targets: {node._outgoing}"
            )
        target = resolved
        worker_template = node._worker_adapters.get(target)
        if worker_template is None:
            return (
                f"ERROR: no worker adapter for target {target!r}."
                f" Available: {list(node._worker_adapters)}"
            )

        if isinstance(wait, str):  # MCP string-in tools may pass "false"
            wait = wait.strip().lower() not in ("false", "0", "no", "")

        h_ids = _parse_hypothesis_ids(hypothesis_ids)
        refusal = self._check_hypothesis_links(h_ids)
        if refusal is not None:
            return refusal

        # Resolve the optional process-phase tag (DoE/DataGeneration/ML/…).
        # Unknown/None → None (soft; never refuses), stored as the canonical
        # value string for the log + critic flags + downstream grouping.
        from ....runtime.phases import resolve_phase
        _phase_obj = resolve_phase(phase)
        _phase = _phase_obj.value if _phase_obj is not None else None

        nudge = self._milestone_gate(target, namespace)
        if nudge is not None:
            return nudge

        delegation_id = self._allocate_delegation_id()
        started_at = datetime.now(tz=timezone.utc).isoformat(timespec="seconds")
        self._register_dispatch(
            delegation_id, target, h_ids, is_falsification_attempt,
            _phase, namespace, started_at,
        )

        # Constraint snapshot NOW, at dispatch — single source of truth (see
        # constraint_snapshot.py) shared by the logged RUNNING entry below and
        # the banner prepended to the worker's own task message, so what gets
        # persisted and what the agent is shown can never drift apart the way
        # four independent, partial computations of this previously did.
        from ....runtime.constraint_snapshot import snapshot_for_node
        _snapshot = snapshot_for_node(node)

        # Provenance: log a RUNNING entry NOW, at dispatch — before the worker
        # runs and flushes ledger rows. If this delegation is cancelled or
        # killed mid-flight (wall/eval budget), its ledgered evals stay traceable
        # to a logged delegation instead of becoming orphan rows. The terminal
        # DONE/FAILED record (same id) supersedes this via last-wins collapse.
        if node._delegation_log is not None:
            node._delegation_log.record_started(
                id=delegation_id,
                from_node=node._name,
                to_node=target,
                task=intent,
                hypothesis_ids=h_ids,
                started_at=started_at,
                is_falsification_attempt=bool(is_falsification_attempt),
                phase=_phase,
                constraints=_snapshot.as_dict(),
            )

        task_msg = self._compose_task_message(
            delegation_id, target, intent, expected_report,
            hypothesis_ids, is_falsification_attempt, _snapshot,
        )

        # Each delegation gets its OWN adapter copy (D1/D2 concurrency fix).
        worker = (
            worker_template.copy() if hasattr(worker_template, "copy")
            else worker_template
        )
        self._sandbox_worker_writes(worker, delegation_id, target)

        session = WorkerSession(
            node,
            worker=worker,
            delegation_id=delegation_id,
            target=target,
            intent=intent,
            task_msg=task_msg,
            hypothesis_ids=h_ids,
            is_falsification_attempt=is_falsification_attempt,
            phase=_phase,
            namespace=namespace,
            started_at=started_at,
        )
        session.install_worker_tools()

        t = threading.Thread(
            target=session.run, daemon=True, name=delegation_id)
        with node._registry_lock:
            node._threads[delegation_id] = t  # A3: inside lock
        t.start()

        # Reset two-shot Done() gate so the next Done() warns again.
        node._done_warned = False

        if wait:
            # Synchronous mode: block until the delegation finishes.
            t.join()
            with node._registry_lock:
                entry = dict(node._registry.get(delegation_id, {}))
            status = entry.get("status", "Errored")
            if status == "Done":
                cp = self._falsification_checkpoint(delegation_id)
                body = f"Done\n\n{entry['result']}"
                return body + (("\n\n" + cp) if cp else "")
            return f"Errored:\n{entry.get('result', '(no details)')}"

        return (
            f"Delegation started. ID: {delegation_id!r}. "
            f"Use GetStatus('{delegation_id}') to poll for completion."
        )

    def _resolve_target(self, target: str) -> str | None:
        """Resolve a requested target to an outgoing node name, or None."""
        node = self.node
        outgoing = node._outgoing
        resolved = resolve_target(
            target, outgoing,
            {t: getattr(node._spec.nodes.get(t), "role", "") for t in outgoing},
        )
        if resolved is not None and resolved != target:
            # Forward-compatible alias resolution: the agent named the target by
            # capability (e.g. 'pipeline_executor' -> 'implementer'). Proceed and
            # record it for observability instead of bouncing the agent.
            node._record_intervention(
                "TARGET_ALIAS", resolved,
                f"delegation target {target!r} resolved to {resolved!r}.",
            )
        return resolved

    def _check_hypothesis_links(self, h_ids: list[str]) -> str | None:
        """Enforce hypothesis linkage when the ledger is active; else None."""
        node = self.node
        if node._ledger is None:
            return None
        if not h_ids:
            # Defer this requirement while the process backlog is still open
            # (setup phase): you propose hypotheses AFTER engaging with the
            # problem and setting up the oracle, so early setup delegations
            # (oracle wrapping, literature review) have nothing to link to
            # yet. Once the backlog is cleared, every delegation must cite a
            # hypothesis. (Tied to the existing milestone backlog, not a
            # phase taxonomy the agent controls.)
            _ms = getattr(node, "_milestones", None)
            _backlog_open = _ms is not None and bool(_ms.pending())
            if not _backlog_open:
                return (
                    "ERROR: hypothesis_ids must not be empty. "
                    "Every delegation must be linked to at "
                    "least one hypothesis. Call "
                    "HypothesisList() to see open hypotheses."
                )
            # backlog still open → permit this setup-phase delegation
            # without a hypothesis link.
        known = {h["id"] for h in node._ledger.list_all()}
        unknown = [h for h in h_ids if h not in known]
        if unknown:
            return (
                f"ERROR: unknown hypothesis IDs {unknown}."
                f" Valid IDs: "
                f"{sorted(known) or '(none proposed yet)'}."
            )
        return None

    def _milestone_gate(self, target: str, namespace: str | None) -> str | None:
        """The process backlog nudge, or None to proceed.

        Applies ONLY to delegations to the f3dasm implementer (the agent that
        runs experiments) — never the literature_reviewer/datagenerator that
        satisfy a milestone. Keyed on the resolved TARGET ROLE (reliable), not
        the agent's self-declared phase. This is a NUDGE, not a hard block: the
        milestones (assess-literature, oracle-ready, …) are good prompts, not a
        safety invariant, and a new design legitimately needs its own setup.
        Two-shot confirm, RECURRING PER NAMESPACE — nudge once per namespace,
        proceed on a re-delegate. MilestoneComplete/MilestoneSkip remain the
        clean path.
        """
        node = self.node
        _ms = getattr(node, "_milestones", None)
        _target_role = getattr(
            node._spec.nodes.get(target), "role", "") if node._spec else ""
        if _ms is None or _target_role != "implementer":
            return None
        from ....epistemics.milestones import implementer_block
        _pend = implementer_block(_ms, node)
        if not _pend:
            return None
        _ns_key = namespace or "__default__"
        if not hasattr(node, "_milestone_ack"):
            node._milestone_ack = set()
        if _ns_key in node._milestone_ack:
            # confirmed (and re-nudges for each new namespace) → proceed
            return None
        node._milestone_ack.add(_ns_key)
        _ids = ", ".join(f"{m['id']} ({m['description'][:50]}…)"
                         for m in _pend)
        node._record_intervention(
            "MILESTONE_BLOCK", target,
            f"{len(_pend)} backlog item(s) precede the implementer")
        _scope = f"design '{namespace}'" if namespace else "this study"
        return (
            f"[CONFIRM] process backlog still open for {_scope}: "
            f"{_ids}. The usual path is to resolve each first — "
            "MilestoneComplete(id, brief), or MilestoneSkip(id, "
            "reason) if it doesn't apply. If you mean to run the "
            "implementer anyway, re-delegate (same target) to "
            "confirm. (Not a tool error; a process nudge.)"
        )

    def _allocate_delegation_id(self) -> str:
        """Allocate this delegation's id.

        Called AFTER the milestone gate so a blocked attempt does not BURN an
        id (next_id() advances a monotonic counter on every call; allocating
        before the gate left a permanent gap in the sequence, e.g. the
        milestone-blocked first implementer attempt always ate D002). Globally
        unique when a shared DelegationLog is present (multiple orchestrating
        nodes share one log); per-node counter otherwise.
        """
        node = self.node
        if node._delegation_log is None:
            with node._registry_lock:
                node._delegation_seq += 1
                return f"D{node._delegation_seq:03d}"
        delegation_id = node._delegation_log.next_id()
        # Keep per-node seq in sync so checkpoint paths that read
        # _delegation_seq stay consistent.
        with node._registry_lock:
            try:
                node._delegation_seq = int(delegation_id[1:])
            except (ValueError, IndexError):
                pass
        return delegation_id

    def _register_dispatch(
        self,
        delegation_id: str,
        target: str,
        h_ids: list[str],
        is_falsification_attempt: bool,
        phase: str | None,
        namespace: str | None,
        started_at: str,
    ) -> None:
        """Open this delegation's registry entry."""
        node = self.node
        with node._registry_lock:
            node._registry[delegation_id] = {
                "status": "Working",
                "result": None,
                "evals": 0,
                "start_time": time.monotonic(),
                "hypothesis_ids": h_ids,
                "is_falsification_attempt": bool(is_falsification_attempt),
                "phase": phase,
                "started_at": started_at,
                "target": target,
                "namespace": (namespace or None),
                "followup_question": None,
                "followup_answer": None,
                "followup_event": threading.Event(),
                "getstatus_count": 0,
                "followup_count": 0,
                # Read-time falsification ritual: flips True once the
                # checkpoint has been shown for this delegation's report
                # (fire-once anti-nag). The Done()-gate dangling check is
                # content-based and independent of this flag.
                "reconciled": False,
            }

    def _compose_task_message(
        self,
        delegation_id: str,
        target: str,
        intent: str,
        expected_report: str,
        hypothesis_ids: list | None,
        is_falsification_attempt: bool,
        snapshot: Any,
    ) -> str:
        """Build the worker's task message: constraints, edge preamble, brief."""
        node = self.node
        edge = node._spec.edge(node._name, target)
        preamble = edge.preamble if edge else ""
        task_msg = (
            f"<workspace_subfolder>{delegation_id}/</workspace_subfolder>\n\n"
            + intent
        )
        if expected_report:
            task_msg += (
                f"\n\n**Required deliverables / acceptance"
                f" criteria:**\n{expected_report}"
            )
        # The registered hypothesis this work is filed against, from the
        # ledger a3dasm already holds — so the worker that produces the
        # evidence can see the criterion its evidence will be judged by,
        # instead of that criterion first surfacing at reconciliation time
        # when the work is already done. See _hypothesis_brief.
        _hyp_brief = self._hypothesis_brief(
            hypothesis_ids, is_falsification_attempt)
        if _hyp_brief:
            task_msg += "\n\n" + _hyp_brief

        if preamble:
            task_msg = preamble + "\n\n" + task_msg

        # Prepend the constraint snapshot so every worker starts budget-aware
        # (eval AND wall-clock, not wall-clock only) — automatically, in-band;
        # not something it has to go query for. GetStatus handles mid-run
        # updates.
        return snapshot.as_text() + "\n\n" + task_msg

    def _sandbox_worker_writes(
        self, worker: Any, delegation_id: str, target: str
    ) -> None:
        """Confine this worker's Write to its own ``{delegation_id}/``."""
        node = self.node
        if node._study_dir is None:
            return
        _workspace = (
            node._workspace_dir.resolve()
            if node._workspace_dir is not None
            else (Path(node._study_dir or ".") / "debug" / "delegations").resolve()
        )
        _delegation_ws = (_workspace / delegation_id).resolve()

        def Write(path: str, body: str, _ws=_delegation_ws, _did=delegation_id) -> str:
            """Write a file. Restricted to your own {delegation_id}/ folder."""
            # Absorb a redundant leading "{delegation_id}/": the sandbox is
            # ALREADY rooted at {delegation_id}/, but the prompt calls it
            # "your D### subfolder", so agents naturally prefix paths with it
            # — which would nest D###/D###/. Strip one leading D### component
            # (only when a real filename remains after it). Do NOT lstrip
            # "/": an absolute path must stay absolute so the relative_to
            # boundary check below still rejects it.
            _norm = (path or "").strip()
            _first, _sep, _rest = _norm.partition("/")
            if _first == _did and _rest:
                _norm = _rest
            try:
                candidate = (_ws / _norm).resolve()
            except Exception as exc:  # noqa: BLE001
                return f"ERROR: invalid path {path!r}: {exc}"
            try:
                candidate.relative_to(_ws)
            except ValueError:
                return (
                    f"ERROR: write rejected — {candidate} is outside "
                    f"{_ws}. Write only to {_did}/."
                )
            candidate.parent.mkdir(parents=True, exist_ok=True)
            candidate.write_text(body, encoding="utf-8")
            return f"Written: {candidate}"

        worker.closure_tools["Write"] = node._wrap_closure(Write, target)

    # ── Polling, waiting, cancelling ─────────────────────────────────────────

    def GetStatus(self, delegation_id: str) -> str:
        """Poll a background delegation; also delivers push notifications.

        Returns one of:
          'Working (running for Xs, polled N times)' — still running
          'Done\\n\\n<full report>'                  — completed
          'Errored:\\n<traceback>'                   — failed

        Polling is for async (wait=False) delegations. If this task had no
        reason to overlap other work, Delegate(wait=True) would have returned
        the result directly with no polling — worth a thought next time.
        """
        node = self.node
        prefix = node._drain_notifications()

        # Drain any budget warnings queued for this delegation.
        with node._pending_worker_msgs_lock:
            worker_msgs = node._pending_worker_msgs.pop(delegation_id, [])
        if worker_msgs:
            prefix += wrap_notice("\n".join(worker_msgs))

        with node._registry_lock:
            entry = node._registry.get(delegation_id)
            if entry is None:
                return self._status_from_log(delegation_id, prefix)
            status = entry["status"]
            if status in ("Working", "FollowUp"):
                # Increment poll count and record timing.
                entry["getstatus_count"] = entry.get("getstatus_count", 0) + 1
                poll = {
                    "count": entry["getstatus_count"],
                    "last": entry.get("last_getstatus_time"),
                    "start": entry["start_time"],
                    "prev_stamped": entry.get("last_stamped", 0),
                    "last_progress": entry.get(
                        "last_progress_time", entry["start_time"]),
                    "note": entry.get("progress_note"),
                }
                entry["last_getstatus_time"] = time.monotonic()

        # Status token FIRST (contract: callers dispatch on the
        # leading word); queued notifications follow the report.
        _tail = ("\n\n" + prefix.rstrip()) if prefix.strip() else ""
        if status == "Done":
            cp = self._falsification_checkpoint(delegation_id)
            body = f"Done\n\n{entry['result']}"
            if cp:
                body += "\n\n" + cp
            return body + _tail
        if status not in ("Working", "FollowUp"):
            return f"Errored:\n{entry['result']}" + _tail
        return self._working_report(delegation_id, poll) + _tail

    def _status_from_log(self, delegation_id: str, prefix: str) -> str:
        """Registry cache miss: consult the authoritative delegation log.

        The in-memory registry can be rebuilt empty after a node
        reconstruction while the log retains every delegation — this is the
        "Known IDs: []" symptom (audit BF-0).
        """
        node = self.node
        _lstatus, _ldeliv = node._log_status(delegation_id)
        if _lstatus == "DONE":
            return prefix + f"Done\n\n{_ldeliv}"
        if _lstatus == "FAILED":
            return prefix + f"Errored:\n{_ldeliv}"
        if _lstatus == "RUNNING":
            return prefix + (
                "Working (still running; live progress is unavailable "
                "after a session rebuild — re-poll shortly and the "
                "result will appear here when it completes)"
            )
        return (
            prefix +
            f"ERROR: unknown delegation ID {delegation_id!r}. "
            f"Known IDs: {list(node._registry)}"
        )

    def _working_report(self, delegation_id: str, poll: dict) -> str:
        """The 'still Working' status line: real progress, then any nudges."""
        now_mono = time.monotonic()
        elapsed = int(now_mono - poll["start"])
        poll_count = poll["count"]

        progress_desc, cur_stamped = self._progress_description(
            delegation_id, poll, now_mono, elapsed)

        note_desc = ""
        if poll["note"]:
            _ntext, _nts = poll["note"]
            note_desc = (
                f" · worker note: {_ntext!r} ({int(now_mono - _nts)}s ago)")

        hints: list[str] = []
        last_poll = poll["last"]
        # Rate warning: polled too recently.
        if last_poll is not None and (now_mono - last_poll) < 30:
            hints.append(
                f"NOTE: you polled {delegation_id} only "
                f"{now_mono - last_poll:.0f}s ago. "
                "The worker runs in a background thread — polling faster "
                "does not make it finish sooner. Do other work in the "
                "meantime."
            )
        hints.extend(self._poll_escalation(
            delegation_id, poll_count, elapsed, cur_stamped, progress_desc))
        hints.extend(self._budget_broadcast(delegation_id))

        # Status token FIRST (documented contract: callers may
        # dispatch on the leading word); hints and queued
        # notifications follow.
        hint_str = (
            ("\n\n" + wrap_notice("\n".join(hints), trailing=""))
            if hints else ""
        )
        return (
            f"Working (running for {elapsed}s, polled {poll_count}× · "
            + progress_desc + note_desc + ")" + hint_str
        )

    def _progress_description(
        self, delegation_id: str, poll: dict, now_mono: float, elapsed: int
    ) -> tuple[str, int]:
        """Real ledger progress for a running delegation, plus its RSS.

        Surfaces real progress so the delegator can tell "progressing" from
        "stuck" instead of inferring it from wall-time (the blindness that
        drove over-cancelling). Also folds in backlog #6: zero stamped after a
        long wall-time IS the stuck signal. Counts the delegation's rows across
        EVERY experiment store (provenance-based) — else a campaign that wrote
        to its experiment's store polls as 0 progress and the stuck signal
        would over-cancel a healthy worker.
        """
        node = self.node
        _run_exp = (
            node._current_notes_dir.parent.parent / "experiment_data"
            if node._current_notes_dir is not None else None
        )
        cur_stamped = (
            _stamped_eval_count(_run_exp, delegation_id) if _run_exp else 0
        )
        prev_stamped = poll["prev_stamped"]
        delta = cur_stamped - prev_stamped
        last_progress = poll["last_progress"]
        if cur_stamped > prev_stamped:
            last_progress = now_mono
        with node._registry_lock:
            _e = node._registry.get(delegation_id)
            if _e is not None:
                _e["last_stamped"] = cur_stamped
                _e["last_progress_time"] = last_progress
        stale = int(now_mono - last_progress)
        if cur_stamped > 0 and delta > 0:
            progress_desc = (
                f"{cur_stamped} evals stamped (+{delta} since last poll) "
                "— progressing")
        elif cur_stamped > 0:
            progress_desc = (
                f"{cur_stamped} evals stamped, none new for {stale}s")
        else:
            progress_desc = f"0 evals stamped after {elapsed}s"
        # Per-delegation memory telemetry (resource-governance L3): surface this
        # delegation's process-tree RSS so the strategizer can SEE a fat campaign
        # and Confer the implementer. Best-effort; appended only if known.
        if _run_exp is not None:
            try:
                from ....infra.watchdog_cleanup import (
                    delegation_peak_rss,
                    delegation_rss,
                )
                _rss = delegation_rss(_run_exp.parent, delegation_id)
                if _rss > 0:
                    progress_desc += f"; ~{_rss / 1024 ** 2:.0f} MB RSS"
                    _peak = delegation_peak_rss(delegation_id)
                    if _peak > _rss:
                        progress_desc += f" (peak ~{_peak / 1024 ** 2:.0f} MB)"
            except Exception:  # noqa: BLE001
                pass
        return progress_desc, cur_stamped

    def _poll_escalation(
        self,
        delegation_id: str,
        poll_count: int,
        elapsed: int,
        cur_stamped: int,
        progress_desc: str,
    ) -> list[str]:
        """Nudges for an agent polling in a tight loop.

        Polling does NOT make the worker finish sooner, so from the first
        escalation we spell out the three real ways forward (same options as
        the premature-Done nudge) — so the agent never grinds out 30 status
        checks when it could just wait.
        """
        if poll_count < 5:
            return []
        firmness = "STOP polling in a tight loop. " if poll_count >= 15 else ""
        if cur_stamped > 0:
            # Demonstrably progressing — anchor the nudge on the numbers so
            # the agent doesn't cancel a healthy campaign out of impatience.
            return [
                f"{firmness}Polled {poll_count}× ({elapsed}s) — but "
                f"{delegation_id} IS progressing ({progress_desc}). Polling "
                "won't speed it up. Best move: (a) do other work now; or "
                "(b) call Wait() with NO argument — it blocks until "
                "whichever delegation finishes first and hands you its "
                "report, so several in flight need no polling at all. "
                "Do NOT cancel a "
                "progressing delegation to save time — its ledgered evals "
                "persist regardless, so cancelling only discards its report."
            ]
        # Zero stamped (backlog #6 stuck signal): cancelling is now a
        # defensible call, but only here.
        return [
            f"{firmness}Polled {poll_count}× ({elapsed}s) and "
            f"{progress_desc}. Options: (a) do other work; (b) call "
            "Wait() with NO argument — it blocks until whichever "
            "delegation finishes first, with zero polling — a worker "
            "may still be setting up before its first "
            "eval. A delegation that has stamped NOTHING for a long "
            "time may be genuinely stuck; the run watchdog will reclaim "
            "it."
        ]

    def _budget_broadcast(self, delegation_id: str) -> list[str]:
        """Broadcast a newly-crossed 10%-overbudget threshold, once.

        Queued for every other Working delegation and returned for this one.
        """
        node = self.node
        budget = node._budget_seconds
        run_start = node._run_start
        if budget is None or run_start is None:
            return []
        pct = ((time.time() - run_start) / budget) * 100
        # Thresholds: 80, 90, 100, 110, 120, …
        threshold = int(pct // 10) * 10
        if threshold < 80:
            return []
        with node._pending_worker_msgs_lock:
            if threshold in node._budget_notified_pcts:
                return []
            node._budget_notified_pcts.add(threshold)
            _backstop_mult = run_backstop_multiple()
            _over = backstop_enabled() and pct >= _backstop_mult * 100
            msg = (
                f"BACKSTOP IMMINENT: {pct:.0f}% of time "
                "budget — past the "
                f"{int(_backstop_mult)}x cost "
                "backstop. Stop polling and call Done() "
                "NOW with a partial report."
            ) if _over else (
                f"BUDGET: {pct:.0f}% of time budget consumed. "
                "Wrap up your current work and return a partial "
                "report as soon as possible."
            )
            # Queue for all currently Working delegations.
            with node._registry_lock:
                active = [
                    did for did, e in node._registry.items()
                    if e["status"] in ("Working", "FollowUp")
                    and did != delegation_id
                ]
            for did in active:
                node._pending_worker_msgs.setdefault(did, []).append(msg)
        return [msg]

    def CancelDelegation(self, delegation_id: str) -> str:
        """Detach a delegation whose RESULT you no longer want.

        Cancel ONLY when the output is genuinely unwanted — a wrong approach, a
        superseded plan, a true dead-end. Do NOT cancel a delegation because it
        is slow: a Working delegation is almost always still producing real,
        ledgered evaluations (the worker runs in a background thread — slow is
        not stuck). Cancelling discards its REPORT, so its findings never reach
        your conclusion; its already-written ledger rows remain (and still
        count). If you just want to make progress meanwhile, do other work in
        parallel and let it finish. A delegation that has already produced
        ledgered evals is two-shot: call twice to confirm."""
        node = self.node
        prefix = node._drain_notifications()
        with node._registry_lock:
            entry = node._registry.get(delegation_id)
            if entry is None:
                return (
                    prefix + f"No delegation {delegation_id!r}. "
                    f"Known: {list(node._registry)}"
                )
            st = entry.get("status")
            if st not in ("Working", "FollowUp"):
                return (
                    prefix + f"Delegation {delegation_id} is {st!r}, not "
                    "running — nothing to cancel."
                )
            # Harden against impatience: a delegation already writing ledgered
            # evals is progressing, not stuck. Require a deliberate second call
            # so a slow-but-healthy campaign can't be discarded on a whim. Count
            # across EVERY experiment store (provenance-based) — else a campaign
            # that wrote to its experiment's store reads 0 and loses this guard.
            _run_exp = (
                node._current_notes_dir.parent.parent / "experiment_data"
                if node._current_notes_dir is not None else None
            )
            _stamped = (
                _stamped_eval_count(_run_exp, delegation_id) if _run_exp else 0
            )
            if _stamped > 0 and not entry.get("cancel_pending"):
                entry["cancel_pending"] = True
                return (
                    prefix + f"HOLD: {delegation_id} has already written "
                    f"{_stamped} provenance-stamped evaluation(s) to the "
                    "canonical ledger — it is progressing, not stuck. "
                    "Cancelling discards its REPORT (its findings won't reach "
                    "your conclusion); the evals remain. If it is merely slow, "
                    "do other work in parallel and let it finish. If its result "
                    "is genuinely unwanted, call CancelDelegation('"
                    + delegation_id + "') again to confirm."
                )
            entry["status"] = "Cancelled"
        with node._notifications_lock:
            node._notifications.append(
                f"[Delegation {delegation_id} cancelled — detached; its "
                "report is discarded (its ledgered evals remain)]"
            )
        return (
            prefix + f"Delegation {delegation_id} cancelled (detached): "
            "excluded from the run, its result will be ignored. You may "
            "proceed (e.g. call Done() if nothing else is running) or start "
            "other work."
        )

    def Wait(self, delegation_id: str | None = None) -> str:
        """Block until a delegation finishes (Done or Errored), then return its
        result. Use instead of polling with GetStatus() — holds the current
        turn open with no extra turns consumed.

        OMIT delegation_id to wait for whichever delegation finishes FIRST.
        That is how you collect a fan-out: dispatch several with
        Delegate(wait=False), then call Wait() once per worker — each call
        hands back one finished delegation's report (labelled with its ID) and
        blocks only while nothing is ready. Naming an ID instead waits for that
        specific worker, which leaves any others finishing unread, so prefer
        the bare form whenever more than one delegation is in flight.

        Refuses when there is nothing to wait for, and refuses rather than
        hanging when waiting cannot make progress — every in-flight delegation
        parked on a FollowUp (answer it with Reply), or already gone without
        reporting (read it with GetStatus).

        Returns the same text as GetStatus() once the delegation completes."""
        prefix = self.node._drain_notifications()
        if delegation_id is None:
            return self._wait_for_any(prefix)
        return self._wait_for_one(delegation_id, prefix)

    def _wait_for_any(self, prefix: str) -> str:
        """Wait for whichever delegation finishes first.

        There is no join() across threads, so poll the registry: a short tick
        for responsiveness, draining notifications and monitor drift on the
        same ~10s cadence :meth:`_wait_for_one` uses.
        """
        node = self.node
        _tick, _n = 1.0, 0
        while True:
            with node._registry_lock:
                ready = [
                    (i, e) for i, e in node._registry.items()
                    # Cancelled is terminal but its result is explicitly
                    # excluded from the run, so it is never harvestable.
                    if e.get("status") in ("Done", "Errored")
                    and not e.get("waited")
                ]
                if ready:
                    did, entry = ready[0]
                    entry["waited"] = True
                    cp = entry.get("checkpoint", "")
                    body = (f"[{did}] {entry['status']}\n\n"
                            f"{entry.get('result', '')}")
                    return prefix + body + (("\n\n" + cp) if cp else "")
                open_ids = sorted(
                    i for i, e in node._registry.items()
                    if e.get("status") in ("Working", "FollowUp")
                )
                # Classify what is actually still capable of finishing.
                # A blocking tool call ends no turn, so the run's time
                # backstop cannot fire while we are in here (same trap
                # ReadNote guards against) — this loop must therefore
                # never be able to wait on something that will never
                # arrive. A thread that has died without recording a
                # terminal status is exactly that: nothing else in the
                # runtime marks the registry on its behalf.
                waitable, blocked, dead = [], [], []
                for i in open_ids:
                    t = node._threads.get(i)
                    if t is not None and not t.is_alive():
                        dead.append(i)
                    elif node._registry[i].get("status") == "FollowUp":
                        blocked.append(i)
                    else:
                        # No registered thread means we cannot prove it is
                        # gone; assume it is still coming.
                        waitable.append(i)
            if not open_ids:
                return prefix + (
                    "ERROR: nothing to wait for — no delegation is in "
                    "flight and every finished one has already been read. "
                    "Delegate(...) work first."
                )
            if not waitable:
                bits = []
                if blocked:
                    bits.append(
                        "parked on a FollowUp question "
                        f"({', '.join(blocked)}) — call GetStatus(id) to "
                        "read it, then Reply(id, answer) to unblock it"
                    )
                if dead:
                    bits.append(
                        f"no longer running but never reported "
                        f"({', '.join(dead)}) — call GetStatus(id) for its "
                        "state"
                    )
                return prefix + (
                    "ERROR: waiting cannot make progress; every in-flight "
                    "delegation is " + "; and ".join(bits) + "."
                )
            time.sleep(_tick)
            _n += 1
            if _n % 10 == 0:
                prefix += self._drain_while_waiting()

    def _wait_for_one(self, delegation_id: str, prefix: str) -> str:
        """Wait for one named delegation."""
        node = self.node
        with node._registry_lock:
            entry = node._registry.get(delegation_id)
            if entry is None:
                return prefix + (
                    f"ERROR: unknown delegation {delegation_id!r}. "
                    f"Known: {list(node._registry)}"
                )
            if entry["status"] in ("Done", "Errored"):
                entry["waited"] = True
                cp = entry.get("checkpoint", "")
                body = f"{entry['status']}\n\n{entry.get('result', '')}"
                return prefix + body + (("\n\n" + cp) if cp else "")
            t = node._threads.get(delegation_id)

        if t is not None:
            while t.is_alive():
                t.join(timeout=10.0)
                prefix += self._drain_while_waiting()

        with node._registry_lock:
            entry = node._registry.get(delegation_id, {})
            # Read once: a bare Wait() must not hand this same report back.
            if entry:
                entry["waited"] = True
        cp = entry.get("checkpoint", "")
        body = f"{entry.get('status', 'Unknown')}\n\n{entry.get('result', '')}"
        return prefix + body + (("\n\n" + cp) if cp else "")

    def _drain_while_waiting(self) -> str:
        """Notifications and science drift, drained mid-Wait.

        Wakes a strategizer that's asleep in Wait() for a live campaign —
        without this, a science-monitor nudge (e.g. DUPLICATE_EVALUATION) only
        surfaces on the NEXT tool call, by which point the whole delegation
        (and its eval budget) has already finished. Same drain() every other
        call site uses; each poll tick is a fresh check, not a repeat.
        """
        node = self.node
        out = ""
        with node._notifications_lock:
            _notifs = list(node._notifications)
            node._notifications.clear()
        for _notif in _notifs:
            out += _notif + "\n\n"
        if node._science_monitor is not None:
            drift = node._science_monitor.drain()
            if drift:
                out += wrap_notice(drift)
        return out

    def Reply(self, delegation_id: str, answer: str) -> str:
        """Answer a worker's FollowUp question and unblock it.

        Call this after GetStatus returns 'FollowUp: <question>'.
        The answer is injected into the worker's context and it resumes.
        """
        node = self.node
        prefix = node._drain_notifications()
        with node._registry_lock:
            entry = node._registry.get(delegation_id)
            if entry is None:
                return prefix + f"ERROR: unknown delegation {delegation_id!r}."
            if entry.get("status") != "FollowUp":
                return (
                    prefix +
                    f"ERROR: delegation {delegation_id!r} is not awaiting a "
                    f"FollowUp (status: {entry.get('status')!r})."
                )
            entry["followup_answer"] = answer
            evt = entry["followup_event"]
        evt.set()
        return prefix + f"Reply sent to {delegation_id}. Worker resuming."

    # ── Asking the human ─────────────────────────────────────────────────────

    def FollowUp(self, question: str) -> str:
        """Ask your delegating party one clarifying question before proceeding.

        Routes to whoever sent you this task: the human operator if you are
        the entry node, or the agent that delegated to you if you are a worker.
        One FollowUp per delegation.  The answer is injected directly into
        your context.  If no answer is available, proceed with best judgment.
        Use it only for genuine briefing ambiguities (or a result so
        surprising it may signal a bug) — never for rhetorical/confirmatory
        questions or to replace your own reasoning.
        """
        node = self.node
        if node._ask_count >= node._max_ask:
            return (
                f"FollowUp limit reached ({node._max_ask} per run). "
                "Proceed autonomously with the information you have."
            )
        # _ask_count is incremented only once the question can actually be
        # put to someone (below). Counting it here spent the run's whole
        # quota on questions that were closed unattended microseconds later
        # and never displayed anywhere.
        # Two ways to reach a human, and the run takes whichever answers
        # first: a terminal (as before) and the viewer, which answers by
        # writing into the run's own debug/followups/ directory. The viewer
        # is a separate process and may not exist, so the channel is on
        # disk rather than in memory.
        from ....infra.operator_channel import (
            ask_question,
            close_question,
            is_watched,
        )

        _tty = bool(node._interactive) and _stdin_is_tty()
        _run_dir = node._current_run_dir
        if _run_dir is None:
            # Nowhere to publish the question, so nowhere an answer could
            # come from except a terminal.
            # No deadline is available on this path and there is no viewer
            # to answer, so it must not block: an unattended terminal would
            # otherwise hang the run forever, which is the exact failure the
            # isatty() guard was introduced to prevent.
            typed = self._typed_answer_now() if _tty else None
            return typed or _UNATTENDED

        qid = ask_question(_run_dir, node._name, question)
        if qid is None:
            # The channel could not be written — a full or read-only debug
            # dir. Degrade to the unattended answer rather than raising a
            # disk error out of a clarifying question.
            return _UNATTENDED

        # Waiting is only reasonable when somebody could actually answer.
        # With no terminal and nobody watching in the viewer, a wait is not
        # patience, it is a stall that spends the run's wall budget on a
        # question no one will ever see — so that case returns immediately,
        # exactly as it did before this channel existed.
        if not (_tty or is_watched(_run_dir)):
            close_question(_run_dir, qid, "unattended")
            return _UNATTENDED

        node._ask_count += 1
        if _tty:
            print(f"\n[Node {node._name}] {question}\nAnswer: ",
                  end="", flush=True)
        return self._await_operator_answer(_run_dir, qid, _tty)

    def _typed_answer_now(self) -> str | None:
        """A non-blocking read of anything already typed at the terminal."""
        import select as _select
        import sys as _sys
        try:
            if _select.select([_sys.stdin], [], [], 0)[0]:
                typed = _sys.stdin.readline()
                if typed.strip():
                    self.node._ask_count += 1
                    return typed.strip()
        except (OSError, ValueError):
            pass
        return None

    def _await_operator_answer(
        self, run_dir: Path, qid: str, tty: bool
    ) -> str:
        """Poll the terminal and the viewer together until one answers.

        Blocking on input() would make a terminal-attached run deaf to the
        viewer, which is the case the operator is most likely to be using.
        """
        import select as _select
        import sys as _sys

        from ....infra.operator_channel import (
            answer_question as _answer_q,
        )
        from ....infra.operator_channel import (
            close_question,
            is_watched,
        )
        from ....infra.operator_channel import (
            read_answer as _read_answer,
        )
        from ....runtime.settings import get_int as _get_int

        _deadline = time.monotonic() + _get_int("followup_wait_s", 600)
        while time.monotonic() < _deadline:
            typed_ready = False
            if tty:
                try:
                    typed_ready = bool(
                        _select.select([_sys.stdin], [], [], 0.5)[0])
                except (OSError, ValueError):
                    # A stdin that cannot be selected is not a stdin that can
                    # answer; stop treating it as one, and keep pacing from
                    # the sleep below rather than spinning.
                    tty = False
                    typed_ready = False
            if not tty:
                time.sleep(0.5)

            if typed_ready:
                try:
                    typed = _sys.stdin.readline()
                except (EOFError, KeyboardInterrupt, ValueError):
                    typed = ""
                if typed == "":
                    # EOF. select() reports a closed tty readable forever and
                    # readline() returns instantly, so continuing to poll it
                    # spins a core flat out for the whole deadline — measured
                    # at ~285k iterations/second. Nobody can type into a
                    # closed stdin, so stop watching it.
                    tty = False
                elif typed.strip():
                    # Only return a typed answer if it was actually recorded.
                    # Losing this race to the viewer means the record holds a
                    # DIFFERENT answer from the one handed to the agent.
                    if _answer_q(run_dir, qid, typed.strip()):
                        return typed.strip()
                    from_typed_race = _read_answer(run_dir, qid)
                    if from_typed_race:
                        return from_typed_race

            from_viewer = _read_answer(run_dir, qid)
            if from_viewer:
                if tty:
                    print(f"[answered in the viewer: {from_viewer}]", flush=True)
                return from_viewer

            # A watcher that goes away mid-wait ends the wait: the reason
            # for waiting was that someone was there.
            if not tty and not is_watched(run_dir):
                close_question(run_dir, qid, "unattended")
                return _UNATTENDED

        close_question(run_dir, qid, "timeout")
        return _UNATTENDED

    # ── Episodic memory ──────────────────────────────────────────────────────

    def RecallHistory(self, n: int = 5) -> str:
        """Return the last n delegations received by this node as (task, deliverable) pairs.
        Call at the start of a delegation to recall prior work. Returns oldest-first."""
        node = self.node
        if node._delegation_log is None:
            return "No delegation log available."
        try:
            n = int(n)  # the model may pass "5"; query_received does [-n:]
        except (TypeError, ValueError):
            n = 5
        # The graph's entry/orchestrating node is ALWAYS the from_node on
        # every delegation, never the to_node — query_received filters on
        # to_node, so this is structurally, permanently empty for it (not
        # a transient "nothing happened yet" state). Granting the tool
        # here at all is a topology accident (a node holds these tools
        # because it has outgoing edges, entry or not); a plain "No prior
        # delegations found" reads as amnesia rather than as "wrong tool
        # for this role" (misdiagnosed as a context/turn-boundary memory
        # bug in run 20260718T132852's DONE retrospective — 7 delegations
        # and 102 evals already existed at the time).
        _entry = getattr(node._spec, "entry", None)
        if _entry is not None and node._name == _entry:
            return (
                "RecallHistory recalls delegations RECEIVED by this "
                "node — the entry/orchestrating node dispatches "
                "delegations, it never receives one, so this is always "
                "empty here (not a memory gap). Use RecallStore/"
                "QueryStore for evaluation history, or HypothesisList/"
                "HypothesisGet for the hypothesis ledger, instead."
            )
        records = node._delegation_log.query_received(node._name, n)
        if not records:
            return "No prior delegations found."
        parts = []
        for i, r in enumerate(records, 1):
            parts.append(
                f"== Prior delegation {i} ==\n"
                f"Task: {r['task']}\n\n"
                f"Deliverable:\n{r['deliverable']}"
            )
        return "\n\n---\n\n".join(parts)

    # ── Peer messaging ───────────────────────────────────────────────────────

    def Confer(self, target: str, message: str) -> str:
        """Send an async message to another node in the run.

        Returns immediately — neither side blocks. Use it to correct or
        steer a delegation that is ALREADY RUNNING, rather than waiting
        for a wrong result and re-delegating.

        target is a node name, a delegation id (D004 — address a
        specific delegation when two of one role are running), or the
        orchestrating node. A running delegation gets the message
        prefixed onto its next tool result; an idle node's message waits
        until that node itself Confers. The reply tells you which
        happened — read it, because "delivered" and "queued" are
        different outcomes.

        Reply by convention with Confer(sender_name, "re #N: <answer>").
        """
        # Prepends the notification drain, which also delivers this node's
        # own inbox — the one thing that differs from a worker's Confer.
        prefix = self.node._drain_notifications()
        return prefix + ConferTools(self.node, self.node._name).Confer(
            target, message)


_UNATTENDED = (
    "No operator is present to answer. Proceed autonomously "
    "using only information available in the task message "
    "and files in the study directory."
)


def _stdin_is_tty() -> bool:
    """Whether stdin is a terminal somebody could type an answer into.

    ``.isatty()`` on a CLOSED stream raises ValueError rather than returning
    False — reachable under supervisors that close fd 0 instead of reopening
    it on /dev/null.
    """
    import sys as _sys
    try:
        return bool(getattr(_sys.stdin, "isatty", lambda: False)())
    except (ValueError, OSError):
        return False


def _parse_hypothesis_ids(hypothesis_ids: list | str | None) -> list[str]:
    """Decode the several shapes an LLM passes hypothesis ids in.

    ``'["H1","H2"]'`` (JSON), ``'H1, H2'`` (joined), ``'H1'``, or a real list.
    """
    if not isinstance(hypothesis_ids, str):
        return [str(h) for h in (hypothesis_ids or [])]
    raw = hypothesis_ids.strip()
    if raw.startswith("["):
        import json as _json
        try:
            decoded = _json.loads(raw)
        except _json.JSONDecodeError:
            return [raw]
        return [str(h) for h in decoded] if isinstance(decoded, list) else [raw]
    if "," in raw:
        return [p.strip() for p in raw.split(",") if p.strip()]
    return [raw]


def build_delegation_closures(node) -> dict:
    """The delegation-family tools for one node, by registered name.

    Every value is a bound method of :class:`DelegationTools` — ``self`` is
    dropped by ``inspect.signature``, so the schema the backends infer and the
    catalog the prompt renders are exactly those of the method's own
    parameters and docstring.
    """
    t = DelegationTools(node)
    return {
        "Delegate": with_doc(t.Delegate, t.delegate_doc()),
        "Wait": t.Wait,
        "Reply": t.Reply,
        "FollowUp": t.FollowUp,
        "RecallHistory": t.RecallHistory,
        "GetStatus": t.GetStatus,
        "CancelDelegation": t.CancelDelegation,
        "Confer": t.Confer,
    }
