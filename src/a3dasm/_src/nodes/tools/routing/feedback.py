"""Critic-facing / run-closing tool closures: Done, AskForFeedback. Built
per-node via build_feedback_closures(node) -- the node is passed in so
closures reach its state. Extracted verbatim from the former
build_routing_tools()."""
from __future__ import annotations

from datetime import datetime, timezone

from ...parsing import _parse_verdict

# Post-Done exit interview for the strategizer. Asked as a SEPARATE turn only
# after the critic accepted the conclusion — so the strategizer never carries
# the interview in its working context (no pollution). It answers with one more
# Done() whose summary is just a ### Retrospective block.
_EXIT_INTERVIEW = (
    "Your conclusion has been accepted by the critic and recorded — the run "
    "is effectively closed. One last thing before we finalise: a quick "
    "question about the SYSTEM you worked within (its rules, tools, and the "
    "monitor/critic feedback), NOT the science. Call Done() ONE more time "
    "with a summary containing only a ### Retrospective block:\n"
    "- CONSISTENCY: ok | flagged — did any rule, tool, monitor message, or "
    "critic finding contradict another, or contradict what you were told "
    "elsewhere (e.g. a rule that rejected evidence you believe was correct)? "
    "Write 'flagged' and QUOTE both sides; otherwise 'ok'. (Most important.)\n"
    "- DECISION: the one strategic choice you were least sure the system "
    "wanted, and why you made it.\n"
    "- FRICTION: anything counterintuitive about the rules/tools, INCLUDING what "
    "you recovered from (an errored tool call, a wrong-guessed tool name), not "
    "just blockers; 'none' only if truly zero.\n"
    "- BLOCKED: was there anything you NEEDED to do your job but COULDN'T — a "
    "missing tool, permission, or way to test/inspect your own work (e.g. no "
    "way to run or debug a deliverable you had to author)? Name it specifically, "
    "or 'none'. (We want CAPABILITY GAPS, not just counterintuitive rules — be "
    "honest; an unreported gap can't be fixed.)\n"
    "This will NOT reopen the run."
)


# Retrospective for runs that did NOT pass (UNGATED / FAILED). The most painful
# runs carry the most friction signal, yet they used to close with no interview
# at all — so capability gaps (e.g. "I couldn't run my own deliverable") were
# never surfaced. Capture them here, with the same BLOCKED probe.
_FAILED_RETROSPECTIVE = (
    "The run is closing WITHOUT a passing conclusion (it is recorded as "
    "UNGATED/FAILED — this is final and will NOT reopen). Before it finalises, "
    "a quick retrospective about the SYSTEM, not the science. Call Done() ONE "
    "more time with a summary containing only a ### Retrospective block:\n"
    "- BLOCKED: the single biggest thing you NEEDED but COULDN'T do — a missing "
    "tool, permission, or way to test/inspect your own work (e.g. no way to run "
    "or debug the deliverable you had to author). Name it specifically. (This "
    "is the most important field — be candid; an unreported gap can't be "
    "fixed.)\n"
    "- BLOCKER: in one line, the proximate reason the run did not pass.\n"
    "- FRICTION: any rule/tool that worked against you, INCLUDING what you "
    "recovered from (an errored call, a wrong-guessed tool name), not just "
    "blockers; 'none' only if truly zero.\n"
    "This will NOT reopen the run."
)



def build_feedback_closures(node) -> dict:
    route = node._route

    def Done(summary: str) -> str:
        """Signal end of run with a summary of findings (two-shot).

        Call only when: a best design is in hand with numerical support from
        Reports; at least one falsification attempt has been carried out; every
        PRIMARY success criterion is MET (an INCONCLUSIVE/unmet one is not — run
        the affordable experiment that would settle it if budget remains); and
        pipeline.ipynb has been authored via WriteDeliverable("pipeline.ipynb", …).
        summary should state the best design + supporting numbers + the
        falsification outcome + remaining uncertainty + (if closing with budget
        left) why the remaining budget cannot settle any unmet criterion.

        First call: issues a WARNING and lists any open delegations or
        unmet conditions; does NOT close.
        Second call: closes the run.

        Refused if any delegation is still Working — call GetStatus()
        on all pending delegations first. (A delegation you launched
        wait=True would already be collected here, with no pending poll.)
        """
        prefix = node._drain_notifications()
        # Liveness reconciled against the authoritative persistent log: a
        # delegation the log shows terminal is never "pending", so a stale
        # in-memory cache can no longer make Done() refuse forever (audit
        # BF-0/BF-2: the run4 deadlock, where a finished delegation read
        # "Working" until the watchdog killed the run UNGATED).
        pending = node._pending_delegations()
        if pending:
            # Soft nudge (NOT an "ERROR:" return, so it isn't counted as a
            # tool error): closing now is premature, but offer the three real
            # ways forward instead of a dead-end bounce.
            return (
                prefix +
                f"{len(pending)} delegation(s) still running: {pending}. "
                "Closing now is premature — you have two options:\n"
                "  (a) keep working: inspect results so far, write notes, or "
                "start another delegation while these finish;\n"
                "  (b) wait, then GetStatus(<id>) on each and interpret its "
                "results before you conclude.\n"
                "Re-call Done() once none are still running."
            )
        # Exit-interview capture (final stage): the conclusion is already
        # accepted + recorded; this Done() carries ONLY the retrospective.
        # Capture it, then actually close. The strategizer hears about the
        # interview ONLY after the critic accepted — never during its
        # working turns, so its orchestration context stays clean.
        if node._awaiting_retro:
            node._awaiting_retro = False
            node._record_retrospective("strategizer", "DONE", summary)
            node._done_warned = False
            route["kind"] = "done"
            route["summary"] = node._final_summary
            return prefix + "Run complete."
        # Milestone close-gate (Spec #1, HARD): every milestone must be DONE or
        # SKIPPED before the run can close. Auto-satisfy first (met gates tick
        # themselves). Forces engagement; MilestoneSkip(id, reason) is the
        # escape so it never deadlocks. Checked on every Done() call (the agent
        # can't bypass via the two-shot).
        _ms = getattr(node, "_milestones", None)
        if _ms is not None:
            _ms.auto_satisfy(node)
            _pend = _ms.pending()
            if _pend:
                node._milestone_block_count = getattr(
                    node, "_milestone_block_count", 0) + 1
                if node._milestone_block_count <= 3:
                    items = "; ".join(
                        f"{m['id']} ({m['description']})" for m in _pend)
                    return (
                        prefix + "Cannot close yet — these milestones are "
                        f"still PENDING: {items}. For each: "
                        "MilestoneComplete(id, brief) with a one-line why, or "
                        "MilestoneSkip(id, reason) if this study doesn't need "
                        "it. Then re-call Done(). (Process gate, not a tool "
                        "error.)"
                    )
                # Bounded escape (mirrors the 3-strikes UNGATED gate): after 3
                # blocked closes, auto-skip the rest so the run can never
                # deadlock. Recorded — the critic sees the forced skips and can
                # flag them.
                for m in _pend:
                    _ms.skip(m["id"],
                             "auto-skipped: unresolved after 3 close attempts")
                node._record_intervention(
                    "MILESTONE_AUTO_SKIP", "",
                    f"{len(_pend)} milestone(s) auto-skipped after 3 close "
                    "attempts")
        # Two-shot gate: first call warns, second call closes.
        if not node._done_warned:
            node._done_warned = True
            open_hypotheses: list[str] = []
            if node._ledger is not None:
                open_hypotheses = [
                    h["id"]
                    for h in node._ledger.list_all()
                    if h.get("current_status") == "OPEN"
                ]
            warn_parts = [
                "WARNING: first Done() call — confirm you are ready to close.",
                "Call Done() again to confirm and end the run.",
            ]
            if open_hypotheses:
                warn_parts.append(
                    f"Open hypotheses still in OPEN state: "
                    f"{open_hypotheses}. "
                    "Consider updating their status before closing."
                )
            # Dangling falsification attempts: a delegation flagged (or
            # post-hoc linked) as a falsification attempt whose hypothesis is
            # still OPEN means a test ran but its verdict was never recorded.
            # Content-based (independent of the read-time fire-once flag).
            if open_hypotheses:
                _open = set(open_hypotheses)
                dangling: list[str] = []
                with node._registry_lock:
                    for d_id, e in node._registry.items():
                        if (
                            e.get("status") == "Done"
                            and e.get("is_falsification_attempt")
                        ):
                            tied = [
                                h for h in (e.get("hypothesis_ids") or [])
                                if h in _open
                            ]
                            if tied:
                                dangling.append(f"{d_id}→{tied}")
                if dangling:
                    warn_parts.append(
                        "Falsification attempts whose hypotheses are still "
                        f"OPEN (record their verdict first): {dangling}. "
                        "Use HypothesisUpdate against each one's pre-registered "
                        "prediction."
                    )
            # (Pending milestones are now a HARD close-gate handled above — by
            # the time we reach this two-shot warn, all milestones are resolved.)
            # WARNING goes first; then any pending notifications.
            return "  ".join(warn_parts) + (("\n\n" + prefix.rstrip()) if prefix.strip() else "")
        # Pre-critic reproducibility gate (HARD): the deliverable must EXECUTE
        # and reproduce lazily BEFORE any critic consult is spent on it. A
        # broken / non-lazy pipeline.ipynb bounces straight back to the strategizer
        # to fix — the critic never wastes a turn reviewing a deliverable that
        # cannot even run. Bounded (N=3) so a persistently-broken pipeline still
        # lets the run end (the post-accept gate then marks it UNGATED).
        _REPRO_MAX = 6  # bounce budget before the run is declared FAILED
        _repro = node._reproduction_gate()
        if _repro is not None:
            node._repro_attempts = getattr(node, "_repro_attempts", 0) + 1
            n = node._repro_attempts
            if n <= _REPRO_MAX:
                node._record_intervention(
                    "REPRO_GATE_BOUNCE", "",
                    f"pre-critic reproduction gate failed (attempt {n}/{_REPRO_MAX})")
                # Escalate: after the first failure, push the agent to DEBUG with
                # CheckDeliverable() rather than re-Write blindly — it is the only
                # way to run pipeline.ipynb and see the real error.
                escalate = (
                    "" if n == 1 else
                    f"\n\nThis is failure {n}/{_REPRO_MAX}. Do NOT re-Write "
                    "pipeline.ipynb blindly. Use CheckDeliverable() to RUN it and "
                    "read the full error, fix the EXACT problem, CheckDeliverable() "
                    "again until it PASSES, then call Done(). After "
                    f"{_REPRO_MAX} failures the run is closed FAILED.")
                return (
                    prefix + "Cannot close yet — pipeline.ipynb failed the "
                    "reproduction gate (the runtime ran it before involving the "
                    "critic):\n\n" + _repro
                    + "\n\nFix it via WriteDeliverable('pipeline.ipynb', …) — and "
                    "verify with CheckDeliverable() before re-calling Done()."
                    + escalate)
            # Genuine non-convergence: the agent could not produce a reproducing
            # deliverable in _REPRO_MAX sighted attempts. Close FAILED (loud,
            # distinct from GATED/UNGATED) via the retrospective round — do NOT
            # spend the critic on a deliverable that does not even reproduce.
            node._record_intervention(
                "REPRO_GATE_FAILED", "",
                f"pipeline.ipynb never reproduced after {n} attempts — run FAILED")
            banner = (
                "## ⛔ FAILED RUN — DELIVERABLE NEVER REPRODUCED\n\n"
                f"pipeline.ipynb failed the reproduction gate on all {n} attempts; "
                "the run could not produce a runnable, lazy deliverable that "
                "re-derives the headline from the canonical ledger. This is a "
                "hard failure, not a gated or ungated conclusion.\n\n"
                "### Last gate error\n" + _repro + "\n\n---\n\n"
            )
            node._awaiting_retro = True
            node._final_summary = banner + summary
            node._done_warned = False
            return prefix + _FAILED_RETROSPECTIVE

        # Second call — run critic gate if critic is in the graph.
        if node._find_critic_name() is not None:
            # Synchronous critic gate
            _notes_dir = node._current_notes_dir
            _study_dir = node._study_dir
            _debug_dir = (
                _notes_dir.parent
                if _notes_dir is not None else None
            )
            notes_path = str(_notes_dir or "")
            _store_dir = (
                str(_debug_dir.parent / "experiment_data")
                if _debug_dir is not None else "(unknown)"
            )
            from ...critic_gate import problem_statement_block
            task_msg = (
                "<mode>GATE</mode>\n\n"
                "Final gate check before run closes. PASS to accept the "
                "conclusion; REVISE/REJECT only for a CRITICAL or MAJOR "
                "objection.\n\n"
                + problem_statement_block(_study_dir)
                + "<paths>\n"
                f"study_dir             = {_study_dir}\n"
                f"debug_dir             = {_debug_dir}\n"
                f"canonical_store       = {_store_dir}\n"
                "  ^ audit the ledger yourself: call RecallStore()/"
                "QueryStore(...) (namespace-aware — they see every design "
                "namespace, not just the default store). Do NOT hand-roll "
                "ExperimentData.from_file(project_dir=...) directly on this "
                "path alone — it misses any namespace store. Do NOT rely "
                "solely on the strategizer's self-reported numbers.\n"
                "delegation_log        = "
                f"{_debug_dir}/delegation_log.jsonl\n"
                f"diagnostics           = "
                f"{_debug_dir}/diagnostics.jsonl\n"
                f"strategizer_notes     = {notes_path}\n"
                "delegations_workspace = "
                f"{_debug_dir}/delegations/\n"
                f"deliverable           = {_study_dir}/pipeline.ipynb "
                "(the runtime EXECUTES the notebook lazily after this gate to "
                "verify the headline re-derives from the ledger with zero new "
                "evals; the notebook's own markdown cells ARE the writeup — "
                "there is no solution.md, do NOT flag it as missing)\n"
                "</paths>\n\n"
                # FULL conclusion — never truncate what the adversarial gate
                # must validate (a head-excerpt would let an over-claim in
                # the body pass unseen). A Done() summary is small; context
                # is not a concern, and final_summary.md is also available.
                f"Proposed conclusion:\n{summary}"
            )
            import json as _json
            ledger_dump = "(no hypotheses)"
            if node._ledger is not None:
                ledger_dump = _json.dumps(
                    {
                        h["id"]: node._ledger.get(h["id"])
                        for h in node._ledger.list_all()
                    },
                    indent=2,
                )
            attempts: list[str] = []
            if node._delegation_log is not None:
                attempts = [
                    f"{r['id']}: "
                    f"phase={r.get('phase')} "
                    f"hypotheses={r.get('hypothesis_ids')} "
                    f"is_falsification_attempt="
                    f"{r.get('is_falsification_attempt', False)}"
                    + (
                        " (linked post-hoc — scrutinise adequacy)"
                        if r.get("attempt_linked_post_hoc") else ""
                    )
                    for r in node._delegation_log.query_all()
                ]
            # Constraint snapshot NOW — single source of truth
            # (constraint_snapshot.py), the same one every worker delegation
            # and the FEEDBACK-mode critic call get. Tell the critic what was
            # actually spent so it judges the BEST HONEST conclusion reachable
            # within budget, rather than demanding falsification work the
            # budget no longer allows (which strands the close).
            from ....epistemics.constraint_snapshot import snapshot_for_node
            _snapshot = snapshot_for_node(node)
            # Milestones: show the critic each process milestone's resolution +
            # note/reason, so it can flag a hollow SKIP (a study that skipped a
            # gate it actually needed) — skips are unilateral, audited here.
            _ms_lines = []
            _ms_obj = getattr(node, "_milestones", None)
            if _ms_obj is not None:
                for m in _ms_obj.list_all():
                    _ms_lines.append(
                        f"{m['id']} [{m['status']}]: {m['description']}"
                        + (f" — note: {m['note']}" if m.get('note') else ""))
            _ms_block = "\n".join(_ms_lines) or "(none)"
            task_msg += (
                "\n\n<hypothesis_ledger>\n" + ledger_dump
                + "\n</hypothesis_ledger>\n\n"
                "<milestones>\n" + _ms_block
                + "\n</milestones>\n\n"
                "<delegation_flags>\n"
                + "\n".join(attempts)
                + "\n</delegation_flags>\n\n"
                + _snapshot.as_text() + "\n"
                "If the budget is exhausted, judge the BEST HONEST "
                "conclusion reachable within the evals actually spent: an "
                "honest INCONCLUSIVE/negative result whose falsification "
                "attempts were adequate FOR THE REMAINING BUDGET can PASS. "
                "Do NOT REVISE solely to demand evaluations the budget no "
                "longer allows — note them as future work instead.\n\n"
                "For each hypothesis, judge whether its stated "
                "falsification_criterion was actually tested by "
                "a delegation flagged is_falsification_attempt "
                "— adequacy of the test (given the budget), not mere "
                "presence of the flag."
            )
            _gate_started_at = datetime.now(
                tz=timezone.utc).isoformat(timespec="seconds")
            critique_text = node._invoke_critic(task_msg)
            _critic_usage = getattr(node, "_last_critic_usage", {}) or {}
            verdict = _parse_verdict(critique_text)
            # This is a delegation like any other (strategizer -> critic,
            # GATE mode) — log it as one. Previously the ONLY node-node
            # interaction never logged as a delegation, unlike the
            # FEEDBACK-mode call (AskForFeedback), which was inconsistent:
            # the interaction that actually decides whether the run closes
            # was invisible to delegation_log.jsonl.
            if node._delegation_log is not None:
                node._delegation_log.record(
                    id=f"GATE{datetime.now(tz=timezone.utc).strftime('%H%M%S')}",
                    from_node=node._name,
                    to_node=node._find_critic_name(),
                    task="Done() GATE acceptance check",
                    deliverable=critique_text,
                    hypothesis_ids=[],
                    started_at=_gate_started_at,
                    completed_at=datetime.now(
                        tz=timezone.utc).isoformat(timespec="seconds"),
                    status=f"GATE:{verdict}",
                    tokens_in=_critic_usage.get("input_tokens", 0) or 0,
                    tokens_out=_critic_usage.get("output_tokens", 0) or 0,
                    cost_usd=_critic_usage.get("total_cost_usd"),
                    constraints=_snapshot.as_dict(),
                )

            if verdict == "PASS":
                # Conclusion accepted + recorded. Now — and only now —
                # ask the exit interview as a separate turn.
                node._done_warned = False
                node._revise_count = 0
                node._awaiting_retro = True
                node._final_summary = summary
                return prefix + _EXIT_INTERVIEW

            # Non-PASS: reset the two-shot and count the revision internally.
            # After a bounded number of unsatisfiable verdicts, close GRACEFULLY
            # UNGATED rather than looping to recursion-limit. Policy (user, this
            # session): disclose the SUBSTANCE — the critic's standing objections
            # and that leaving them unresolved closes the run UNGATED — but NOT
            # the numeric attempt count (advertising "call Done() N times to
            # close" teaches the agent to exhaust the critic instead of earning a
            # PASS). So the agent is never blindsided by an unexplained ending,
            # and the limit stays un-gameable.
            node._done_warned = False
            node._revise_count = getattr(node, "_revise_count", 0) + 1
            if node._revise_count >= 3:
                banner = (
                    "## ⚠ UNGATED RUN\n\n"
                    "This run is NOT validated: the conclusion did not earn a "
                    "critic PASS — its objections (below) remained unresolved. "
                    "Closing honestly with them on record rather than looping.\n\n"
                    "### Outstanding critic findings\n"
                    + critique_text.strip() + "\n\n---\n\n"
                )
                node._revise_count = 0
                # Route through the retrospective round (D): even an UNGATED
                # close gets interviewed for capability gaps — these are the
                # runs that carry the most friction signal.
                node._awaiting_retro = True
                node._final_summary = banner + summary
                node._done_warned = False
                return prefix + _FAILED_RETROSPECTIVE
            return (
                prefix +
                f"Critic verdict: {verdict}. Address the findings below and "
                "revise your conclusion, then call Done() again — a run closes "
                "only on a critic PASS. If these objections stay unresolved, the "
                "run will close UNGATED with them on record, so resolve the "
                "substance rather than re-submitting unchanged.\n\n" + critique_text
            )

        # No critic — no review stage, so no exit interview; close
        # directly (the interview is "your work was reviewed, now a
        # question", which only applies when a critic gate ran).
        node._done_warned = False
        route["kind"] = "done"
        route["summary"] = summary
        return prefix + "Run complete."


    # AskForFeedback is only injected when a critic node is
    # connected AND this is the entry node (only the entry node
    # gates Done).
    critic_name: str | None = node._find_critic_name()
    spec = node._spec

    out: dict = {"Done": Done}
    if critic_name is not None:
        _critic_name = critic_name
        _node = node
        _critic_desc = spec.nodes[critic_name].description if spec else ""

        def AskForFeedback(hypothesis_ids: list | None = None) -> str:
            """Placeholder — __doc__ overridden below."""
            # Resolve hypothesis IDs
            h_ids: list[str] = []
            if hypothesis_ids is not None:
                h_ids = list(hypothesis_ids)
            elif _node._ledger is not None:
                h_ids = [h["id"] for h in _node._ledger.list_all()]

            started_at = datetime.now(tz=timezone.utc).isoformat(
                timespec="seconds"
            )
            # This is a delegation like any other (strategizer -> critic) —
            # same constraint snapshot, single source of truth, see
            # constraint_snapshot.py.
            from ....epistemics.constraint_snapshot import snapshot_for_node
            _snapshot = snapshot_for_node(_node)
            task_msg = _node._build_feedback_task_msg(
                h_ids, constraints_text=_snapshot.as_text())
            text = _node._invoke_critic(task_msg)
            _fb_usage = getattr(_node, "_last_critic_usage", {}) or {}

            # Log to delegation log
            if _node._delegation_log is not None:
                _node._delegation_log.record(
                    id=f"FB{datetime.now(tz=timezone.utc).strftime('%H%M%S')}",
                    from_node=_node._name,
                    to_node=_critic_name,
                    task="AskForFeedback (synchronous audit)",
                    deliverable=text,
                    hypothesis_ids=h_ids,
                    started_at=started_at,
                    completed_at=datetime.now(
                        tz=timezone.utc
                    ).isoformat(timespec="seconds"),
                    status="FEEDBACK",
                    tokens_in=_fb_usage.get("input_tokens", 0) or 0,
                    tokens_out=_fb_usage.get("output_tokens", 0) or 0,
                    cost_usd=_fb_usage.get("total_cost_usd"),
                    constraints=_snapshot.as_dict(),
                )

            return text


        AskForFeedback.__doc__ = (
            f"Synchronous find-only audit by: {_critic_desc} "
            "PASS is not a valid verdict here — REVISE, REJECT, or NOTED "
            "(no CRITICAL/MAJOR issue found; not an acceptance). "
            "hypothesis_ids: H-ids to focus on; None = all hypotheses auto-injected. "
            "Use Done() for the final gate check."
        )
        out["AskForFeedback"] = AskForFeedback

    return out
