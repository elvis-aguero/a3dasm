"""Orchestrator-only tool closures over the two process ledgers.

Hypothesis ledger (epistemics): ``HypothesisPropose`` / ``HypothesisUpdate``
/ ``LinkFalsificationAttempt`` — the MUTATE side. The read-only
``HypothesisList`` / ``HypothesisGet`` live in the declaration-gated shared
builder (``nodes/tools/routing``) so leaf workers can be granted them too.

Milestone ledger (process policy): ``MilestoneList`` / ``MilestonePropose`` /
``MilestoneComplete`` / ``MilestoneSkip``.

A mixin on the strategizer; every closure reads the ledgers off the node
(``self._ledger``, ``self._milestones``) at call time.
"""

from __future__ import annotations


class LedgerToolsMixin:
    def _build_hypothesis_closures(self) -> dict:
        """Build HypothesisPropose/Update/List/Get closures."""
        node = self

        from ..prompts.tool_catalog import tool_examples

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
            from ..epistemics.verdict_validator import (
                CLOSING_STATUSES as _CLOSING,
            )
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
