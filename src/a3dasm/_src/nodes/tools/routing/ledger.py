"""Orchestrator-only tools over the two process ledgers.

Hypothesis ledger (epistemics), the MUTATE side: ``HypothesisPropose`` /
``HypothesisUpdate`` / ``LinkFalsificationAttempt``. The read-only
``HypothesisList`` / ``HypothesisGet`` live in the declaration-gated shared
builder (``store.py``) so a leaf can be granted them too.

Milestone ledger (process policy): ``MilestoneList`` / ``MilestonePropose`` /
``MilestoneComplete`` / ``MilestoneSkip``.

``LedgerTools`` holds them, bound to one node; every tool reads the ledgers off
the node (``self.node._ledger``, ``self.node._milestones``) at call time, so a
ledger that only appears mid-run is still picked up.

``HypothesisUpdate`` carries the Popperian rules (CLAUDE.md §4). They are
stated here as named checks rather than one long body, in the order it applies
them: decode the evidence, the SUPPORTED-without-an-attempt confirm, the
single-source attribution rule, the cited-delegation-must-be-complete rule,
then provenance. Structure only — no rule's substance lives anywhere else.
"""
from __future__ import annotations

from typing import Any

from ....prompts.tool_catalog import tool_examples

_NO_LEDGER = "ERROR: hypothesis ledger not available in this run."
_NO_MILESTONES = "ERROR: milestone ledger not available in this run."


class LedgerTools:
    """The hypothesis- and milestone-ledger tools, bound to one node."""

    def __init__(self, node: Any) -> None:
        self.node = node

    # ── Hypothesis ledger (MUTATE) ───────────────────────────────────────────

    @tool_examples(
        "HypothesisPropose('Optimal t/L is near 0.08 — thin walls maximise "
        "buckling', 'a point with t/L in [0.10,0.14] beats "
        "buckling_load_norm 1.47', 'best at t/L ~ 0.08 ± 0.02', 0.55)",
    )
    def HypothesisPropose(
        self,
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
        node = self.node
        if node._ledger is None:
            return _NO_LEDGER
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
        self,
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
        node = self.node
        if node._ledger is None:
            return _NO_LEDGER

        evidence, err = _decode_evidence(evidence)
        if err is not None:
            return err
        refusal = self._supported_needs_attempt(hypothesis_id, status, comment)
        if refusal is not None:
            return refusal
        refusal = self._check_cited_delegation(hypothesis_id, evidence)
        if refusal is not None:
            return refusal

        result = node._ledger.update(
            hypothesis_id,
            status,
            comment,
            evidence,
            posterior,
            self._resolve_triggered_by(evidence),
        )
        return result + self._verdict_advisory(
            hypothesis_id, status, comment, evidence, result)

    def _supported_needs_attempt(
        self, hypothesis_id: str, status: str, comment: str
    ) -> str | None:
        """SUPPORTED without a completed falsification attempt: a TWO-SHOT CONFIRM.

        Not a hard block (§4, user-approved). A verdict is reversible when
        justified in writing, so this is a deliberate pause — not an impossible
        action. First call nudges; a re-call with a written justification in
        `comment` confirms. The verdict validator and the gate critic remain
        the downstream falsification floor.
        """
        node = self.node
        if status != "SUPPORTED" or node._delegation_log is None:
            return None
        completed = [
            r for r in node._delegation_log.query_all()
            if r.get("status") == "DONE"
            and r.get("is_falsification_attempt")
            and hypothesis_id in (r.get("hypothesis_ids") or [])
        ]
        if completed:
            return None
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
        return None

    def _check_cited_delegation(
        self, hypothesis_id: str, evidence: dict | None
    ) -> str | None:
        """A verdict must be attributable to ONE completed delegation.

        Single-source attribution (the principle behind what used to be a
        prompt format-rule): a closing verdict cites the ONE delegation whose
        report contains the cited numbers. A list / comma-joined value can't be
        attributed to a single source, so it is rejected here — the tool is the
        right place for this, not the prompt (CLAUDE.md §2).
        """
        node = self.node
        d_cited = (evidence or {}).get("delegation")
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
            d_cited is None
            or d_cited == "D000"
            or node._delegation_log is None
        ):
            return None
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
        return None

    def _resolve_triggered_by(self, evidence: dict | None) -> str | None:
        """The delegation this verdict RESTS ON.

        Provenance: a verdict is attributed to the source it rests on, so
        ``triggered_by`` is the delegation the agent CITED as evidence
        (validated as a single completed delegation / the D000 ground-truth
        anchor). Only when no evidence delegation was cited (non-closing
        updates) does it fall back to the most-recently-completed delegation.
        This previously always used last_completed_id, which mislabelled the
        audit trail whenever an unrelated delegation finished after the cited
        one (run 20260706T204732: H5 cited D011 but recorded
        triggered_by=D013).
        """
        node = self.node
        d_cited = (evidence or {}).get("delegation")
        if isinstance(d_cited, str) and d_cited:
            return d_cited
        triggered_by = None
        if node._delegation_log is not None:
            triggered_by = node._delegation_log.last_completed_id(node._name)
        if triggered_by is not None:
            return triggered_by
        with node._registry_lock:
            done_entries = [
                (d_id, entry)
                for d_id, entry in node._registry.items()
                if entry.get("status") in ("Done", "Errored")
            ]
        return done_entries[-1][0] if done_entries else None

    def _verdict_advisory(
        self,
        hypothesis_id: str,
        status: str,
        comment: str,
        evidence: dict | None,
        result: str,
    ) -> str:
        """Advisory live verdict-substance check (#9), as a suffix or "".

        Closing verdicts only, and only when a NEW entry was actually appended
        ("Updated …") — not on ERROR/SETTLED no-ops. Non-blocking: it appends a
        charter critique to what the agent sees this turn, but never changes
        the update.
        """
        from ....epistemics.verdict_validator import (
            CLOSING_STATUSES as _CLOSING,
        )
        if status not in _CLOSING or not result.startswith("Updated "):
            return ""
        advisory = self.node._run_verdict_validator(
            hypothesis_id, status, comment, evidence,
        )
        return f"\n{advisory}" if advisory else ""

    def LinkFalsificationAttempt(
        self, delegation_id: str, hypothesis_id: str
    ) -> str:
        """Retroactively mark a completed delegation as a falsification
        ATTEMPT of a registered hypothesis (the read-time safety net for
        when the attempt was not declared up front at Delegate time).

        Links ONLY — it does NOT record a verdict and CANNOT change the
        hypothesis's pre-registered prediction. You must still call
        HypothesisUpdate to record the verdict, judged against that
        immutable prediction. Link only if the delegation genuinely tested
        the prediction — never retrofit an exploratory result."""
        node = self.node
        if node._ledger is None:
            return _NO_LEDGER
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
            node._delegation_log.mark_attempt(delegation_id, hypothesis_id)
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

    # ── Milestone ledger (process policy) ────────────────────────────────────

    def MilestoneList(self) -> str:
        """List process milestones with id, status, description.

        Milestones are PROCESS steps (do X before Y; get Z ready), distinct
        from hypotheses (epistemics). Default milestones self-resolve when
        their condition is met; you author your own with MilestonePropose."""
        if self.node._milestones is None:
            return "Milestone ledger not available in this run."
        return self.node._milestones.format()

    def MilestonePropose(self, description: str) -> str:
        """Add your own process milestone. Returns its id (M1, M2, …).

        While pending, it joins the backlog that gates delegating to the
        implementer (exactly like the default milestones) — so use it to
        hold yourself to a process step you don't want to skip."""
        if self.node._milestones is None:
            return _NO_MILESTONES
        return self.node._milestones.propose(description)

    def MilestoneComplete(self, milestone_id: str, note: str) -> str:
        """Mark a milestone DONE. A brief `note` (one line on WHY it's
        satisfied — what was done / which delegation) is REQUIRED, so
        ticking is a deliberate, auditable act, not a rubber stamp."""
        if self.node._milestones is None:
            return _NO_MILESTONES
        if not note or not note.strip():
            return (
                "ERROR: a brief note is required to complete a milestone — "
                "one line on why it's satisfied (what you did / which "
                "delegation). This keeps ticking honest and auditable."
            )
        return self.node._milestones.complete(milestone_id, note)

    def MilestoneSkip(self, milestone_id: str, reason: str) -> str:
        """Skip a milestone this study legitimately doesn't need (give a
        reason). The escape hatch so soft gates never deadlock you."""
        if self.node._milestones is None:
            return _NO_MILESTONES
        return self.node._milestones.skip(milestone_id, reason)


def _decode_evidence(evidence):
    """Accept the JSON-string shape an MCP caller may send. Returns (value, err)."""
    if not isinstance(evidence, str):
        return evidence, None
    import json as _json
    try:
        return _json.loads(evidence), None
    except _json.JSONDecodeError:
        return None, (
            "ERROR: evidence must be a JSON object like "
            '{"delegation": "D004", "numbers": {...}}.'
        )


def build_ledger_closures(node) -> dict:
    """The ledger tools for one node, by registered name.

    Every one is orchestrator-only and MUTATES a ledger; the caller gates them
    on the agent's declared ``tools``, like every other capability. The
    read-only HypothesisList/HypothesisGet are NOT here — they live in
    ``store.py``'s shared builder so a leaf can be granted them too.
    """
    t = LedgerTools(node)
    return {
        "HypothesisPropose": t.HypothesisPropose,
        "HypothesisUpdate": t.HypothesisUpdate,
        "LinkFalsificationAttempt": t.LinkFalsificationAttempt,
        "MilestoneList": t.MilestoneList,
        "MilestonePropose": t.MilestonePropose,
        "MilestoneComplete": t.MilestoneComplete,
        "MilestoneSkip": t.MilestoneSkip,
    }
