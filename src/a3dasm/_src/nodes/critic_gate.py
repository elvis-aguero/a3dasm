"""Critic consultation: find the connected critic, run a synchronous FEEDBACK/GATE
audit, persist the verdict, build the feedback task message. A mixin on the
strategizer (uses its instance attrs + RecordingMixin methods via MRO)."""
from __future__ import annotations

import os
from pathlib import Path

# Bound on how many earlier reviews are echoed back to the critic, and the
# total char budget for the digest — injected context must be current and
# bounded, not an ever-growing transcript.
_MAX_PRIOR_REVIEWS = 6
_PRIOR_REVIEWS_CHAR_BUDGET = 6000

# #9 live verdict validator: on the Nth repeat flag of the SAME hypothesis,
# append a louder warning pointing at the gate critic (the lightweight "teeth";
# a full critic re-audit on repeat is a deferred §4 follow-up).
VERDICT_FLAG_ESCALATE_AFTER = 2

_VERDICT_VALIDATOR_OFF = {"0", "false", "off", "no"}


def verdict_validator_enabled() -> bool:
    """The #9 live verdict validator is ON by default. Set the env var
    ``F3DASM_VERDICT_VALIDATOR=0`` (or false/off/no) to disable it entirely —
    HypothesisUpdate then behaves exactly as it did before #9 (no judge call, no
    note, no diagnostics). The single, easy kill switch for the sentinel.
    """
    return (
        os.environ.get("F3DASM_VERDICT_VALIDATOR", "1").strip().lower()
        not in _VERDICT_VALIDATOR_OFF
    )


def _prior_rulings_digest(h: dict, *, max_entries: int = 6, max_chars: int = 240) -> str:
    """Digest of the verdict validator's EARLIER rulings on this same hypothesis,
    from its status_log. The current (just-applied) entry being judged is the
    last one — exclude it; everything before is prior history. Without this the
    judge is stateless and a borderline verdict oscillates between calls; with it
    the judge sees what it ruled before (and why) and must justify any reversal.
    Returns "" when there is no prior ruling (the first verdict on a hypothesis).
    """
    log = (h or {}).get("status_log") or []
    prior = log[:-1]  # the last entry is the verdict currently under judgement
    if not prior:
        return ""
    lines = []
    for e in prior[-max_entries:]:
        st = e.get("status", "?")
        cm = " ".join((e.get("comment") or "").split())[:max_chars]
        line = f"- ruled {st}: {cm}"
        note = " ".join((e.get("validator_note") or "").split())[:max_chars]
        if note:
            line += f"\n    [your ruling then: {note}]"
        lines.append(line)
    return "\n".join(lines)


def _extract_md_section(text: str, header: str) -> str:
    """Return the body under a `### Header` up to the next `### ` heading.

    Used to pull just the Verdict and Findings out of a persisted review,
    dropping Actions/Numbers/Retrospective noise. Empty string if absent.
    """
    lines = text.splitlines()
    out: list[str] = []
    capturing = False
    for line in lines:
        if line.strip() == header:
            capturing = True
            continue
        if capturing and line.lstrip().startswith("### "):
            break
        if capturing:
            out.append(line)
    return "\n".join(out).strip()


class CriticGateMixin:
    """Mixin carrying the four critic-consultation helpers for StrategizerNode.

    Relies on instance attributes and RecordingMixin methods resolved via MRO:
      self._spec, self._outgoing, self._worker_adapters, self._current_notes_dir,
      self._critic_calls, self._study_dir, self._name,
      self._record_usage, self._role_of, self._record_retrospective.
    Defines no __init__.
    """

    def _find_critic_name(self) -> str | None:
        """Name of the first connected critic worker, or None."""
        spec = self._spec
        if spec is None or not hasattr(spec, "nodes"):
            return None
        for target in self._outgoing:
            agent = spec.nodes.get(target)
            if (
                agent is not None
                and getattr(agent, "role", None) == "critic"
                and target in self._worker_adapters
            ):
                return target
        return None

    def _prior_reviews_digest(self) -> str:
        """A bounded `<prior_reviews_this_run>` block of THIS run's earlier
        critic reviews (verdict + findings only), or "" if none.

        The critic is invoked one-shot per gate with no live session and no
        RecallHistory, so without this it cannot see what it already ruled and
        can silently contradict an earlier verdict (the H1 SUPPORTED->FALSIFIED
        ->back whipsaw that drove the REVISE-spin). Echoing its standing
        objections back forces consistency: it may still reverse, but only by
        saying so and citing new evidence — never silently.
        """
        notes = self._current_notes_dir
        if notes is None:
            return ""
        review_dir = Path(notes).parent / "critic_reviews"
        if not review_dir.is_dir():
            return ""
        files = sorted(review_dir.glob("call_*.md"))
        if not files:
            return ""
        elided = max(0, len(files) - _MAX_PRIOR_REVIEWS)
        kept = files[-_MAX_PRIOR_REVIEWS:]
        blocks: list[str] = []
        for f in kept:
            try:
                text = f.read_text(encoding="utf-8")
            except Exception:  # noqa: BLE001
                continue
            n = f.stem.replace("call_", "")
            verdict = _extract_md_section(text, "### Verdict") or "(unparsed)"
            findings = _extract_md_section(text, "### Findings") or "(none)"
            blocks.append(
                f"--- call_{n} ---\n"
                f"Verdict: {verdict}\n"
                f"Findings:\n{findings}"
            )
        # Char budget: drop oldest kept blocks until under budget.
        while blocks and sum(len(b) for b in blocks) > _PRIOR_REVIEWS_CHAR_BUDGET:
            blocks.pop(0)
            elided += 1
        if not blocks:
            return ""
        elided_note = (
            f"\n({elided} earlier review(s) elided for brevity.)"
            if elided else ""
        )
        return (
            "<prior_reviews_this_run>\n"
            "You have already reviewed this run. Your standing verdicts and "
            "objections are below. Be CONSISTENT with them: do not silently "
            "contradict a verdict you reached earlier, and do not re-raise an "
            "objection the strategizer has since resolved. You MAY reverse a "
            "prior position, but only by stating which call you are reversing "
            "and citing the Charter clause and the NEW evidence that justifies "
            "it.\n"
            + "\n\n".join(blocks)
            + elided_note
            + "\n</prior_reviews_this_run>\n\n"
        )

    def _invoke_critic(self, task_msg: str) -> str:
        """Synchronously invoke the connected critic; returns its
        text or an ERROR string.

        The critic is a worker too: under F3DASM_DEBUG its full transcript is
        streamed to disk, its verdict/review is ALWAYS persisted (the PASS
        branch doesn't echo it to the strategizer, so this is the only place
        the deciding verdict is auditable), and its ### Retrospective is
        recorded like every other node's (#7).
        """
        critic_name = self._find_critic_name()
        if critic_name is None:
            return "ERROR: no critic connected."
        # Cross-round memory: prepend this run's earlier reviews so the critic
        # stays consistent instead of whipsawing its own verdicts (Fix A).
        task_msg = self._prior_reviews_digest() + task_msg
        adapter = self._worker_adapters[critic_name]
        worker = (
            adapter.copy() if hasattr(adapter, "copy") else adapter
        )
        self._critic_calls = getattr(self, "_critic_calls", 0) + 1
        _n = self._critic_calls
        from ..backends.base import (
            debug_enabled as _dbg,
        )
        from ..backends.base import (
            get_transcript_sink as _get_sink,
        )
        from ..backends.base import (
            set_transcript_sink as _set_sink,
        )
        _notes = self._current_notes_dir
        _prev_sink = _get_sink()
        if _dbg() and _notes is not None:
            _set_sink(str(
                _notes.parent / "transcripts" / "critic"
                / f"call_{_n:03d}.jsonl"))
        try:
            critique = worker.invoke(
                [{"role": "user", "content": task_msg}]
            )
        except Exception as _exc:  # noqa: BLE001
            # An infrastructure failure invoking the critic — NOT a problem with
            # your deliverable. Give a one-line cause, not a raw traceback, and a
            # constructive next step.
            critique = (
                "ERROR: the critic could not be invoked due to an "
                f"infrastructure error ({type(_exc).__name__}: "
                f"{str(_exc)[:200]}). This is not a defect in your deliverable. "
                "Re-call Done() to retry the gate; if it persists, the run will "
                "close without a critic PASS (the failure is on record)."
            )
        finally:
            # Restore the strategizer's own sink (same thread-local).
            if _dbg() and _notes is not None:
                _set_sink(_prev_sink)
        # Account the critic's tokens/cost — critic consults are real LLM calls
        # and must land in token_totals AND telemetry (they were previously
        # uncounted, undercounting run cost and omitting the 'critic' role).
        self._record_usage(
            getattr(worker, "last_usage", {}) or {},
            role=self._role_of(critic_name),
            model=getattr(worker, "model", None),
            phase="critic_review",
            delegation_id=f"critic-{_n}",
        )
        # Always-on: persist the verdict/review to disk + record retrospective.
        self._persist_critic_review(_n, critique)
        self._record_retrospective("critic", f"critic-{_n}", critique)
        return critique

    # ── #9: live hypothesis-verdict validator (advisory, non-blocking) ────────

    def _invoke_verdict_validator(self, prompt: str) -> str:
        """One-shot LLM judgement reusing the CRITIC's adapter (same judge as the
        gate), but lighter than ``_invoke_critic`` — it does NOT persist a critic
        review or a retrospective. Records token usage so the call lands in the
        run cost. Best-effort: returns "" if no critic is connected or the call
        fails (the validator is advisory; a missing judge must not break the run).
        """
        critic_name = self._find_critic_name()
        if critic_name is None:
            return ""
        adapter = self._worker_adapters[critic_name]
        worker = adapter.copy() if hasattr(adapter, "copy") else adapter
        # Tight budget: this advisory judge must NOT inherit a real agent turn's
        # 5×600s stream/retry budget. A hung CLI stream once froze a whole run
        # for ~89 min here (run 20260627T211310). idle=120 + retry_max=1 abort
        # ~2 min after the stream goes silent; the call is advisory, so on any
        # failure the verdict stands (see _run_verdict_validator).
        try:
            reply = worker.invoke(
                [{"role": "user", "content": prompt}],
                idle_timeout=120.0, retry_max=1,
            )
        except TypeError:
            # A backend/stub without the budget kwargs — fall back gracefully.
            reply = worker.invoke([{"role": "user", "content": prompt}])
        except Exception:  # noqa: BLE001
            return ""
        self._record_usage(
            getattr(worker, "last_usage", {}) or {},
            role=self._role_of(critic_name),
            model=getattr(worker, "model", None),
            phase="verdict_validation",
            delegation_id="verdict-validator",
        )
        return reply or ""

    def _cited_delegation_brief(self, evidence: dict | None) -> str | None:
        """A short brief on the cited delegation — what it was asked to do and
        whether it was flagged a falsification attempt — so the judge can weigh
        §2 attempt-adequacy. The numeric result the verdict rests on is carried
        separately in ``evidence['numbers']``. None if no delegation is cited or
        the record is absent.
        """
        d = (evidence or {}).get("delegation")
        if not d or self._delegation_log is None:
            return None
        rec = next(
            (r for r in self._delegation_log.query_all() if r.get("id") == d),
            None,
        )
        if rec is None:
            return None
        return (
            f"delegation {d} (to {rec.get('to_node')}, "
            f"is_falsification_attempt={rec.get('is_falsification_attempt')}, "
            f"evals={rec.get('evals')}):\n"
            f"  task: {rec.get('task', '')}\n"
            f"  deliverable: {rec.get('deliverable', '')}"
        )

    def _run_verdict_validator(
        self, h_id: str, status: str, comment: str, evidence: dict | None,
    ) -> str:
        """Advise (never block) on a closing verdict's substance against the
        charter. Persists the critique on the verdict it judged, emits a
        ``VERDICT_SUBSTANCE_FLAG`` diagnostics event on a flag, and on the Nth
        repeat flag of the same hypothesis appends a louder gate-critic warning.
        Returns the text to append to the HypothesisUpdate result ("" = no concern
        / validator unavailable). NEVER raises — the update it annotates must
        always stand (Q1=(B), advise-with-teeth).
        """
        if not verdict_validator_enabled():
            return ""  # kill switch (F3DASM_VERDICT_VALIDATOR=0) — fully bypassed
        try:
            from ..verdict_validator import build_judge_prompt, parse_judge_reply
            h = (self._ledger.get(h_id) or {}) if self._ledger else {}
            if not h:
                return ""
            prompt = build_judge_prompt(
                statement=h.get("statement", ""),
                prediction=h.get("prediction", ""),
                criterion=h.get("falsification_criterion", ""),
                status=status,
                comment=comment,
                evidence=evidence,
                delegation_report=self._cited_delegation_brief(evidence),
                prior_rulings=_prior_rulings_digest(h),
            )
            reply = self._invoke_verdict_validator(prompt)
            if not reply:
                return ""  # no judge / call failed — silent; the update stands
            flagged, critique = parse_judge_reply(reply)
            if not flagged:
                if self._ledger:
                    self._ledger.annotate_last(h_id, "validated: no charter concern")
                return ""
            # Flagged: persist on the verdict, surface to the agent, count, escalate.
            if self._ledger:
                self._ledger.annotate_last(h_id, critique)
            self._record_science_drift({
                "error_type": "VERDICT_SUBSTANCE_FLAG",
                "hypothesis": h_id,
                "status": status,
                "message": critique[:300],
            })
            counts = self.__dict__.setdefault("_verdict_flag_counts", {})
            counts[h_id] = counts.get(h_id, 0) + 1
            msg = f"[VERDICT VALIDATOR] {critique}"
            if counts[h_id] >= VERDICT_FLAG_ESCALATE_AFTER:
                msg += (
                    f"\n[VERDICT VALIDATOR] {h_id}'s verdict has now been flagged "
                    f"{counts[h_id]}× — the gate critic will scrutinise this. "
                    "Re-examine the cited charter clause before relying on it."
                )
            return msg
        except Exception:  # noqa: BLE001
            return ""  # advisory must never break the update

    def _persist_critic_review(self, n: int, critique_text: str) -> None:
        """Write the critic's full review to debug/critic_reviews/ so the
        deciding verdict is auditable regardless of PASS/REVISE. Best-effort."""
        try:
            notes = self._current_notes_dir
            if notes is None:
                return
            d = Path(notes).parent / "critic_reviews"
            d.mkdir(parents=True, exist_ok=True)
            (d / f"call_{n:03d}.md").write_text(
                critique_text or "", encoding="utf-8")
        except Exception:  # noqa: BLE001
            pass

    def _build_feedback_task_msg(self, h_ids: list) -> str:
        """<mode>FEEDBACK</mode> task message with paths block."""
        notes_path = str(self._current_notes_dir or "")
        _notes_dir = self._current_notes_dir
        _study_dir = self._study_dir
        _debug_dir = (
            _notes_dir.parent if _notes_dir is not None else None
        )
        return (
            "<mode>FEEDBACK</mode>\n\n"
            "Perform a synchronous find-only adversarial audit.  "
            "PASS is not an available verdict — return REVISE or "
            "REJECT with your findings.\n\n"
            "<paths>\n"
            f"study_dir             = {_study_dir}\n"
            f"debug_dir             = {_debug_dir}\n"
            f"delegation_log        = {_debug_dir}/delegation_log.jsonl\n"
            f"diagnostics           = {_debug_dir}/diagnostics.jsonl\n"
            f"strategizer_notes     = {notes_path}\n"
            f"delegations_workspace = {_debug_dir}/delegations/\n"
            f"deliverable            = {_study_dir}/pipeline.ipynb "
            "(the runtime EXECUTES the notebook lazily after this gate to verify "
            "the headline re-derives from the ledger with zero new evals; the "
            "notebook's own markdown cells ARE the writeup — there is no "
            "solution.md, do NOT flag it as missing)\n"
            "</paths>\n\n"
            f"Focus hypotheses: {h_ids if h_ids else 'all'}"
        )
