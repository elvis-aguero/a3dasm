"""Structured hypothesis ledger for agentic runs.

Manages one file under the study directory:

- ``hypotheses.json``   — keyed by hypothesis ID; append-only
                          status_log following a Popperian schema.

Every hypothesis must carry a falsification criterion, a prediction,
and a prior probability.  Every status update must carry evidence and
a posterior probability.  The schema is strict: ``from_dict`` raises
on missing fields; old ledger files are not compatible.

Delegation logging has moved to
:class:`~delegation_log.DelegationLog`
(``debug/delegation_log.jsonl``).
"""

from __future__ import annotations

import json
import os
import re
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

__all__ = ["HypothesisLedger", "HypothesisEntry", "StatusLogEntry"]

VALID_STATUSES = frozenset(
    {"OPEN", "SUPPORTED", "FALSIFIED", "INCONCLUSIVE"})
CLOSING_STATUSES = frozenset(
    {"SUPPORTED", "FALSIFIED", "INCONCLUSIVE"})
MAX_OPEN = 3
MAX_FIELD_LEN = 500

_QUANT_RE = re.compile(r"[<>=≤≥]|\d")
_NUMBERED_RE = re.compile(
    r"\(1\).*\(2\)|\b1\.\s.*\b2\.\s", re.DOTALL
)


def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat(timespec="seconds")


def _is_compound(statement: str) -> bool:
    """Heuristic: claim contains two quantified sub-claims."""
    if _NUMBERED_RE.search(statement):
        return True
    # Connectors are deliberately narrow (" and ", "; " only).
    # False negatives are preferred over blocking legitimate single
    # claims; the critic catches what the heuristic misses.
    for conn in (" and ", "; "):
        if conn in statement:
            left, right = statement.split(conn, 1)
            if _QUANT_RE.search(left) and _QUANT_RE.search(right):
                return True
    return False


@dataclass
class StatusLogEntry:
    status: str
    comment: str
    evidence: dict | None
    posterior: float | None
    triggered_by: str | None
    ts: str = field(default_factory=_now_iso)
    # Advisory critique from the #9 live verdict validator on THIS status change
    # (None = not validated / no concern). Defaulted, so old ledger entries that
    # predate the field still load via from_dict.
    validator_note: str | None = None


@dataclass
class HypothesisEntry:
    id: str
    statement: str
    falsification_criterion: str
    prediction: str
    prior: float
    proposed_by: str
    proposed_at: str
    status_log: list[StatusLogEntry] = field(default_factory=list)

    @property
    def current_status(self) -> str:
        return (
            self.status_log[-1].status if self.status_log else "OPEN"
        )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "statement": self.statement,
            "falsification_criterion": self.falsification_criterion,
            "prediction": self.prediction,
            "prior": self.prior,
            "proposed_by": self.proposed_by,
            "proposed_at": self.proposed_at,
            "status_log": [asdict(e) for e in self.status_log],
        }

    @classmethod
    def from_dict(cls, data: dict) -> HypothesisEntry:
        # STRICT: missing fields raise.  Old ledger files do not load.
        log = [StatusLogEntry(**e) for e in data["status_log"]]
        return cls(
            id=data["id"],
            statement=data["statement"],
            falsification_criterion=data["falsification_criterion"],
            prediction=data["prediction"],
            prior=float(data["prior"]),
            proposed_by=data["proposed_by"],
            proposed_at=data["proposed_at"],
            status_log=log,
        )


class HypothesisLedger:
    """Manages ``hypotheses.json`` on disk.

    All mutating operations are atomic under a threading lock so that
    concurrent delegation threads (background workers) cannot corrupt
    state.
    """

    def __init__(self, notes_dir: Path) -> None:
        self._notes_dir = Path(notes_dir)
        self._hypotheses_path = self._notes_dir / "hypotheses.json"
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Hypothesis operations
    # ------------------------------------------------------------------

    def propose(
        self,
        statement: str,
        falsification_criterion: str,
        prediction: str,
        prior,
        proposed_by: str,
    ) -> str:
        """Propose a hypothesis; returns its ID or an ERROR string."""
        try:
            prior_f = float(prior)
        except (TypeError, ValueError):
            return (
                f"ERROR: prior must be a number in (0, 1), got {prior!r}. It is "
                "your honest plausibility for the claim BEFORE testing — e.g. "
                "0.6 if you lean toward it, 0.3 if you doubt it. Pass a float."
            )
        if not 0.0 < prior_f < 1.0:
            return (
                f"ERROR: prior must be strictly between 0 and 1, got {prior_f}. "
                "0 or 1 means certainty, and a claim you're certain of isn't a "
                "hypothesis to test — pick a value like 0.4-0.7 reflecting how "
                "plausible it is before evidence."
            )
        # Structural guards stay HARD (empty fields break the schema). The two
        # FORMATTING guards below — over-length and compound-claim — are now
        # NUDGES: the hypothesis is still registered, with advice appended, so a
        # slightly-long or compound phrasing never costs the agent a round-trip.
        # (Quality is caught at the gate; a blocked propose only burns turns.)
        nudges: list[str] = []
        for name, value in (
            ("statement", statement),
            ("falsification_criterion", falsification_criterion),
            ("prediction", prediction),
        ):
            if not value or not str(value).strip():
                return f"ERROR: {name} must be non-empty."
            if len(str(value)) > MAX_FIELD_LEN:
                nudges.append(
                    f"[NUDGE] {name} is {len(str(value))} chars "
                    f"(over the {MAX_FIELD_LEN}-char guideline) — a hypothesis "
                    "reads best as one concise claim; consider tightening it."
                )
        if _is_compound(statement):
            nudges.append(
                "[NUDGE] statement looks like a compound claim (multiple "
                "quantified sub-claims). One falsifiable claim per hypothesis "
                "tests more cleanly — consider splitting it next time."
            )

        def _normalize(s: str) -> str:
            return " ".join(s.casefold().split())

        norm_new = _normalize(statement)

        with self._lock:
            data = self._load()
            # Duplicate-statement guard: reject semantically identical claims.
            for existing_id, existing in data.items():
                if _normalize(existing["statement"]) == norm_new:
                    return (
                        f"ERROR: duplicate of {existing_id}. Identical"
                        " hypothesis already exists — test it or revise"
                        " the claim."
                    )
            open_count = sum(
                1 for h in data.values()
                if h["status_log"]
                and h["status_log"][-1]["status"] == "OPEN"
            )
            # Open-count ceiling is a TIP, not a gate: opening a hypothesis is
            # fully reversible (close or retract it), and exploring several
            # designs at once legitimately wants more than MAX_OPEN open. So
            # register it and, when over the soft ceiling, append advice to close
            # the ones already settled — guidance, never a refusal.
            if open_count >= MAX_OPEN:
                nudges.append(
                    f"[NUDGE] you now have {open_count + 1} OPEN hypotheses "
                    f"(soft ceiling {MAX_OPEN}). Tracking several at once is fine "
                    "— e.g. one per design you are exploring — but stale open "
                    "hypotheses dilute focus: close the ones you have settled "
                    "(SUPPORTED / FALSIFIED / INCONCLUSIVE) when you can."
                )
            h_id = f"H{len(data) + 1}"
            ts = _now_iso()
            entry = HypothesisEntry(
                id=h_id,
                statement=statement,
                falsification_criterion=falsification_criterion,
                prediction=prediction,
                prior=prior_f,
                proposed_by=proposed_by,
                proposed_at=ts,
                status_log=[StatusLogEntry(
                    status="OPEN",
                    comment="initial proposal",
                    evidence=None,
                    posterior=prior_f,
                    triggered_by=None,
                    ts=ts,
                )],
            )
            data[h_id] = entry.to_dict()
            self._save(data)
            # ID first (callers parse it out), nudges appended as advice.
            return h_id if not nudges else h_id + "\n" + "\n".join(nudges)

    def update(
        self,
        h_id: str,
        status: str,
        comment: str,
        evidence: dict | None,
        posterior,
        triggered_by: str | None,
    ) -> str:
        """Append a status-change entry; returns confirmation or ERROR.
        """
        if status not in VALID_STATUSES:
            return (
                f"ERROR: invalid status {status!r}. Use one of: OPEN (still "
                "testing), SUPPORTED (survived an adequate refutation attempt), "
                "FALSIFIED (an adequate test contradicted the prediction), "
                "INCONCLUSIVE (the test was too weak to decide)."
            )
        try:
            post_f = float(posterior)
        except (TypeError, ValueError):
            return (
                f"ERROR: posterior must be a number in [0, 1], got {posterior!r}. "
                "It is your updated belief in the claim AFTER this evidence — "
                "higher than the prior if the evidence supported it, lower if it "
                "pushed against. Pass a float."
            )
        if not 0.0 <= post_f <= 1.0:
            return (
                f"ERROR: posterior must be in [0, 1], got {post_f}. It is a "
                "probability (your updated belief in the claim), not a score or "
                "an objective value."
            )
        has_delegation = (
            isinstance(evidence, dict) and "delegation" in evidence
        )
        if status in CLOSING_STATUSES and not has_delegation:
            return (
                f"ERROR: closing status {status!r} requires evidence "
                "with a 'delegation' key, e.g. "
                '{"delegation": "D004", "numbers": {"best_y": 1.62}}.'
            )
        with self._lock:
            data = self._load()
            if h_id not in data:
                return f"ERROR: hypothesis {h_id!r} not found."
            log = data[h_id]["status_log"]
            current = log[-1]["status"] if log else "OPEN"
            cited = {
                (e.get("evidence") or {}).get("delegation")
                for e in log
                if e.get("evidence")
            }
            new_d = (evidence or {}).get("delegation")
            # A no-op is when LITERALLY nothing changes — same status, no new
            # delegation, AND the same numbers/posterior as the latest entry.
            # Correcting the cited numbers or moving the posterior IS a real
            # update (it is exactly what a NUMBERS_MATCH nudge asks for), so it
            # must be allowed — otherwise the agent is trapped between "correct
            # the evidence" and "you may not update", and loops.
            last = log[-1] if log else {}
            last_numbers = (last.get("evidence") or {}).get("numbers") or {}
            new_numbers = (evidence or {}).get("numbers") or {}
            nothing_changed = (
                status == current
                and (new_d is None or new_d in cited)
                and new_numbers == last_numbers
                and post_f == last.get("posterior")
            )
            if nothing_changed:
                return (
                    f"{h_id} is already {current} with this exact evidence — "
                    "it is SETTLED. Do not re-submit it. Move on to your next "
                    "step: run a new test, open a new hypothesis, or call "
                    "Done(). Re-update only on a status change, a new "
                    "delegation, or corrected numbers/posterior."
                )
            if (
                status == "OPEN"
                and current != "OPEN"
                and not has_delegation
            ):
                # Un-falsifying RE-ASSERTS a hypothesis the evidence killed —
                # that is a NEW claim, not a retraction, so it still requires its
                # own delegation (anti-dodge: no escaping a falsification on
                # "second thoughts").
                if current == "FALSIFIED":
                    return (
                        "ERROR: reopening a FALSIFIED hypothesis requires new "
                        "evidence with a 'delegation' key — un-falsifying is a "
                        "new claim, not a retraction."
                    )
                # Retracting SUPPORTED/INCONCLUSIVE → OPEN is WITHDRAWING a
                # claim, not asserting one — it must not deadlock for lack of a
                # NEW delegation (the close-time self-correction: "I marked this
                # SUPPORTED without a falsification attempt; per Charter §2 it
                # should be OPEN"). Carry forward the evidence the verdict was
                # based on; every closed hypothesis cited one. The comment
                # explains the correction and the status_log keeps the full
                # audit trail (so the critic still sees a frivolous reopen).
                prior_cited = next(
                    (
                        d for d in (
                            (e.get("evidence") or {}).get("delegation")
                            for e in reversed(log)
                        ) if d
                    ),
                    None,
                )
                if prior_cited is None:
                    return (
                        "ERROR: reopening a closed hypothesis requires "
                        "evidence with a 'delegation' key."
                    )
                evidence = {**(evidence or {}), "delegation": prior_cited}
                has_delegation = True
            entry = StatusLogEntry(
                status=status,
                comment=comment,
                evidence=evidence,
                posterior=post_f,
                triggered_by=triggered_by,
                ts=_now_iso(),
            )
            log.append(asdict(entry))
            self._save(data)
            return (
                f"Updated {h_id}: status → {status} "
                f"(belief {post_f})."
            )

    def annotate_last(self, h_id: str, note: str) -> None:
        """Stamp an advisory note on the most recent status_log entry of ``h_id``.

        The #9 live verdict validator records its critique here — on the exact
        status change it judged. Best-effort: a no-op if the hypothesis or its
        log is absent, and it never raises (an advisory must never break the
        update it annotates).
        """
        with self._lock:
            data = self._load()
            h = data.get(h_id)
            if not h or not h.get("status_log"):
                return
            h["status_log"][-1]["validator_note"] = note
            self._save(data)

    def list_all(self) -> list[dict]:
        """Summaries: id, statement, current_status, prior, belief."""
        with self._lock:
            data = self._load()
        result = []
        for h in data.values():
            log = h.get("status_log", [])
            current = log[-1]["status"] if log else "OPEN"
            posts = [
                e["posterior"] for e in log
                if e.get("posterior") is not None
            ]
            belief = posts[-1] if posts else h["prior"]
            result.append({
                "id": h["id"],
                "statement": h["statement"],
                "current_status": current,
                "prior": h["prior"],
                "belief": belief,
            })
        return result

    def get(self, h_id: str) -> dict | None:
        """Return the full hypothesis dict, or None if not found."""
        with self._lock:
            data = self._load()
        return data.get(h_id)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load(self) -> dict:
        """Load and validate hypotheses from disk.

        Returns an empty dict if the file is absent.  Validates every
        entry through ``HypothesisEntry.from_dict``; raises
        ``KeyError`` / ``TypeError`` / ``ValueError`` on schema
        mismatch or on-disk corruption — this is intentional (no
        backward compatibility).
        """
        if not self._hypotheses_path.exists():
            return {}
        raw: dict = json.loads(
            self._hypotheses_path.read_text(encoding="utf-8")
        )
        for entry in raw.values():
            HypothesisEntry.from_dict(entry)  # raises on bad schema
        return raw

    def _save(self, data: dict) -> None:
        """Atomically write hypotheses to disk.

        Writes to a sibling temp file then calls ``os.replace`` so
        that a crash mid-write cannot leave a truncated file.
        """
        text = json.dumps(data, indent=2, ensure_ascii=False)
        tmp = self._hypotheses_path.with_suffix(".json.tmp")
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, self._hypotheses_path)
