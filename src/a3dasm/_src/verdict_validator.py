"""Pure logic for the #9 live hypothesis-verdict validator.

When the strategizer asserts a CLOSING verdict (SUPPORTED / FALSIFIED /
INCONCLUSIVE) via ``HypothesisUpdate``, an independent referee judges — live,
against the SAME charter the gate critic uses — whether the verdict's SUBSTANCE
obeys the Popperian rules: did the cited work severely test the registered
prediction (§2), does the verdict follow from the result (§3), and (for
FALSIFIED) does it rest on the same registered prediction with no goalpost-move
(§4).

The check is ADVISORY and NON-BLOCKING (the referee is a fallible LLM): its
critique lands in the tool result and persists on the verdict, but the update
always stands. This module holds only the pure, side-effect-free halves — the
prompt builder and the reply parser — so they are unit-testable without a model
or a node. The wiring (invoking the model, persisting the note, escalating) lives
in the HypothesisUpdate closure.
"""
from __future__ import annotations

import re

from a3dasm._src.knowledge.charter import FALSIFICATION_CHARTER

# Statuses that carry an epistemic verdict worth judging. OPEN is mere
# registration — no verdict, nothing to validate.
CLOSING_STATUSES = frozenset({"SUPPORTED", "FALSIFIED", "INCONCLUSIVE"})

_SUBSTANCE_RE = re.compile(r"SUBSTANCE:\s*(OK|FLAG)", re.IGNORECASE)
_CRITIQUE_RE = re.compile(r"CRITIQUE:\s*(.*)", re.IGNORECASE | re.DOTALL)


def build_judge_prompt(
    *,
    statement: str,
    prediction: str,
    criterion: str,
    status: str,
    comment: str,
    evidence: dict | None,
    delegation_report: str | None,
    prior_rulings: str = "",
) -> str:
    """Assemble the one-shot referee prompt for a proposed closing verdict.

    Injects ``FALSIFICATION_CHARTER`` verbatim (the same text the gate critic
    sees) so the live judge and the gate apply one identical standard.

    ``prior_rulings`` (optional) is a digest of this referee's earlier rulings on
    the SAME hypothesis this run — without it the judge is stateless and re-judges
    a borderline case from scratch each call, which lets the verdict oscillate
    (the gate critic already avoids this with its prior-reviews digest). When
    supplied, the judge is told to stay consistent and reverse only on stated new
    grounds.
    """
    report = (delegation_report or "").strip() or "(no delegation report available)"
    prior_block = ""
    if prior_rulings.strip():
        prior_block = (
            "\nYOUR PRIOR RULINGS ON THIS SAME HYPOTHESIS (oldest first)\n"
            f"{prior_rulings.strip()}\n"
            "Apply the charter CONSISTENTLY with the above. If the cited evidence\n"
            "has not materially changed since a prior ruling, your verdict must\n"
            "not change. You may reverse a prior ruling ONLY by explicitly naming\n"
            "the NEW evidence or reasoning that justifies it — a borderline result\n"
            "must not flip the verdict between calls on identical evidence. ONE\n"
            "reversal is ALWAYS legitimate even on identical evidence: downgrading\n"
            "a prior SUPPORTED or FALSIFIED to INCONCLUSIVE because the registered\n"
            "TEST is now recognised as inadequate (§2/§3) corrects a premature\n"
            "close — that is the charter working, NOT a flip-flop, and you must\n"
            "NOT flag it as a consistency violation. What stays forbidden is\n"
            "oscillating between two ADEQUATE readings of the same result.\n"
        )
    return f"""\
You are an independent referee applying the SCIENTIFIC-METHOD CHARTER to ONE
proposed hypothesis-verdict change. Judge ONLY whether the verdict's SUBSTANCE
obeys the charter — not its wording, not the deliverable, not anything else.

<charter>
{FALSIFICATION_CHARTER}
</charter>

THE HYPOTHESIS
  statement: {statement}
  registered prediction: {prediction}
  falsification criterion: {criterion}

THE PROPOSED VERDICT
  status -> {status}
  rationale (comment): {comment}
  cited evidence: {evidence!r}

THE CITED DELEGATION'S RESULT (the work the verdict rests on)
{report}
{prior_block}
Check three things, each by clause number:
  1. §2 ATTEMPT ADEQUACY — did the cited work SEVERELY test the REGISTERED
     prediction (could it have refuted the claim had it been false), or is it a
     token / off-target probe?
  2. §3 VERDICT FOLLOWS — does {status} follow from the cited result?
     (adequate & contradicted -> FALSIFIED; adequate & survived -> SUPPORTED;
      inadequate or confounded -> INCONCLUSIVE.)
  3. §4 NO GOALPOST MOVE — if FALSIFIED, does it rest on the SAME registered
     prediction, not a post-hoc observation?

Reply in EXACTLY this format and nothing else:
SUBSTANCE: OK
— or —
SUBSTANCE: FLAG
CRITIQUE: <one or two sentences naming the clause and the concrete problem>

OK = all three checks pass. FLAG = at least one fails; the CRITIQUE must name the
clause and the specific problem. When genuinely unsure, prefer OK — this is
advisory and the strategizer stays in control."""


def parse_judge_reply(text: str) -> tuple[bool, str]:
    """Parse the referee's reply into ``(flagged, critique)``.

    Lenient by design: an unparseable reply returns ``(False, raw_text)`` — it
    never fabricates a flag (the check is advisory), but keeps the raw text so a
    malformed judgement is still visible in the audit trail rather than silently
    dropped.
    """
    if not text or not text.strip():
        return (False, "")
    m = _SUBSTANCE_RE.search(text)
    if m is None:
        return (False, text.strip())  # unparseable — surface it, don't flag
    if m.group(1).upper() == "OK":
        return (False, "")
    # FLAG: pull the critique if present, else fall back to the whole reply.
    c = _CRITIQUE_RE.search(text)
    critique = (c.group(1).strip() if c else text.strip())
    return (True, critique or "flagged without a stated reason")
