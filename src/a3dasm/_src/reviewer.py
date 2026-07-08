"""Pre-run problem-statement reviewer (Item B).

A *general, parsimonious* check that a problem statement is well-posed BEFORE
an autonomous run begins.  It judges the statement against the five universal
elements of a fundable problem statement — and nothing benchmark-specific, so
the rubric does not overfit to the handful of studies we happen to have.

The reviewer is **advisory**: it always writes a report and never blocks an
autonomous run.  When the run is interactive and gaps are found, it offers an
optional refine step (the same ``input()`` channel the in-graph ``FollowUp``
uses), appending accepted human clarifications to the statement.

This module is pure (prompt + parsing + formatting); the orchestration lives
in :meth:`AgenticRun._review_problem_statement`.
"""
from __future__ import annotations

import json
import re

from .backends.base import Agent

# The FIVE universal elements of a well-posed problem statement.  Deliberately
# domain-agnostic — these generalise across optimisation, inverse-design,
# discovery, and characterisation studies alike.
REVIEW_ELEMENTS: list[tuple[str, str]] = [
    (
        "objective",
        "A single, clear, measurable objective / success criterion that can "
        "be ledgered — what counts as done, and how 'better' is judged.",
    ),
    (
        "design_space",
        "A defined design space: the decision variables, their ranges, and "
        "their types / units.",
    ),
    (
        "ground_truth",
        "How ground truth is obtained: the oracle / evaluator that turns a "
        "candidate design into an outcome.",
    ),
    (
        "validity",
        "What makes a result valid / feasible: the validity gates or regime "
        "of applicability that separate a real result from an artefact.",
    ),
    (
        "deliverable",
        "The concrete deliverable the run must produce.",
    ),
]


def _build_system_prompt() -> str:
    elems = "\n".join(
        f"  {i}. {key} — {desc}"
        for i, (key, desc) in enumerate(REVIEW_ELEMENTS, start=1)
    )
    keys = ", ".join(f'"{k}"' for k, _ in REVIEW_ELEMENTS)
    return (
        "You are reviewing a scientific / engineering PROBLEM STATEMENT just "
        "before an autonomous research run begins. Your job is to judge "
        "whether the statement is WELL-POSED — not to solve it, not to add "
        "domain requirements.\n\n"
        "Judge it ONLY against these five universal elements of a well-posed "
        "problem statement:\n"
        f"{elems}\n\n"
        "Be GENERAL and PARSIMONIOUS. Do NOT invent domain- or "
        "benchmark-specific requirements; an element is a 'gap' only if a "
        "competent researcher could not begin the work without first asking "
        "for it. If the statement reasonably implies an element, mark it "
        "'ok'.\n\n"
        "Return STRICT JSON ONLY (no prose, no code fence is required but is "
        "tolerated), of the form:\n"
        '{"elements": [{"element": <one of '
        f"{keys}"
        '>, "status": "ok" | "gap", "note": "<short, specific>"}], '
        '"summary": "<one or two sentences>"}\n'
        "Include exactly one entry per element."
    )


REVIEWER_SYSTEM_PROMPT = _build_system_prompt()


class ProblemStatementReviewerAgent(Agent):
    """Ephemeral, tool-less reviewer agent (one-shot strategizer-model call)."""

    role = "reviewer"
    tools = frozenset()
    reset_on_checkpoint = True
    system_prompt = REVIEWER_SYSTEM_PROMPT
    description = "Pre-run problem-statement well-posedness reviewer."


# --- parsing / formatting (pure) --------------------------------------------

def parse_review(text: str) -> dict:
    """Extract the review JSON from an LLM reply.

    Robust to surrounding prose and ```json code fences.  On any failure it
    returns an empty-elements review (so the caller manufactures NO gaps and
    the run proceeds — the reviewer is advisory, never a blocker).
    """
    if not isinstance(text, str):
        return {"elements": [], "summary": "", "parse_error": True}
    # strip code fences if present
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    candidate = fenced.group(1) if fenced else None
    if candidate is None:
        # first balanced-ish {...} blob
        m = re.search(r"\{.*\}", text, re.DOTALL)
        candidate = m.group(0) if m else None
    if candidate is None:
        return {"elements": [], "summary": text.strip()[:200],
                "parse_error": True}
    try:
        obj = json.loads(candidate)
    except (ValueError, json.JSONDecodeError):
        return {"elements": [], "summary": text.strip()[:200],
                "parse_error": True}
    elements = obj.get("elements") if isinstance(obj, dict) else None
    if not isinstance(elements, list):
        elements = []
    return {
        "elements": elements,
        "summary": (obj.get("summary") if isinstance(obj, dict) else "") or "",
    }


def review_gaps(review: dict) -> list[dict]:
    """The elements flagged as gaps (status == 'gap')."""
    out = []
    for e in review.get("elements", []):
        if isinstance(e, dict) and str(e.get("status", "")).lower() == "gap":
            out.append({
                "element": str(e.get("element", "?")),
                "note": str(e.get("note", "")),
            })
    return out


def format_review_markdown(review: dict, problem: str) -> str:
    lines = ["# Problem-statement review\n"]
    summary = review.get("summary")
    if summary:
        lines.append(f"{summary}\n")
    if review.get("parse_error"):
        lines.append(
            "_(The reviewer reply could not be parsed as JSON; treated as "
            "advisory with no actionable gaps.)_\n"
        )
    lines.append("| element | status | note |")
    lines.append("|---------|--------|------|")
    by_key = {
        str(e.get("element")): e
        for e in review.get("elements", [])
        if isinstance(e, dict)
    }
    for key, _desc in REVIEW_ELEMENTS:
        e = by_key.get(key, {})
        status = str(e.get("status", "—"))
        note = str(e.get("note", "")).replace("|", "\\|")
        lines.append(f"| {key} | {status} | {note} |")
    return "\n".join(lines) + "\n"
