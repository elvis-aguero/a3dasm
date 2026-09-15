"""The run's terminal state: one structured value, rendered into prose.

Never the other way round. Before this module the run's outcome was recovered
by grepping the final report for banner strings (``"⚠ UNGATED RUN"``, ``⛔``,
``"NOT validated"``) and defaulting to GATED when none matched. Two live bugs
came straight out of that:

1. ``LifecycleMixin._halt_resumable`` wrote ``status="halted"``, then the close
   path overwrote it with the grep result. The HALTED banner matches none of
   the patterns, so a run killed by the time backstop, the USD ceiling, or 12
   consecutive delegation errors was recorded as a **validated success**.
2. With no critic in the graph, ``_close`` routes straight to done and no
   banner is ever emitted — so a critic-less run read GATED. Nothing reviewed
   it; there was no gate to pass.

The fix is to decide the outcome where it is known and carry it as data. Every
terminal path records ``outcome`` / ``termination`` / ``reviewed`` on the
route; the node copies them onto the state; ``AgenticRun`` persists them. The
banner is rendered FROM that value, so the prose can never disagree with the
record.

``reviewed`` is kept distinct from ``outcome`` on purpose: "the critic passed
it" and "no critic ever looked" are different facts, and an ablation that
removes the critic needs to tell them apart.
"""

from __future__ import annotations

__all__ = [
    "GATED",
    "UNGATED",
    "FAILED",
    "OUTCOMES",
    "DONE",
    "NO_CLOSE",
    "BACKSTOP_TIME",
    "BACKSTOP_USD",
    "REPEATED_ERRORS",
    "RECURSION_LIMIT",
    "CRASHED",
    "KILLED",
    "TERMINATIONS",
    "HALT_TERMINATIONS",
    "ungated_banner",
    "resolve",
]

# --- outcome: what the run's conclusions are worth -------------------------
GATED = "GATED"      # a critic gate ran and passed it
UNGATED = "UNGATED"  # on record, not validated
FAILED = "FAILED"    # the deliverable never reproduced

OUTCOMES = (GATED, UNGATED, FAILED)

# --- termination: how the run stopped --------------------------------------
# Orthogonal to outcome. A run can terminate `done` and still be UNGATED (no
# critic), or terminate `backstop_time` with real conclusions on record.
DONE = "done"                        # the agent closed deliberately
NO_CLOSE = "no_close"                # ended without an accepted Done()
BACKSTOP_TIME = "backstop_time"      # run_backstop_multiple x wall budget
BACKSTOP_USD = "backstop_usd"        # hard USD ceiling
REPEATED_ERRORS = "repeated_errors"  # max_consecutive_errors to one target
RECURSION_LIMIT = "recursion_limit"  # LangGraph step ceiling
CRASHED = "crashed"                  # unhandled exception
KILLED = "killed"                    # external supervisor (wall-clock watchdog)

TERMINATIONS = (
    DONE, NO_CLOSE, BACKSTOP_TIME, BACKSTOP_USD,
    REPEATED_ERRORS, RECURSION_LIMIT, CRASHED, KILLED,
)

# Terminations that mean the run was stopped rather than finished. None of
# these can be GATED: nothing reviewed the conclusions, because the run never
# reached its gate. Analysis MUST treat these as censored, not as failures —
# a killed run may have been minutes from a PASS (see internal/AUDIT-20260623).
HALT_TERMINATIONS = (
    BACKSTOP_TIME, BACKSTOP_USD, REPEATED_ERRORS,
    RECURSION_LIMIT, CRASHED, KILLED,
)


def ungated_banner(flags: list[str]) -> str:
    """The UNGATED banner, rendered from the reasons the run is unvalidated.

    Single source of the wording. ``flags`` are clauses naming each reason.
    """
    return (
        "## ⚠ UNGATED RUN\n\n"
        "This run is NOT validated: " + "; ".join(flags) +
        ".\nTreat all conclusions below as unaudited.\n\n---\n\n"
    )


def resolve(
    outcome: str | None,
    termination: str | None,
    reviewed: bool | None,
) -> tuple[str, str, bool]:
    """Fill in a terminal triple, failing SAFE when a path did not record one.

    A path that terminates without declaring an outcome did not validate
    anything, so the unrecorded case resolves to UNGATED — never to GATED.
    That direction matters: the old default was GATED, which is why a halt
    read as a success.

    GATED is additionally refused for any halt termination and for a run no
    critic reviewed, so the two can never disagree.
    """
    term = termination if termination in TERMINATIONS else NO_CLOSE
    rev = bool(reviewed)
    out = outcome if outcome in OUTCOMES else UNGATED

    if out == GATED and (term in HALT_TERMINATIONS or not rev):
        out = UNGATED
    return out, term, rev
