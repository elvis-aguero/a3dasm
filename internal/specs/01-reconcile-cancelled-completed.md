# Spec 01 — Reconcile cancelled-but-completed delegations

**Backlog #1.** Priority: **highest** — recurring root cause of UNGATED /
hypotheses-left-OPEN across runs. Status: spec.

## Problem (primary evidence)
`CancelDelegation` marks a running delegation `Cancelled` but **cannot kill the
daemon thread** (`routing.py:1166` sets `entry["status"]="Cancelled"`; no
`join`/kill — Python can't safely force-kill a thread). The thread keeps running,
finishes, stamps real evals into the canonical ledger, and produces a report.
The run then holds **two contradictory truths**:

- **Registry** (what the strategizer reads): on completion the thread sees
  `status=="Cancelled"` and *skips* updating the result — `result` stays `None`,
  only `evals` is updated (`routing.py:725-734`). So `GetStatus` returns
  `"Errored:\nNone"` (`:986`) — the strategizer never sees the report.
- **delegation_log + disk** (what the critic reads): the completion path
  **unconditionally** calls `delegation_log.record(..., status="DONE",
  deliverable=text, evals=…)` (`routing.py:772-797`) — `status="DONE"` is
  **hard-coded** (`:784`), never conditioned on `_detached`. The critic reads
  `delegation_log.query_all()` (`routing.py:1420-1432`) and its budget sum
  (`:1437-1442`) — so it sees a DONE delegation with a full report and evals.

Run evidence (run `20260615T192313`): D005/D006 cancelled-detached, absent from
the strategizer's view, yet 145/32 ledgered evals + `D006/REPORT_SUMMARY.txt`
("H1 SUPPORTED"). The strategizer says "the falsification was not executed" while
the critic reads the report → non-converging gate loop. Re-confirmed
`20260616T102539` (3D): D006 cancelled at 90%, "lost only the report summary".

This is **inconsistent state, not hallucination**. There is even an
acknowledging comment in `agent_runtime.py` ("captures cancelled-but-completed
delegations whose evals are real").

## Design (DRY — one source of truth; "detach" means "reconcile on completion")
Since the thread cannot be killed, "cancel" must mean **detach-and-reconcile**,
not "pretend it never happened". Make every consumer see the *same* truth, and
let the strategizer *reclaim* the work it paid for.

1. **Truthful delegation_log status.** Parameterize the hard-coded
   `status="DONE"` (`routing.py:784`): when `_detached`, record
   `status="CANCELLED_COMPLETED"` and keep `deliverable=text` (for audit) plus
   `evals`. One write site, one change — no second record path (DRY).
2. **One canonical eval count — from the ledger, never the log sum.** The critic
   budget sum (`routing.py:1437-1442`) currently sums `delegation_log` evals;
   reuse the existing ledger-sourced count (`RunStateSummary` /
   `_stamped_eval_count`, already the run-total source per the Q1 fix) so a
   cancelled-completed delegation's real evals are counted once, consistently,
   regardless of log status. DRY: same helper the run-total and GetStatus use.
3. **Reclaim path — surface the completion to the strategizer.** On
   cancelled-then-completed (`routing.py:748-751`, where the notification is
   already emitted), push a notification via `node._notifications` (drained by
   `_drain_notifications`, `strategizer.py:199`): *"D006, which you cancelled,
   finished: N ledgered evals + report at <path>. Its evidence is valid — you may
   still cite it."* So the strategizer can link the falsification it paid for
   instead of declaring it un-run. Optionally store the report on the registry
   entry so `GetStatus(D006)` returns it (status `CANCELLED_COMPLETED`) rather
   than `Errored: None`.
4. **Critic sees consistent state.** With status `CANCELLED_COMPLETED` in the
   log and the same value in the registry/notification, the critic and
   strategizer never see contradictory delegation state — the loop premise is
   gone.

**Principled framing to carry into prompts:** cancel = "I detach from waiting on
this," not "this never ran." The docstrings (`CancelDelegation`, the poll/Done
nudges) should say a cancelled delegation's *ledgered evals and report are
preserved and reconciled* — which also reinforces spec 06's anti-over-cancel push.

## TDD plan (tests first; extend `tests/agentic/test_cancel_delegation.py`)
1. `test_cancelled_then_completed_logs_truthful_status` — spawn a delegation that
   stamps evals, cancel it, let the thread finish → `delegation_log` entry has
   `status=="CANCELLED_COMPLETED"` (NOT "DONE"), `deliverable` preserved.
2. `test_eval_count_from_ledger_not_log_sum` — a cancelled-completed delegation's
   evals are counted exactly once via the ledger-sourced count (no double-count,
   no drop).
3. `test_strategizer_notified_of_cancelled_completion` — after completion,
   `_drain_notifications()` yields the reclaim notice naming the delegation +
   eval count.
4. `test_getstatus_returns_report_for_cancelled_completed` — `GetStatus` on a
   reconciled delegation returns its report under a `CANCELLED_COMPLETED` token,
   not `Errored: None`.
5. `test_critic_and_registry_agree_on_status` — the value the critic reads
   (`query_all`) and the registry status are consistent for the same delegation.

## Risks / out of scope
- **Race window** between the registry-lock update and the `delegation_log` write
  (separate locks, `routing.py:725` vs `delegation_log.py:93`). Low-probability;
  note it, don't add a two-phase commit unless a test shows it bites.
- **Out of scope:** cooperatively *stopping* the worker before it stamps (needs
  the typed `abort` lever from spec 02 — a worker that checks an abort flag at
  checkpoints). That's the *prevention* half; this spec is the *reconciliation*
  half. They compose.

## Done when
A cancelled-then-completed delegation is recorded truthfully in one status,
counted once from the ledger, surfaced back to the strategizer for reclaim, and
the critic+registry agree — with tests 1-5 green. Re-run the 8d/3D studies and
confirm no "cancelled, could not link falsification" note recurs (the KPI test).
