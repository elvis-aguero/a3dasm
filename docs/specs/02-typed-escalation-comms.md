# Spec 02 — Richer delegator↔worker comms: typed blocker/escalation

**Backlog #2.** Priority: high (supplies the *actuators* spec 01 and 06 need).
Status: spec.

## Problem (primary evidence)
The delegator↔worker protocol is near single-shot and the delegator **cannot
steer** a running worker — it can only abandon it:
- `Delegate(...)` spawns a daemon thread (`routing.py:906-909`).
- Worker → delegator: **one blocking `FollowUp`** per delegation (`routing.py`
  `_run()` ~`:471`): sets `status="FollowUp"`, appends to `_notifications`,
  `evt.wait(timeout=300)`, capped at `followup_count>=1`. `Reply(id, answer)`
  (`routing.py:1179`) sets `followup_answer` + signals the event.
- New this session: `ReportProgress(note)` — non-blocking worker→delegator note.
- The **only** delegator→worker lever is `CancelDelegation` (`routing.py:1120`) —
  binary abandon. There is no *provide input / grant budget / revise scope /
  reroute* action. Evidence this hurts: the over-cancel failure mode (3D D006
  cancelled at 90%) — the delegator had no gentler lever than cancel.

## Principle
A **blocker** differs from a **clarification** by *who must act*: a clarification
needs information *back*; a blocker needs the delegator to take an **action** the
worker can't take itself. So the fix is delegator *levers*, not more worker
questions.

## Design (DRY — reuse the FollowUp blocking plumbing; add a typed reply)
1. **`Escalate(blocker, kind)` worker tool** (`kind ∈ {missing_input,
   over_budget, capability_gap, scope_conflict, unrecoverable}`). It **blocks**
   by reusing the exact FollowUp machinery: set a registry status (`"Escalated"`),
   append a notification naming the blocker+kind, `evt.wait(timeout=…)`. DRY: do
   not invent a second wait/event — generalize the existing
   `followup_event`/`Reply` path to carry a typed payload.
2. **Typed delegator decision** the runtime *applies* (not just text back):
   - `provide(text)` — inject the missing input (== today's `Reply`, reused).
   - `grant_budget(n)` — bump this delegation's eval budget by `n` (the lever the
     over-cancel evidence most needed).
   - `abort` — cooperative clean stop: set an abort flag the worker checks at
     checkpoints, so it stops *before* stamping more / writing a torn report
     (this is the prevention half spec 01 calls out).
   - *(defer, YAGNI until evidence)* `revise_scope(text)`, `reroute(target)`.
3. **Apply + log.** Each decision mutates the right state (budget field; abort
   flag; injected message) and is recorded to `delegation_log` with the
   escalation and its resolution (auditable; fits the science-integrity ethos).
   Reuse `delegation_log.record` (extend status vocabulary).

**YAGNI stance (evidence-driven):** ship `provide` + `grant_budget` + `abort`
first — each has a concrete mechanism and direct evidence of need. Hold
`revise_scope`/`reroute` until a run shows the gap; note the omission in the KB
so it's a known limitation, not a silent cap.

## TDD plan (tests first)
1. `test_escalate_blocks_until_typed_reply` — worker `Escalate` blocks; a typed
   reply releases it; the worker receives the typed payload.
2. `test_grant_budget_raises_delegation_budget` — `grant_budget(n)` increases the
   delegation's eval allowance by exactly `n`; the worker can continue past the
   old cap.
3. `test_abort_sets_cooperative_flag_and_stops_clean` — `abort` sets a flag the
   worker checks; no torn report; reconciles via spec 01's path.
4. `test_provide_injects_input_like_reply` — parity with the existing FollowUp
   answer path (DRY guard: same plumbing).
5. `test_escalation_and_resolution_logged` — both the escalation and the typed
   decision appear in `delegation_log`.
6. `test_escalate_is_bounded` — like FollowUp's ≤1 cap, escalations are bounded
   so a worker can't spin.

## Risks / out of scope
- **Risk:** `grant_budget` interacts with the eval-budget accounting — make sure
  the bumped budget flows to the ledger/`get_evaluator` accounting, not just a
  local counter.
- **Risk:** cooperative `abort` only stops at checkpoints; a worker mid-eval
  finishes that eval (acceptable — it's ledgered).
- **Out of scope:** mid-flight bidirectional streaming chat (the worker is a
  synchronous SDK session; block-and-reply is the tractable model). Full
  `reroute`/`revise_scope` (deferred).

## Done when
A blocked worker can `Escalate(kind)`; the delegator answers with
`provide`/`grant_budget`/`abort`; the runtime applies each; both are logged;
tests 1-6 green. Compose with spec 01 (abort → clean reconcile) and spec 06
(stuck push → grant_budget/abort).
