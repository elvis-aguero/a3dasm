# Spec 06 — Detect a delegation running but making zero ledger progress

**Backlog #6.** Priority: medium. Status: **detection half DONE this session;
push + levers remain.**

## Goal
Distinguish a delegation that is *progressing* from one that is *alive but stuck*
(no new ledger rows), surface it without the delegator having to poll, and give
it a lever other than the blunt `CancelDelegation`.

## Primary evidence — what already landed (commit `46478ee2`)
- `GetStatus` now reports per-delegation ledger progress: `"N evals stamped
  (+k since last poll) — progressing"` vs `"0 evals stamped after Ns"`, tracked
  via `entry["last_stamped"]` / `entry["last_progress_time"]` and
  `_stamped_eval_count` (`routing.py`, the "Still working" block).
- The poll-escalation nudge branches on it: progressing → "do NOT cancel to save
  time"; zero-stamped → cancel framed as reasonable only there.
- Tests: `test_getstatus_reports_ledger_progress_when_evals_stamped`,
  `test_getstatus_flags_zero_progress_as_possible_stuck`.

So the **signal exists** (status Working + climbing wall-time + zero ledger
delta), but it is **pull-only**: the strategizer learns it only when it chooses
to `GetStatus`. And the only response is still `CancelDelegation`.

## Design (DRY — push via the existing notification drain; actuate via spec 02)
1. **Push, don't pull.** `_drain_notifications()` (`strategizer.py:199`) runs at
   the start of *every* strategizer tool call and prepends to the response — it
   already piggybacks ScienceMonitor escalations (`strategizer.py:209-219`). Add
   a cheap scan there: for each `Working` delegation, compute staleness from
   `last_progress_time` + `_stamped_eval_count` (same helpers GetStatus uses) and,
   **once per delegation** (a `stuck_notified` flag, mirroring the
   fire-once pattern), append a notice: *"D004 has run Ns with 0 ledger rows —
   it may be stuck."* No new thread, no polling — reuse the drain that already
   fires. DRY: the staleness computation is the same as GetStatus; factor it into
   one helper both call.
2. **Threshold (open question).** Default: warn once past `max(120s, 15% of the
   time budget)` with zero rows; escalate past `2×` that. Adaptive-to-first-row
   latency is a v2.
3. **The lever is spec 02, not a new cancel.** A stuck notice should point at the
   typed decisions from spec 02 (`grant_budget` if it's close, `abort` if truly
   stuck) — NOT just `CancelDelegation`. Until spec 02 lands, the notice steers to
   the existing cancel (zero-progress is the one case where it's defensible).
4. **Compose with spec 01.** If the delegator aborts a stuck delegation that has
   *some* stamped evals, spec 01 reconciles them — no orphaned work.

## TDD plan (tests first; extend `test_cancel_delegation.py`)
1. `test_drain_emits_stuck_notice_for_zero_progress` — a `Working` delegation with
   0 stamped past the threshold → `_drain_notifications()` returns a stuck notice
   naming it + the wall-time.
2. `test_stuck_notice_fires_once` — the notice is emitted once per delegation, not
   every drain (fire-once flag).
3. `test_no_stuck_notice_when_progressing` — a delegation stamping evals never
   triggers the stuck notice.
4. `test_stuck_helper_shared_with_getstatus` — the staleness/progress computation
   is one helper used by both GetStatus and the drain scan (DRY guard).

## Risks / out of scope
- **Risk:** a worker legitimately spends time *before its first eval* (setup);
  the threshold must tolerate that (hence `max(120s, …)`, and "0 rows" framed as
  "may be" not "is" stuck) — consistent with the GetStatus wording already shipped.
- **Out of scope:** the actuator levers themselves (spec 02); adaptive thresholds.

## Done when
The strategizer is *pushed* a once-per-delegation stuck notice (no poll required)
sharing GetStatus's progress helper, steering to spec 02's levers; tests 1-4
green. KPI test: on a run with a genuinely stuck worker, the strategizer reacts
from the push rather than burning the budget waiting.
