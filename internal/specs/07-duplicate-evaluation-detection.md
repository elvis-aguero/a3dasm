# Spec 07 — Detect duplicate/redundant design-point evaluations

**Backlog #24.** Priority: medium. Status: **DONE — shipped simplified.**

**What actually shipped (user-simplified from the design below):** counter-
based, not ratio-based. `DUPLICATE_EVALUATION` fires once 3 NEW duplicate rows
land for a delegation since the last check, then resets to 0; capped at 2
nudges per delegation; rate-limited to one per 60s. `Wait()`'s poll loop
(10s tick) also drains the monitor, so the nudge reaches a strategizer
blocked on a live campaign, not just its next tool call. See
`science_monitor.py` `_check_duplicate_evaluations` / `DUP_NUDGE_THRESHOLD` /
`DUP_MAX_NUDGES` / `DUP_COOLDOWN_S`, `instrumented.py`
`duplicate_eval_stats`, `routing.py` `Wait()`. The 20%-ratio-threshold
proposal in the "Design" section below was NOT built — kept for the record
and because the underlying evidence/root-cause section is still accurate.

## Goal
A delegation can burn real, metered eval budget re-evaluating a design point
that is already `FINISHED`, unchanged, in the ledger — with zero new evidence
to show for it. Surface this the same way `UNSTAMPED_ROWS`/`UNLEDGERED_EVALS`
surface their own class of ledger-integrity drift: a warn-only `ScienceMonitor`
rule, informational to the strategizer, never a hard stop.

## Primary evidence — run `example_study/20260713T221841`
- `D003`'s ledger rows: **160 total, 38 unique `(x1, x2)` points** — 122 rows
  (76%) are exact re-evaluations of a point already in the ledger. One point,
  `(-0.714286, -5.0)`, was evaluated **10 times**.
- Root cause, traced through the delegation's own scripts
  (`debug/delegations/D003/`):
  - `test_simple.py` (a plumbing smoke test) samples 30 points via
    `create_sampler("latin_sampler", seed=42)` and calls the real, metered
    `get_evaluator()` — 30 real evals spent just to check the wiring works.
  - `campaign.py` (segfaulted, exit 139) samples the **same `seed=42`** LHS 30
    points and re-evaluates them before crashing mid-BO.
  - `campaign_v2.py` (the successful rewrite) samples the **same `seed=42`**
    LHS 30 points a third time, then runs its own 20 BO iterations with
    per-iteration seeding (`np.random.seed(42 + bo_iter)`) that overlaps
    `campaign.py`'s.
  - None of the three checked the existing ledger before evaluating — despite
    `QueryStore`/`RecallStore` being in the implementer's own declared
    toolset (`agents/implementer.py`).
- Consequence: H1/H2 were registered assuming ~100 designs' worth of
  evidence; only 38 unique points were actually explored, the surrogate's
  R²=0.5545 was fit on an artificially-inflated-but-not-actually-diverse
  dataset, and both hypotheses came back `INCONCLUSIVE` for test inadequacy
  that was largely self-inflicted, not budget-inherent.
- `get_evaluator`'s laziness (`instrumented.py`) only skips rows already
  `FINISHED` *within the `ExperimentData` object a script loads and replays*.
  It does nothing to stop a **fresh** script invocation from constructing new
  `OPEN` samples at coordinates some *earlier* script invocation already
  evaluated — laziness is row-level, not coordinate-level.

## Parsimony check (CLAUDE.md §2)
> "An evaluation that re-derives a design point already `FINISHED`, unchanged,
> in the ledger wastes budget without adding evidence."

This is not a workaround for the segfault-retry pattern above — it applies
identically to a stuck BO loop proposing the same candidate repeatedly
(observed too: `bo_trace.json` iterations 1–4 are bit-for-bit identical), a
second delegation unknowingly re-covering a first delegation's design space,
or any future crash-and-retry script. A philosopher of science would nod at
it as a general resource-integrity principle; it is not overfit to this run.

## Design (DRY — reuse `experiment_stores()`, mirror `_check_unstamped_rows`)

1. **New helper in `instrumented.py`: `duplicate_eval_stats(store_root,
   delegation_id=None)`.** Loops `experiment_stores(store_root)` (same
   namespace-aware primitive `ledger_breakdown`/`delegation_evals` already
   use — no new store-resolution logic). Per store, groups non-provenance
   input columns rounded to a fixed precision (same `.round(10)` convention
   `_ledger_snapshot`/`_reproduction_gate` already use for row-hashing) and
   counts rows per unique coordinate. Returns
   `{"total_rows": N, "unique_points": M, "duplicate_rows": N - M,
   "worst_offenders": [((x1, x2, ...), count), ...]}`, optionally filtered to
   one `delegation_id`'s own rows (mirrors `delegation_evals`'s per-delegation
   framing) or the whole store when `delegation_id` is `None`.
2. **New `ScienceMonitor._check_duplicate_evaluations()`.** Same shape as
   `_check_unstamped_rows`: no-ops when `store_dir is None`; calls the new
   helper per `DONE` delegation from `all_records` (reuse
   `_check_unledgered`'s iteration, don't duplicate it — factor the "for DONE
   record in all_records" loop if this grows a third per-delegation check);
   fires `Violation("DUPLICATE_EVALUATION", "warn", d_id, message)` when that
   delegation's own duplicate-row ratio exceeds a threshold.
3. **Threshold — default 20%, configurable, warn-only.** Deliberately soft
   (§4: eval budget stays a nudge, never a hard-stop, without approval) and
   deliberately conservative — a *legitimate* reason to re-evaluate a point
   exists (a genuinely stochastic oracle where repeat sampling estimates
   noise), so this can never block, only inform. 20% is a starting point, not
   a tuned constant — open question below.
4. **Message content:** name the delegation, the duplicate ratio, and the
   single worst-offending point + its count (`"D003: 76% of its 160 evals
   (122 rows) re-evaluated a point already in the ledger — e.g. (-0.71, -5.0)
   was evaluated 10 times. Check QueryStore for existing rows at a candidate
   before re-evaluating it."`) — actionable, points at the tool that would
   have prevented it, doesn't just report the number.
5. **Not a fix at the `get_evaluator()` boundary.** Blocking or silently
   deduping inside `InstrumentedDataGenerator` would be a bigger, riskier
   change (what counts as "the same point" — exact float equality? some
   epsilon? — and it would need to distinguish "legitimate repeat sample of a
   stochastic oracle" from "redundant re-evaluation of a deterministic one"
   at write time, not report time). A monitor nudge is the smaller, safer,
   fully-reversible first step; a write-time guard is a v2 if the nudge proves
   insufficient.

## TDD plan (tests first; new `test_duplicate_evaluation_detection.py`,
mirroring `test_canonical_store_phase3.py`'s `TestUnledgeredEvalsRule` shape)
1. `test_duplicate_eval_stats_counts_unique_points` — seed a store with 3
   distinct points evaluated 1, 1, and 4 times; `duplicate_eval_stats` reports
   `total_rows=6, unique_points=3, duplicate_rows=3`.
2. `test_duplicate_eval_stats_is_namespace_aware` — seed rows split across the
   default store and a design-namespace sibling; stats aggregate across both
   via `experiment_stores()` (same discipline as the recent `RecallStore`/
   `QueryStore` namespace-blindness fix — don't regress that lesson here).
3. `test_fires_when_duplicate_ratio_exceeds_threshold` — a `DONE` delegation
   whose rows are &gt;20% duplicate → `DUPLICATE_EVALUATION` in
   `mon.evaluate()`'s rule set.
4. `test_does_not_fire_below_threshold` — a delegation with occasional (&lt;20%)
   duplication → no violation (tolerates legitimate stochastic re-sampling).
5. `test_message_names_worst_offender` — the violation message cites the
   single most-duplicated point and its count, not just an aggregate ratio.
6. Regression fixture: replay this run's exact shape (160 rows / 38 unique
   under one delegation ID) and assert it fires — ties the test directly to
   the evidence above, not a synthetic case only.

## Risks / open questions
- **Threshold tuning is a real open question**, not a detail — too low and it
  nudges on legitimate replicate sampling; too high and it misses smaller but
  still-real waste. Proposing 20% as a starting point; revisit after this
  rule has seen a few more real runs (same "adaptive is a v2" posture as
  spec 06's staleness threshold).
- **Exact-match only, not near-duplicate.** Two BO iterations that land
  within floating-point noise of each other but aren't bit-identical won't
  trigger this. Catching near-duplicates needs a distance threshold, which is
  its own tuning problem — deliberately out of scope for v1; the demonstrated
  failure mode (three scripts reusing `seed=42`) is exact-match already.
- **This is informational, not preventive** — by the time the monitor fires,
  the budget is already spent. It won't stop the *next* delegation from doing
  this; it tells the strategizer (and, via the retrospective, future runs'
  prompt-tuning) that it happened. A write-time guard is the preventive
  version, deferred per Design point 5 above.

## KPI "done when"
- Mechanism claim: the six tests above pass; `mon.evaluate()` on the
  `example_study/20260713T221841` ledger snapshot (as a fixture) returns a
  `DUPLICATE_EVALUATION` violation for `D003`.
- Behavioral claim (needs a fresh e2e run to confirm, not claimed by the
  mechanism alone): a subsequent Haiku e2e run against a study whose
  implementer crashes-and-retries shows a strictly lower duplicate-row ratio
  than this run's 76%, because the strategizer (seeing the nudge) redirects
  the next delegation to check the ledger first. Track in `run_ledger.csv`
  — there is no existing column for this, so `evals_used` vs a to-be-added
  `unique_evals`/`duplicate_ratio` column would need to land alongside this
  rule for the KPI to be machine-checkable, not just diffed by hand.
