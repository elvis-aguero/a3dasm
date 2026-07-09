# Agentic audit — rerun batch (90 min watchdog), 2026-06-23

Two reruns at `WATCHDOG_SECONDS = 2 * BUDGET_SECONDS` (90 min) to test the prior
audit's claim that **time** was the binding constraint at 60 min. Both runs
preserved under `_audit_preserved/`. This is a **friction inventory**, not a
feature scorecard — the new-feature checks are §6, last.

## KPI table (4 runs)

| run | watchdog | outcome | gate attempts | evals | notes |
|---|---|---|---|---|---|
| `20260622T211227` | 60 | GATED | 1 | 1152 | clean baseline |
| `20260623T002417` | 60 | FAILED | 1 (REJECT) | 680 | stub notebook, repro failed 6× |
| `20260623T030008` | 60 | watchdog | 3 (→PASS) | 1751 | near-miss: critic PASSED but killed at the close |
| `20260623T102653` | **90** | **GATED** | 3 (→PASS) | 1002 | the near-miss pattern, finished |
| `20260623T113023` | **90** | killed (runaway) | 0 | **~5302** | re-run eval bloat → GP-fit compute explosion |

**60-vs-90 verdict (double-edged, honest):** 90 min carried run `102653`'s
3-attempt gate loop to a clean GATE (vindicating "time was the constraint" for the
*near-miss* case). But run `113023` shows the other edge — more wall-clock gave the
implementer room to re-run real campaigns 14× and balloon the store to 5.3× budget.
**Time was *a* constraint, not *the* constraint.** The deeper constraints below are
what actually cap quality.

---

## Run `113023` — the eval/compute blowup (root-caused)

**NOT a monolithic runaway loop.** `h1_execute.py` plans exactly 1000 evals (300
LHS + 700 BO, 1 eval/iteration) — well-bounded per execution. The blowup is a
chain, all transcript-confirmed:

1. **Non-idempotent campaign re-runs on the real oracle.** The implementer ran
   `python h1_execute.py` **6×** and `python h2_execute.py` **8×** (transcript grep).
   Each `h1` run re-samples + re-evaluates 1000 points *from scratch* and **appends
   to the shared canonical store** — it is not resumable. ~6×1000 ≈ the 5302 evals.
   The implementer used the **budgeted oracle as a debug scratchpad**, re-running on
   each friction (`bool` not JSON-serializable; `output_col` provenance indexing; GP
   `lbfgs failed to converge` warnings in `h1_execution.log`) instead of `RunScratch`.
2. **No cumulative eval-budget guard.** Each script self-budgets in isolation (h1
   targets 1000, h2 ~50); nothing caps the SUM against the 1000 store budget. **No
   `BUDGET_WARN` fired** the entire run.
3. **Downstream surrogate fit detonates on the bloated store.** `h2_execute.py` fits
   `GaussianProcessRegressor(n_restarts_optimizer=5)` on **all** of D001's accumulated
   evals (5302 points) — O(n³)×5, ×8 runs → ~380% CPU (≈4 cores). The compute
   explosion (the user's fans) is a *consequence* of the eval bloat, not independent.
4. **Nothing could stop it in-run.** Budgets are soft (§4); the runtime's time
   backstop only checks *between strategizer turns*, and the strategizer was
   idle-waiting ~60 min on the long delegations — so only the hard wall-clock
   watchdog bounds a worker that re-runs/over-evaluates. (I killed it manually at
   ~5357 evals when it was pegging 4 cores.)

**Classification:** the re-run-on-real-oracle + no-cumulative-cap + GP-on-full-store
chain is a **bug-class** (the system permits unbounded real-oracle spend with no
guard), distinct from #10's "one big campaign." New, highest-priority.

---

## Run `102653` — GATED, but the transcript shows two confabulations the gate missed

1. **Strategizer retrospective confabulation.** DONE retro: *"the lit review
   completed fast (~2 min) and informed subsequent interpretation of D002's
   results."* Delegation log: D001 (lit) ran 10:30:19→**11:03:34** (~33 min) and
   finished **10 min AFTER** D002 (campaign, completed 10:53:06). The retro — §1's
   highest-signal honesty artifact — rationalized away a real decoupling.
2. **False methodology provenance.** Conclusion: *"Kernel Matérn-5/2… informed by
   D001 literature review."* D002 chose+ran Matérn before D001's findings existed;
   the lit happened to agree and was retro-attributed as the cause. **Uncaught by #9
   (hypothesis-scoped) and the critic** (which flagged only a surface RBF-vs-Matérn
   *doc* inconsistency). No guard exists for false methodology provenance.
3. **Eval-budget starvation across hypotheses.** D002 (H1) consumed 950/1000 evals;
   D003 (H2's CMA-ES test) got 52 — far below its own 900-eval criterion → H2
   untestable by construction. (#9 correctly caught the strategizer trying to
   FALSIFY H2 on 52 evals and drove it to INCONCLUSIVE; critic agreed.)
4. **Wasted gate attempt from partial edits.** Critic REVISE'd twice on the kernel
   doc because the strategizer fixed the `doe` cell but not the `hypotheses` cell —
   the kernel name lives in two cells; sync was incomplete (call_002 burned).
5. **Lit-reviewer retro mislabeled.** D001 (literature) *did* write a substantive
   retro this run (the long-standing gap is improving) — but it's tagged
   `role="implementer"`, which misleads any role-filtered §1 analysis.

---

## Cross-cutting (both runs)

- **`output_col` provenance trap bites WORKERS, not just the notebook.** D002's
  retro FRICTION (run `102653`): underscore-prefixed provenance columns sort first,
  "required careful indexing." The earlier fix (`b3f37e4a`) patched only the
  *notebook deliverable* spec — implementer campaign code hits the identical trap.
  The fix was too narrow.
- **#14 reap misses campaigns by NAME, not just by session.** My own orphan sweeps
  grepped `"campaign"`, but the implementer names scripts `h1_execute.py` /
  `h2_execute.py` — so leaked campaigns from prior runs accumulated and (in run
  `113023`) pegged 4 cores. Parent was a `.claude` shell snapshot → confirms the
  implementer launches campaigns in a **detached shell** (escapes run.py's group).
  Reinforces #14's fix: a **recursive PPID tree-walk** at reap catches them
  regardless of script name; a name-based grep never will.

## §6 — new-feature verification (the small corner)

- **#9 verdict validator:** in `102653`, fired once, correctly (caught H2
  FALSIFIED-on-52-evals as §2–§3 inadequate → INCONCLUSIVE; critic agreed; no
  whipsaw). Confirms it's a useful live nudge. (Run `030008` showed its limits:
  whipsaw + non-determinism — see prior audit.)
- **#13 RunPipelineCell:** used in `102653` ("clear diagnostics, debugging was
  fast" per the strategizer); notebook passed repro.
- **#14 reap:** clean closes (`102653`) leak nothing; the bug is watchdog-kill +
  detached + name-based-sweep specific (above).
- **D002 contiguous-ID (D002 fix):** held in both reruns.

---

## Recommendations (prioritized)

1. **Bound real-oracle spend (NEW, highest).** The implementer must not re-run full
   real campaigns to debug. Options: (a) campaigns resume from the store (skip
   FINISHED rows) instead of re-sampling; (b) a cumulative store-eval cap that
   errors/warns when exceeded (even if soft, fire `BUDGET_WARN`); (c) steer debugging
   to `RunScratch`/a cheap stub, not the metered oracle. This is what actually blew
   the budget 5×, not #10.
2. **Cap surrogate training-set size.** A GP fit on the full store is O(n³); bound it
   (subsample / cap n) so a bloated ledger can't detonate compute.
3. **#14 reap fix** — recursive PPID tree-walk (catches `*_execute.py` regardless of
   name/session).
4. **Guard methodology provenance + retrospective honesty** — nothing catches
   "informed by lit" when the lit post-dated the choice, or a retro claim that
   contradicts the delegation timestamps. A cheap cross-check (delegation
   completed_at vs the claim) could flag both.
5. **Notebook↔ledger multi-cell sync** — the kernel name in two cells caused a wasted
   gate attempt.
6. **Lit retro role label** — tag it `literature_reviewer`, not `implementer`.

## Artifacts
Preserved: `_audit_preserved/{20260623T102653 (GATED), 20260623T113023 (runaway,
killed ~5357 evals)}/` — each with `run_dir/debug/` + the stamped notebook.
Run `113023` was killed mid-runaway (no ledger row / clean close).
