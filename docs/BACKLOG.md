# Agentic system — backlog

Deferred ideas with enough context to resume cold. Not committed work; pick up
when prioritized. Ordered by priority (highest first); item numbers are stable
references, not a queue position — see the order above.

> **Detailed, evidence-grounded specs** for every item live in
> [`specs/`](specs/README.md) (primary-evidence citations, TDD test names, DRY
> reuse, KPI done-when). The entries below are the short version.

## Status checklist
Resolved items keep their write-up below for the record; `(commit)` is what fixed them.

- [ ] **#1** Reconcile cancelled-but-completed delegations — *open, highest priority* (recurring UNGATED root cause; partially mitigated 2026-06-15)
- [ ] **#2** Richer delegator↔worker comms (typed blocker/escalation) — *deferred*
- [ ] **#3** ProblemDefinerAgent pre-strategizer intake stage — *deferred*
- [x] **#4** Single Jupyter-notebook deliverable — *DONE* (the live system; gate runs the notebook via nbclient; toolset completed `db288d5b`)
- [x] **#5** KB entry: running a study on SLURM — *DONE* (`entries/0010-running-a-study-on-slurm.md`; the deliverable pipeline runs on SLURM, the agent graph stays local)
- [ ] **#6** Detect a delegation running but making zero ledger progress — *open*
- [ ] **#7** Remove the literature hand-listed tool docs (BF-13a) — *open* (needs corpus closures to carry docstrings first)
- [x] **#8** Literature str/int error (lit-bug #3) — `64a8230a` (coerce arxiv `max_results` to int)
- [x] **#9** Orchestrator-owned live validator for HypothesisUpdate — *DONE* (advise-with-teeth, charter-grounded, kill-switchable via `F3DASM_VERDICT_VALIDATOR`; merged into dev/confer; validated live run `20260623T002417` — fired, judged H1 correctly vs the gate critic, non-blocking)
- [ ] **#10** Strategizer delegates the optimization as one monolithic un-budgeted campaign — *open, §4 user-owned* (**current binding constraint** — watchdog-kills runs)
- [x] **#11** Orphaned background process survives watchdog kill — `5199b593` (process-group reap)
- [x] **#12** Watchdog kill loses the strategizer's retrospective — `5199b593` (synthetic post-mortem entry)
- [x] **#13** Per-cell notebook debugger — *DONE* (`RunPipelineCell` closure + `diagnose_notebook`; per-cell trace localizes a repro failure by cell name + traceback; runs against a ledger copy; kill-switch-free read-only diagnostic)
- [ ] **#14** Watchdog reap (#11) MISSES detached campaign processes — *open, HIGH (real CPU leak)* — they escape the process-group kill (new session) and outlive the run
- [ ] **#16** ABAQUS subprocess can't import workspace modules (PYTHONPATH) — *open, abaqus2py-owned* (recommendation only; not an f3dasm fix)
- [ ] **#17** Closure + budget-severity model (Memory>Time>Eval; dynamic constraints) — *open, §4 user-owned* — Done() prompt iterated (`0400a653`); runtime nudge + severity model deferred
- [ ] **#18** Critic should flag an infeasible-extremum headline on a constrained study — *open, §4 user-owned* — grounding moved to the critic (`6494489b`) but it only checks the value is real, not feasible
- [ ] **#20** Open design-space discovery (agent invents new low-D parametrizations) — *spec approved, §4 user-owned, awaiting 2D experiment* — see [`OPEN_DESIGN_SPACE_FRAMEWORK.md`](OPEN_DESIGN_SPACE_FRAMEWORK.md); branch `exp/open-design-space`
- [ ] **#23** Rename `literature_reviewer` → `consultant` + give it live-web tools so it answers tech-stack/API/doc questions, not only academic literature — *spec, not built (user decision 2026-06-30)* — see §23 below
- [ ] **#21** RecallStore is namespace-blind — *open, low* (branch `exp/open-design-space`). The strategizer's `RecallStore()` ledger SUMMARY (`routing.py` `_derive_store_dir`/`RecallStore`) reports only the canonical store, so a multi-namespace run's namespace evals don't appear in that view. Eval COUNTING is namespace-aware everywhere (commits cea08d8b→d27a33ae); this is the remaining SUMMARY surface. Aggregating a full `RunStateSummary` (per-delegation/per-source dicts + output stats) across stores is a larger change than the count helper — deferred. The strategizer still sees namespace progress via per-delegation reports, so this is informational, not a metering gap.

---

## 1. Reconcile cancelled-but-completed delegations
**Status:** partially mitigated 2026-06-15 (cancel hardened + eval-count now
from ledger); the core inconsistency remains. **Highest priority** — it is the
recurring root cause of UNGATED / hypotheses-left-OPEN outcomes across runs.

`CancelDelegation` detaches a worker and tells the strategizer "result will be
ignored," but the **detached worker keeps running**, finishes, stamps real evals
into the canonical ledger, and writes its report to disk. The run then holds two
contradictory truths: the official record (delegation_log / registry / what the
strategizer is told) says *ignored / not executed*, while the ledger + on-disk
report say *done*. Primary evidence (run 20260615T192313): D005/D006 were
cancelled-detached, absent from delegation_log, yet have 145/32 ledgered evals +
`D006/REPORT_SUMMARY.txt` ("H1 SUPPORTED"). The strategizer then claimed "the
falsification was not executed" (per the runtime's "ignore it") while the critic
read the disk report — a **non-converging gate loop** rooted in inconsistent
state, NOT agent hallucination or prompt friction. Re-confirmed 2026-06-16: the
8d e2e left H1/H2 OPEN citing "D004 cancelled post-completion."

**Done so far:** (a) `evals_used` now counts from the ledger so those evals
aren't dropped from the run total (`agent_runtime.py`); (b) cancel is hardened
against impatience — two-shot for delegations already producing ledgered evals;
docstring + poll/premature nudges reframed (`routing.py`).

**Still to design:** when a detached worker completes with stamped ledger rows,
**reconcile** it — record its completion in the delegation_log (so the
strategizer sees it finished, not "ignored"), or stop the worker BEFORE it
stamps. Pick one source of truth so strategizer and critic never see
contradictory delegation state. Related: the stuck-delegation detection in #6.

---

## 2. Richer delegator↔worker comms — typed blocker/escalation
**Status:** deferred (prefer benchmarking the current system first). Design explored 2026-06-15.

**Today:** the protocol is near single-shot — `Delegate(task, expected_report, …)`
down, the worker's structured report up, and exactly **one blocking
`FollowUp(question)` clarification** mid-task (≤1 per delegation; routes to the
delegating agent — or the human for the entry node). When no operator/TTY is
present FollowUp now returns an autonomous-proceed notice instead of blocking
(headless `input()` EOFError fixed 2026-06-16). The delegator can only
`CancelDelegation` a running worker — it **cannot steer** one.

**Principle:** a *blocker* differs from a *clarification* on **who must act** —
a clarification needs information back; a blocker needs the delegator to take an
**action** the worker can't take itself. So the fix is not "let the worker ask
more," it's "give the delegator the right levers." The real levers a blocked
worker needs: **provide** a missing input/capability, **grant** more eval
budget, **revise** scope, **reroute** to another specialist, **abort** cleanly.

**Chosen direction (Approach A + C's logging):** a worker tool `Escalate(blocker,
kind)` distinct from `FollowUp` — `kind ∈ {missing_input, over_budget,
capability_gap, scope_conflict, unrecoverable}` — that *blocks* (reuse the
FollowUp wait/Reply plumbing) and the delegator answers with a **typed
decision** (`provide / grant_budget(n) / revise_scope / reroute(target) /
abort`) that the runtime *applies* (budget bump, clean abort, re-delegation).
Log the escalation + its resolution to the delegation log (auditable, fits the
science-integrity ethos). These same `abort`/`grant_budget`/`reroute` levers are
what the strategizer needs to act on the stuck-delegation signal in #6.

**Open question to settle first:** which of the 5 delegator actions earn their
place vs YAGNI?

---

## 3. ProblemDefinerAgent — a pre-strategizer intake stage
**Status:** deferred. Raised 2026-06-16.

A new agent that sits **between the human and the strategizer**, running once at
the very start of a run, before the strategizer takes over. Its job is to turn a
raw human problem statement into a high-signal, airtight brief so the strategizer
spends its budget on science, not on plumbing/ambiguity.

**Responsibilities:**
- **(a) Airtight problem statement** — resolve ambiguity, pin the objective
  (minimise/maximise), constraints, success criterion, and what "the result"
  is. Today this is partly covered by the advisory `_review_problem_statement`
  pre-run pass (`agent_runtime.py:680`); the ProblemDefiner would *own* and
  extend it (interactive with the human, not just advisory).
- **(b) Tech stack + ExperimentData schema** — decide/confirm the f3dasm Domain
  (input variables + bounds + types) and the **output columns** of the canonical
  ExperimentData (objective col name, feasibility cols, units), so the ledger
  schema is fixed before any delegation runs.
- **(c) Hard-to-automate plumbing** — the evaluator entrypoint, eval/wall
  budgets, output_names, any study-specific config that today lives in
  `config.yaml` / `PROBLEM_STATEMENT.md` and is easy to get subtly wrong.

**Why:** it **offloads the strategizer** (which currently has to infer schema,
reconcile config vs problem statement, and self-review well-posedness) and hands
it a higher-quality signal. Net effect: fewer SCIENCE_DRIFT / MILESTONE_BLOCK
diagnostics traceable to an under-specified brief, and a fixed ledger schema from
turn one.

**To design when picked up:** is it a graph node (entry before strategizer) or a
runtime pre-pass like the current problem-statement review? How interactive with
the human (blocking Q&A vs one-shot)? Does it *write* `config.yaml` + an enriched
`PROBLEM_STATEMENT.md` as its output artifacts (so the brief is itself a
reproducible deliverable)? Relationship to the existing
`_review_problem_statement` advisory pass (replace vs wrap).

---

## 4. Single Jupyter-notebook deliverable
**Status:** RESOLVED — this IS the live system: `pipeline.ipynb` is the sole deliverable,
the gate executes it via nbclient (`notebook_exec.run_deliverable` / `_reproduction_gate`)
with the zero-new-evals + `REPRODUCED:` asserts, no `solution.md`/`pipeline.py`, and the
agent authors the cells through the structured tools (the CRUD set completed in `db288d5b`).
Every implication below was implemented. Original write-up kept for the record.
**Status (orig):** deferred. Raised 2026-06-15.

Merge the two deliverables (`solution.md` prose + `pipeline.py` executable) into
**one `.ipynb`** — markdown cells for the writeup, code cells for the lazy
create→run→analyze pipeline. One human-readable, runnable artifact.

**Implications to design when picked up:**
- The reproduction gate (`StrategizerNode._reproduction_gate`) would execute the
  *notebook* lazily (e.g. `jupyter nbconvert --execute` / `nbclient`) instead of
  `python pipeline.py`, keeping the same asserts: exit clean + **zero new oracle
  evals** + headline self-assert (now a pre-critic gate, timeout = max(10% of
  the time budget, 180s)).
- `WriteDeliverable` accepts `.ipynb`; `_missing_deliverables` requires it.
- The runtime currently auto-writes `solution.md` from the `Done()` summary —
  that prose would instead become the notebook's leading markdown cells (decide
  who authors it: agent vs runtime injection).
- Keep the lazy + cache-or-load contract intact (notebook re-run = reproduction,
  no sims, no heavy refit).

---

## 5. KB entry: running a study on SLURM
**Status:** RESOLVED 2026-06-23 — `entries/0010-running-a-study-on-slurm.md`.
Raised 2026-06-15.

KB entry added: `Pipeline.run(mode="slurm", cluster=SlurmCluster(...))`, per-step
`SlurmResources`, `parallel=True` → `cluster_array` striping, canonical-store
FileLock making concurrent array writes + lazy FINISHED-skip resume safe, and the
key boundary — the deliverable pipeline runs on SLURM, the agent graph stays
LOCAL (no slurm path in agent_runtime/run.py). Lightweight by request; deeper
how-to docs deferred (none requested).

---

## 6. Detect a delegation that is running but making zero ledger progress
**Status:** raised 2026-06-16. Cross-links #1 (reconcile) and #2 (steering levers).

A delegation can be **alive but unproductive**: the worker is still `Working`,
wall-time is climbing, but it is stamping **zero new rows** into the canonical
ledger. Today the strategizer cannot tell "slow but progressing" from "stuck /
spinning," so it polls, then either waits until the time budget dies or cancels
on impatience (feeding #1). Primary evidence (run 20260616T004655): the
implementer (D004) was alive ~228s+ yet produced **0 ledgered evals**; the run
ended UNGATED with an empty ledger.

**The signal:** status == Working AND wall-time since last ledger row > threshold
AND ledger delta == 0. That is detectable from the same `RunStateSummary` /
delegation timing the runtime already tracks. Surface it to the strategizer as a
distinct notice ("D004 has run 200s with no ledger progress") rather than letting
it guess from poll counts.

**Then it needs a lever, not just a notice** — which is exactly the typed
delegator decisions in #2 (`grant_budget` if it's genuinely close, `reroute` or
`abort` if it's stuck). Without #2 the only response is still the blunt
`CancelDelegation`. So #6 is the *detector*; #2 supplies the *actuators*; #1
ensures whatever the worker already stamped is reconciled rather than orphaned.

**Open question:** the threshold — fixed seconds, a fraction of the time budget,
or adaptive to the worker's own first-row latency? A cheap robust default: warn
once past max(120s, 15% of budget) with no row, escalate past 2× that.

---

## 7. Remove the literature hand-listed tool docs (BF-13a)
**Status:** raised 2026-06-21. Follow-through on BF-13(b) (commit 6b9c8dd8).

`literature.py` still hand-lists every tool in `<corpus_tools>`/`<discovery_tools>`
with bare signatures; the implementer's `<role>` "Available tools" list does the
same. BF-13(b) made the auto-generated `<tools>` catalog the authoritative,
MCP-qualified source, so these hand-lists are now a second source that can drift
from it.

**Why not done yet:** the corpus closures (`CorpusAdd`/`CorpusSearch`/
`CorpusGetPaper`/`CorpusList`) are docstring-less lambdas in
`build_closure_tools`, and `render_tool_catalog` falls back to "(no description)"
for them. Deleting the hand-list before the closures carry real docstrings would
DEGRADE the catalog (lose the BM25-weighting note, the "ERROR if no full-text"
semantics, the acquisition workflow). The implementer list also mixes native
tools (Read/Bash — bare-correct) with closures (qualified) — a blanket delete
would lose the native-tool descriptions too.

**The principled fix:** give each corpus/discovery closure a real docstring (one
line is enough — the SDK already reads `fn.__doc__` for its tool schema), then
delete the hand-lists and let the qualified auto-catalog be the single source.
Test: assert the literature catalog (qualified) carries a non-"(no description)"
entry for every corpus tool, and that no bare hand-list signature survives.

## 8. Localize literature str/int error (lit-bug #3)
**Status:** RESOLVED 2026-06-22 (commit 64a8230a) — localized via the
string-`max_results` repro in run 20260622T165943; `arxiv_search_papers` /
`arxiv_list_papers` now coerce `max_results` to int. Original write-up below.
**Status (orig):** raised 2026-06-21. Observed in a wet literature run (prior session),
exact site not captured.

A wet literature delegation hit a `str`/`int` type error twice ("str-int error
×2"), separate from the dense-ranking crash (fixed 33d8d3f4) and the name
mismatch (fixed 6b9c8dd8). No traceback was captured this session — the wet test
was killed to save resources before it streamed output.

**Likely neighbourhood:** the citation-count / BM25 weighting path
(`log10(c+1)`), where a `citationCount` arriving as a string from S2/OpenAlex
JSON would break arithmetic; `build_closure_tools` already guards
`int(citation_count or 0)` on `CorpusAdd`, so the unguarded site is probably in
`CorpusRank` or the corpus's internal ranking, or in an OpenAlex/S2 field read.
**Needs a real wet run with the traceback** (`uv run pytest
tests/agentic/test_literature_wet.py -s --no-cov`) to localize before fixing —
do not guess-patch without the stack.

## 9. Orchestrator-owned live validator for HypothesisUpdate
**Status:** IMPLEMENTED (advisory) 2026-06-22 — the #9 live verdict validator
(`verdict_validator.py` + `nodes/critic_gate.py` `_run_verdict_validator`)
validates each closing verdict's substance against the charter at the
HypothesisUpdate boundary, ADVISES (never blocks), escalates on repeat flags.
Runs on the **critic's** model (one refereeing standard, decoupled from the
strategizer it judges). Given prior-rulings **MEMORY** 2026-06-23 (commit
88856cb9) so a borderline verdict can't oscillate between calls — see #15.
**§4 — user owns it.** The DEEPER architectural separation below (dedicated
verdict-adjudicator node / freeze the criterion at proposal time / binding
`LinkFalsificationAttempt`) remains OPEN — the advisory validator is one
realization, not the full separation.

Today the strategizer carries a triple burden for every hypothesis: it (1) states
the hypothesis + falsification criterion, (2) frames the falsification attempt,
and (3) judges the attempt's results — proposer, experiment-designer, and judge in
one agent. The only live guard on `HypothesisUpdate` is RULE-based (the Popperian
charter's `ERROR_RETURN`: cannot mark SUPPORTED without a falsification attempt on
record; cite only completed delegations). It checks *form*, not *substance*.

**Idea:** an **orchestrator-owned LLM** that validates each `HypothesisUpdate`
call LIVE at the tool boundary (where `ERROR_RETURN` fires now), independent of the
strategizer's own reasoning — offloading the judging role. It would check the
substance a rule cannot:
- the falsification attempt actually PROBES the registered prediction (severity —
  could it have refuted?);
- the verdict (SUPPORTED / FALSIFIED / INCONCLUSIVE) FOLLOWS from the cited
  evidence/ledger numbers;
- prediction and falsification_criterion test the SAME claim (catches goalpost-moving).

**Evidence motivating it (n=5 audit, `_audit_preserved/n5_20260621_222224`):**
- run03 strategizer moved its own goalposts — registered prediction ("reach f ≤ −1.0")
  ≠ falsification_criterion (relative budget test); the critic flagged it Charter §4.
  The strategizer self-reported the CONSISTENCY contradiction in its retrospective.
- `LinkFalsificationAttempt` was used only 2× across 6 runs — falsification linkage
  is effectively advisory/bypassed (a run02 critic: "the flag is advisory, not binding").
- verdicts are non-reproducible: the same H2 question (multi-start vs BO) came back
  FALSIFIED (r1, r2) and SUPPORTED (r4, r5) depending on the run's self-chosen budget.

**Open design questions (resume-cold):**
- BLOCK vs ADVISE: does it reject the update (hard, like `ERROR_RETURN`) or annotate
  it for the critic (soft)? Budgets are soft elsewhere — lean advise-then-escalate.
- Relation to the critic: the critic judges the DELIVERABLE at gate-time; this judges
  HYPOTHESIS verdicts LIVE as asserted — complementary, not a replacement.
- Cost/latency: one extra LLM call per HypothesisUpdate; pick a model tier; cache.
- Where it lives: `science_monitor.py` / the HypothesisUpdate tool wrapper — the
  Popperian-rules surface the user owns. Do NOT implement without the user's design call.
- Relates to the deferred "separate the falsification-judge from the hypothesis-author"
  options (freeze criterion at proposal time; dedicated verdict-adjudicator node;
  binding LinkFalsificationAttempt + INCONCLUSIVE as valid closure).

**Done-when (KPI):** verdict reproducibility across same-problem runs improves;
goalpost-moves caught LIVE (not only post-hoc by the critic); the falsification-attempt
linkage becomes load-bearing rather than advisory.

## 10. Strategizer delegates the optimization as ONE monolithic, un-budgeted campaign
**Status:** PARTIALLY MITIGATED 2026-06-22 (commit pending) — added a "scope each
delegation to one hypothesis" principle to the strategizer planning prompt
(`<scientific_process>`), grounded in Charter §3 (bundling confounds the test) with
a fail-fast/self-correct rationale. This NUDGES toward targeted campaigns but is not
enforced — the strategizer already ignored a sharper signal (H1's explicit 300-eval
criterion), so it likely needs the #9-family check to truly bind, and the wall-clock/
algorithm half (uncapped GP × 700 iters, non-resumable re-run) is untouched. Stays open.
**Status (orig):** raised 2026-06-22 (run 20260622T165943 watchdog post-mortem). **§4 — budget/decomposition, user-owned.** The recurring binding constraint once upstream phases are clean.

The strategizer delegates the WHOLE optimization to one implementer call. Verbatim
intent (D004): "Execute a comprehensive 1000-eval black-box minimization campaign …
PHASE 1 LHS 250 … PHASE 2 BO loop ~700 evals … PHASE 3 multi-start 50". Its thinking:
"Now I'll delegate to the implementer to execute the full campaign … fairly
open-ended." No wall-clock budgeting, no chunking (it does NOT delegate Phase 1,
check the budget, then Phase 2). The implementer runs it as a monolithic background
script (optimize_1000eval.py) whose GP-surrogate BO loop is O(n³) per refit over 700
iterations — inherently slow. Result: ~41 min, 673/1000 evals, never finished, watchdog.

**KPI contrast (the tell):** run 20260622T043904 GATED with ~996 evals in **22 min**;
run 20260622T050137 GATED with 2570 evals in 49 min. So a ~1000-eval campaign CAN fit
and gate — the failure is per-eval SLOWNESS (heavy GP-BO refits) + a no-chunk, no-time-
budget delegation, not the eval count itself.

**Classification:** JUDGMENT CALL, not a bug — a reasonable agent following the spec
(pipeline = LHS→GP→BO→optimization; implementer is "the only agent that evaluates")
would delegate "the campaign". But the system gives the strategizer no wall-clock
signal and no incentive to chunk. The implementer's own retrospective even claims an
"~18-min allocation" while the campaign needed 41 — a budget-estimate mismatch.

**Resume-cold options (user's call):** (a) strategizer chunks the campaign (delegate a
bounded slice, check budget/ledger, delegate the next) so it can gate mid-way; (b) give
the implementer a wall-clock-aware budget that caps the campaign and returns partial
results; (c) cheaper default surrogate/acquisition (the slowness is GP-refit cost); (d)
raise the 3600s watchdog. Do NOT pick without the user — budget is soft by charter.

## 11. Orphaned background process survives the watchdog kill (resource leak)
**Status:** RESOLVED 2026-06-22 (commit 5199b593) — run.py is now a process-group
leader (`os.setpgrp`) and the watchdog reaps the group (`reap_process_group` in
`watchdog_cleanup.py`) before `os._exit`. Residual: a child that `setsid`'s escapes
the group (documented). Original write-up below.

The implementer backgrounds its BO campaign as a detached process
(`uv run python …/optimize_1000eval.py &`-style). When the watchdog force-exits
run.py via os._exit(2), that detached child is NOT reaped — it kept running at ~230%
CPU after the run died (had to be killed manually). Backgrounded children outlive the
run. **Fix direction:** the watchdog should kill its whole process group (or the
implementer's Bash backgrounding should be tracked and reaped on close); alternatively
discourage detaching the campaign. No reasonable reading justifies a live orphan after
a kill.

## 12. Watchdog kill loses the strategizer's retrospective (blinds §1 Step 1)
**Status:** RESOLVED 2026-06-22 (commit 5199b593) — the watchdog now appends a synthetic
`role="watchdog"` post-mortem (`write_watchdog_retrospective`) with the last delegation
state + diagnostics + a transcript pointer, so §1 Step 1 gets a breadcrumb. (A real
first-person strategizer retro on watchdog remains impossible — the agent is killed
mid-flight.) Original write-up below.

Retrospectives are written at clean run close; the watchdog's os._exit kills the
strategizer BEFORE it writes one. So EVERY watchdog-killed run has ZERO first-person
signal from the orchestrator — the very runs we most need to diagnose (its DECISION/
FRICTION on delegation, budgeting, gating). Observed: run 20260622T165943 has
retrospectives only for D001/D004, none for the strategizer, so its campaign-
decomposition reasoning had to be reconstructed from the transcript (§1 Step 4) instead
of read directly (Step 1). **Fix direction:** have the watchdog handler flush a
best-effort strategizer retrospective (or a partial "interrupted" one) before os._exit.

---

## 13. Per-cell notebook debugger (agent can't see WHICH cell failed reproduction)
**Status:** RESOLVED 2026-06-23 — `RunPipelineCell` closure (`nodes/tools/routing.py`)
+ `diagnose_notebook` (`notebook_exec.py`); 5 headless tests + live closure smoke.
Original spec below. Surfaced by run 20260623T002417 (FAILED).

**Problem.** `CheckDeliverable` runs `pipeline.ipynb` top-to-bottom via nbclient and
returns a BINARY pass/fail with a high-level message. When reproduction fails the
strategizer cannot see *which* cell raised, its traceback, or its stdout/state — so it
iterates blindly. Primary evidence: run 20260623T002417 burned ~10 gate attempts
(`REPRO_GATE_BOUNCE`×6 → `REPRO_GATE_FAILED`) and FAILED; the strategizer's DONE
retrospective BLOCKED field reads verbatim: *"No tool to execute and debug pipeline.ipynb
cell-by-cell in isolation… the CheckDeliverable gate was binary fail/pass with high-level
error messages only. I needed a cell-level executor (run cell 'analysis' and return
stdout/stderr/state) to pinpoint whether the failure was in data loading, ledger path
resolution, output formatting, or the gate's expectations."* It never diagnosed that its
own pillar cells were stubs (`print('doe')`, …), so the REJECT was unavoidable.

**Why this is parsimonious (not overfit).** "An agent must be able to observe the failure
it is asked to fix" is a general observability principle — a philosopher nods. The binary
gate throws away per-cell information nbclient ALREADY produces. This is not a workaround
for one run; it is the diagnostic counterpart the repro gate has always lacked.

**Proposed tool (DRY — reuse the gate's runner).** Add a strategizer closure, e.g.
`RunPipelineCell(name: str | None = None)`, registered alongside `ShowNotebook` /
`RunScratch` / `CheckDeliverable` (`nodes/tools/routing.py`), backed by the SAME nbclient
execution `notebook_exec.py` already uses for the gate:
- `name` given → execute `pipeline.ipynb` top-to-bottom up to AND INCLUDING the cell with
  that `metadata.name` (fresh kernel, the gate's environment + canonical store), return that
  cell's stdout / stderr / traceback + an `errored` flag. (Top-to-bottom because cells share
  state — imports/vars from earlier cells; "run cell N in isolation" would spuriously fail.)
- `name` omitted → execute the whole notebook and return a PER-CELL trace (each cell:
  ok | errored, the first failing cell's name + traceback, stdout tail) — the granular
  version of `CheckDeliverable`.

**Boundary / safety.** Read-only diagnostic: run in the same sandboxed, zero-new-eval mode
the gate uses (must NOT mutate the canonical ledger — a notebook that calls
`get_evaluator()` against a full store is lazy/no-op; guard the same way the gate does).
Pure diagnostic — does not change the gate's accept/reject criteria (not §4 critic
substance). Likely the minimal change is to expose what nbclient already captures, so much
of the work is surfacing, not new execution.

**Done-when (KPI).** A subsequent run that hits a repro failure resolves it WITHOUT
exhausting gate attempts — i.e. `REPRO_GATE_BOUNCE` count on a recovered run drops to ≤2
(vs 6 in 20260623T002417), or the strategizer's retrospective no longer lists the
cell-level executor as a BLOCKED gap. Headless: a test that a deliberately-broken cell is
pinpointed by name + traceback (not a binary fail).

**Reuse.** `notebook_exec.py` (nbclient runner behind `CheckDeliverable`), the notebook
closures in `nodes/tools/routing.py` (`ShowNotebook`/`RunScratch` registration pattern),
`metadata.name` cell addressing (already the notebook CRUD convention).

---

## 14. Watchdog reap (#11) misses detached campaign processes (real CPU leak)
**Status:** RESOLVED 2026-06-23 — recursive reap via per-delegation PID
self-registration at the oracle entry (`governor_pids.jsonl`) +
`watchdog_cleanup.reap_governor_pids`, which kills each campaign's process tree
(incl. detached/new-session escapees the `killpg` missed). Wired into
`studies/.../run.py` `_watchdog` and the memory watcher; see FEATURES.md §E.
Validated on the resource-governance wet run (zero orphans at close).
Original write-up kept for the record.
Discovered live during audit run 20260623T015907.
HIGH — leaks a full CPU core per watchdog-killed run; orphans accumulate for days.

**Problem.** #11's reap (`watchdog_cleanup.reap_process_group` -> `os.killpg(pgid)`)
kills only run.py's process GROUP. The implementer launches its optimization
campaign as a DETACHED background process (`/tmp/campaign_v2.py`) with poll-loop
bashes waiting on `campaign_report.json` -- these run in a NEW session, so they
escape the group kill (the residual risk the code comments already flagged) and
outlive the watchdog `os._exit(2)`.

**Primary evidence (audit run 20260623T015907, watchdog_killed at 60min).** Minutes
after the watchdog fired, `ps` showed PID 47561 `python3 /tmp/campaign_v2.py` at
125% CPU / 42min CPU still running, plus two `until [ -f .../D003/
campaign_report.json ]` poll bashes (63962, 59331), AND an ancient zombie from a
2-day-old run (26085, "Waiting for 975 evaluations", polling run 20260621T155717).
The leak accumulates across runs/days. Had to `kill` them by hand to free the CPU
before the next run.

**Fix direction (touches the watchdog -- needs approval).** At reap time the
campaign's parent chain is still intact (it reparents to launchd only AFTER run.py
exits), so a recursive process-TREE walk by PPID at reap time catches it where the
process-group kill does not. Options: (a) dependency-free recursive `pgrep -P`
BFS from os.getpid(), SIGTERM each (recommended); (b) `psutil.children(recursive=True)`
(adds dep); (c) a spawned-PID registry. Keep the walk BEFORE `os._exit` (chain intact).

**Done-when.** After a watchdog kill, `ps aux | grep -E 'campaign|campaign_report'`
returns zero leftovers. Headless: extend `test_reap_kills_a_detached_background_process`
to a GRANDCHILD in a new session that the group kill misses (the current test's
child is a session leader pid==pgid, so the group kill happens to catch it).

**Reuse.** `watchdog_cleanup.reap_process_group`, `studies/agentic_black_box_3d/run.py`
`_watchdog`, `tests/agentic/test_watchdog_cleanup.py`.

---

## 15. Verdict oscillation on borderline cases — finding + resolved trigger
**Status:** RESOLVED 2026-06-23. Root cause FIXED (commit 88856cb9, the #9
validator memory); the §4 trigger question was RESOLVED **with no Charter change**
(see "RESOLVED" below). Captured here because the run artifacts that surfaced it
are ephemeral (wiped on the next run).

**Finding (two wet runs, bb3d).** Haiku run `20260623T194849` GATED but took 4 gate
attempts; H1/H2 each oscillated FALSIFIED↔INCONCLUSIVE 4-5× (`VERDICT_SUBSTANCE_FLAG`=4).
The live verdict validator was **stateless** — it re-judged each verdict from
scratch, with no view of its own prior rulings, so on a *borderline* result it
flipped. The Sonnet-strategizer A/B `20260623T212346` GATED in 2 attempts with
**0 oscillation** (best_f ≈ −1.000, the true optimum; cheaper: $1.34 vs $1.61).

**Crucial nuance (don't over-read the A/B).** The validator ran on **Haiku in BOTH
runs** (it reuses the CRITIC adapter, telemetry-confirmed: `verdict_validation |
model=claude-haiku-4-5`), so validator model strength was a *constant*. The Sonnet
win came from the **strategizer** writing grounded, calibrated predictions
(`f ≤ −0.998` vs the known prior-best −0.9958) that produced CLEAR met→SUPPORTED
outcomes — it avoided the borderline trigger UPSTREAM, it did not make the referee
more consistent. So: **oscillation is triggered by borderline verdicts**, not by a
weak referee per se.

**What was fixed.** #9 validator now gets its prior rulings on the same hypothesis
(ledger `status_log`) + a justify-any-reversal guard → can't silently flip. This is
INSURANCE: on harder problems (supercompressible), a severe test that misses a
*well-calibrated* prediction by a hair is genuine knife-edge science, not a
calibration artifact — so the borderline case WILL recur regardless of model
strength, and the stateless defect would have bitten again.

**RESOLVED (§4, user's call 2026-06-23) — NO Charter change.** The question was
whether a near-miss on a severe test is FALSIFIED or INCONCLUSIVE. Conclusion: the
existing Charter §3 already answers it by the correct variable — **adequacy, not
margin** — so a new "boundary" rule would be overfit AND a claim-protecting
loophole:
- near-miss from an **inadequate/under-budgeted** search → INCONCLUSIVE via the
  existing §3 confound/Duhem–Quine clause (did the hypothesis fail, or did the
  optimizer just not get there?);
- near-miss from a **genuinely adequate** test → FALSIFIED, and that is correct
  (the registered prediction was simply false; declining it would "protect a
  favoured claim", which §3/§4 forbid).
A blanket "near-miss → INCONCLUSIVE" collapses those two and lets an agent dodge a
real refutation — a step away from Popper, not a refinement. The campaign
delegations' "zone-based" logic is the loophole, not the intent. If a threshold
turns out to be a bad operationalization, §4 already prescribes the fix: revise the
prediction explicitly and re-test — don't reinterpret the old result. The
oscillation itself is handled by the validator-memory fix + calibration, not by
softening §3.

**Minor tool friction (1 occurrence, note-only).** `EditPipelineCell` rejected a
full-field edit lacking `expected_rev` (an optimistic-concurrency guard) → one
`ERROR_RETURN`; the agent had to `ShowNotebook` first to get the rev. Watch for
recurrence before treating as a fix.

---

## 16. ABAQUS subprocess can't import workspace modules (PYTHONPATH) — abaqus2py-owned

**Status: open — recommendation only (not an f3dasm fix).** During run
`20260624T021359`, all 51 ABAQUS runs of the first D003 attempt failed with
`ImportError: No module named 'supercompressible_lin_buckle_param'`: the workspace
dir was not on `PYTHONPATH` when the ABAQUS subprocess ran (it worked in validation
only because a validate script did `sys.path.insert(0, WORKSPACE)`). This also
contaminated the canonical store with 51 NEW-status rows that had to be cleared
before D004 (meta_errors.md Bug 1).

**Owner: the external `abaqus2py` package**, not f3dasm — `F3DASMAbaqusSimulator`
(which generates the `preprocess.py` wrapper and launches ABAQUS) is imported from
`abaqus2py` (`studies/fragile_becomes_supercompressible/main.py`), which is not in
this repo. **Recommended fix (in abaqus2py):** inject the workspace dir into the
ABAQUS subprocess environment (`PYTHONPATH`) or emit `sys.path.insert(0, WORKSPACE)`
into the generated `preprocess.py`, so worker-authored param modules resolve without
relying on the parent process's cwd/sys.path.

## 17. Closure decoupled from success-criteria + budget; budget-severity model — §4

**Status: open, §4 user-owned. Partially addressed.** Run `20260624T021359` closed
at ~22% of a 12h budget with its PRIMARY success criterion (Stage-2 Riks
`max_strain ≥ 0.90`) left INCONCLUSIVE and the affordable settling experiment (a
refined Riks re-run) never delegated. Root: a three-way gap — the strategizer prompt
made Done() eligible on "best design + one falsification attempt" (no criteria-met
notion), budget reached the agent only as a 95%/100% *brake* (never as runway,
`strategizer.py` budget warnings), Done() is budget-blind (`routing.py`), and the
critic is *forbidden* to weigh budget ("RESOURCE BOOKKEEPING IS NOT VALIDITY",
`agents/critic.py`) and is not scoped to closure timing.

**Done (`0400a653`):** the strategizer PREMATURE CONVERGENCE rule now requires
primary criteria MET (not merely tested), frames budget as runway, and asks for a
recorded reason when closing early; mirrored in the Done() docstring.

**Deferred (the open §4 design question):** a *runtime* closure nudge (soft — a
two-shot reconsider when closing with large budget unused and a criterion
unmet/INCONCLUSIVE, mirroring the 95% wind-down nudge), and a **budget-severity
model**. The user's framing: budgets differ in severity — roughly **Memory > Time >
Eval** — and *accidental* constraints (e.g. a Riks `max_strain` gate) are
**dynamically assigned**, so they should receive *differentiated* treatment rather
than one uniform nudge. How to represent constraint severity/type and key the nudge
on it is unresolved. The benchmark `PROBLEM_STATEMENT.md` framing (objective excludes
strain; Riks demoted to a separate "Validation requirement"; criterion #4 buried)
contributed and is owned by the user (handled in the benchmark repo, not f3dasm —
robustness must not depend on a perfectly-framed problem statement).

## 18. Critic should flag an infeasible-extremum headline on a constrained study — §4

**Status: open, §4 user-owned.** Commit `6494489b` removed the runtime
`REPRODUCED` extremum machine-check (it forced constrained studies to headline
their infeasible unconstrained extremum) and shifted headline grounding onto the
critic. But the critic's HEADLINE PROVENANCE mandate (`agents/critic.py:31-41`)
only verifies the headline value is **real** (traces to a ledger row) — it does
NOT verify the headline is **feasible**. So nothing currently flags the specific
failure of run 20260624T021359: a headline equal to the unconstrained extremum
(λ_cr_nd = 0.90709, a NON-coilable design) on a study whose objective is
`maximize λ_cr_nd subject to coilable = True`.

**Proposed (one clause, §4):** add to the critic's headline check that, when the
study declares a feasibility constraint (a constraint output column), the headline
must be the best **feasible** design; a headline equal to an infeasible extremum
is a MAJOR finding. The constraint identity could come from
run_config (an explicit constraint column) or be inferred from the deliverable's
stated objective. Deferred pending the user's call on how to represent the
constraint to the critic (it is the epistemic-contract owner's decision).

**Why not the runtime instead:** the runtime can't judge which ledger column is
the constraint or what counts as feasible generically without more config; the
critic already reads the deliverable's stated objective, so it is the better
place to judge feasibility-of-headline. See `6494489b` and `agents/critic.py`.

## 19. Scientific adequacy is enforced as vibes while reproduction is enforced hard — §4

**Status: partially addressed (`8ae9a402`, `93e6f0f2`); the load-bearing question
is OPEN and empirical.** Diagnosis from run `20260625T014520` (supercompressible
14D). Numbers (faithful): 257 evals, 47 coilable (~18% feasible), exactly **1**
above the Bessa baseline (66.04 vs 65.3, a 1.1% edge), **0** above the +15% floor
(75.1). Closed voluntarily at 3 h 36 m wall (not watchdog-killed). The deliverable
declared "66.04 … represents the effective performance ceiling in this design
space" (H6 cell) and H4 FALSIFIED.

**The failure (established):** a synthesis-level over-generalisation. The run
converted "my underpowered search did not clear the floor" into "the 14D space
cannot." Its OWN deliverable documents the search as weak — ml cell: "CV R² ≈ 0.51
on 27 coilable points"; doe cell cites Loeppky 2009 "≥10×d" (=140) then uses 50;
D005 retrospective: GP length-scale pinned at bound 1000, "poorly tuned." The BO's
feasible hit-rate (3/37 ≈ 8%) was *worse* than the LHS (18%). The whole campaign
was pre-committed to a narrow box (`ratio_Ixx ∈ [5e-7,1.4e-6]`, `phase2_plan.md`)
derived from a 1-D scaling formula **before** the first LHS returned — so 14D was
never broadly explored; a physics-intuition ray was.

**Mechanism (established by reading the code):**
- The Charter's §2 severity rule IS present and IS applied — but only
  **per-hypothesis**. H4's registered prediction was scoped to "this search finds
  ≥75.1," so FALSIFIED is charter-legal. The over-reach lives in the **synthesised
  prose**, which no registered hypothesis covers.
- The live `verdict_validator.py` shares ONLY the Charter (not the critic's
  checklist) and judges ONE hypothesis verdict at a time — it is **structurally
  blind to the synthesis**.
- Both critic calls (`call_001` REVISE, `call_002` PASS) spent 100% on
  reproduction mechanics (composable BO, lazy guard, IN_PROGRESS jobs); neither
  engaged criteria 2/3/4 against the headline. The critic prompt’s criterion 6 was
  marked "binding" and its prose equated "scientific integrity" with "provenance +
  replicability", steering attention there. The gate proved the notebook
  *reproduces*, never that the science was *sound* (CLAUDE.md §4.5).
- Budget cost-prior: planned ~72 sims for 14D on an unverified ~600 s/sim
  assumption; measured median was 16.4 s (36.5x off), never recalibrated.

**Done this session (general principles, not patches — each passes the parsimony
test):**
- `8ae9a402` — Charter §2 extended so an achievement/absence claim ("some/no
  design reaches X") is adequately tested only if the search had the POWER to find
  the instance (coverage + a surrogate above chance); a stalled search →
  INCONCLUSIVE. Lives in the Charter, so the gate critic AND the live validator
  inherit it (the validator can now flag a premature FALSIFIED **on the fly at
  HypothesisUpdate**). Critic criterion 4 made CRITICAL-eligible for whole-space
  headlines; criterion-6 "integrity = reproduction" sentence corrected to
  "necessary but not sufficient." Strategizer PREMATURE CONVERGENCE: a stalled
  optimizer/plateau is NOT a valid "budget can't settle it" reason.
- `93e6f0f2` — per-delegation wall-time KPIs auto-appended to the report (attacks
  the cost-prior; interpretation-free to avoid overfitting).

**OPEN — the empirical question these edits do NOT answer.** Whether the failure
is *scaffold* (the harness graded reproduction, so the model optimized it) or
*model disposition* (commercial LLMs fine-tuned to terminate with a confident
answer, antagonistic to staying INCONCLUSIVE) is **unresolved and untested**. The
edits are an intervention, not a proven fix. Decisive test (one run): re-run this
study with `8ae9a402`+`93e6f0f2` live — if the agent stays INCONCLUSIVE / keeps
exploring, or the validator/critic flags the ceiling claim → scaffold; if it still
manufactures a confident ceiling → disposition. Do NOT record these edits as
"fixed" until that run exists. Note in favour of scaffold (not proof): the same
agent marked H5 INCONCLUSIVE correctly, so the capability is present.

**OPEN — structural gap (§4, user-owned).** The synthesis-level over-claim is
reachable only by the critic (criterion 4); the validator cannot see it because it
judges one hypothesis verdict at a time. Widening the validator's scope to the
synthesised headline — so it can catch this on the fly rather than at Done() — is
an architectural change the user owns. Related: #18 (critic blind to
infeasible-extremum headlines) and #17 (budget-severity model) are the same class
— the critic/validator scrutinise mechanics and atomic verdicts, not the headline
as a communicated scientific claim.

---

## #22 — Stall watchdog: liveness = "file written", not "progress made" (backstop defect)

**Severity:** medium (a backstop, not a primary control — the real cure for the
hang it failed to bound is the per-call validator timeout, shipped in `9d58b2a3`).

**What happened.** Run `20260627T211310` (watchdog_killed, 2h20m). A verdict-validator
LLM call hung at 22:03:29 (see `9d58b2a3` for the root cause). The study's stall
watchdog (`studies/agentic_namespace_ring/run.py` `_watchdog`, using
`watchdog_cleanup.seconds_since_last_activity`) is supposed to force-exit a hung run
after `STALL_SECONDS` (1200s here). It did fire — but only at **idle 5378s (~89 min)**,
~4.5× its own threshold. The watchdog post-mortem records `force-killed at 5378s`.

**Suspected cause (UNCONFIRMED — snapshot mtimes corrupted by the batch's non-`-p`
cp, so not provable from preserved artifacts):** `seconds_since_last_activity` =
most-recent mtime of ANY file under the run dir. Liveness so defined is satisfied by
a hung-but-still-twitching CLI subprocess (partial transcript flushes, checkpoint WAL,
telemetry) — i.e. *activity ≠ progress*. A run can write bytes while making zero
scientific progress, resetting the idle clock. (Alternative: daemon-thread starvation
under a GIL-holding loop — also unproven.)

**Proposed fix (deferred — the user flagged the watchdog as a SYMPTOM; do not
re-prioritise it over root causes):** define "stall" as *no PROGRESS* — no new ledgered
evaluations and no delegation state-transition for the window — rather than *no file
written*. Catches both a true hang and a grind-without-progress, and never kills a run
that is still producing evals (honours "never penalise parallel/slow-but-live work").
Needs a progress signal the watchdog can read cheaply (e.g. max over ledger row count +
delegation_log completed count). Validate headless before trusting it.

**Why not now:** with the validator call bounded (`9d58b2a3`), the specific hang that
exposed this can no longer run 89 min — it aborts in ~2 min. The watchdog defect only
matters for a *different*, not-yet-observed hang that the per-call timeouts don't cover.
Fix it when such a case appears, or as deliberate hardening — not as symptom-chasing.

---

## 23. `consultant` — broaden the literature_reviewer into a research+docs consultant

**Status:** spec only; NOT built (user decision 2026-06-30: "one agent, general
but sharp … spec it and put it on the backlog"). Name decided: **`consultant`**.

**Motivation (evidence).** Run `20260629T191754` (supercompressible-material-
creative) shows the datagenerator/implementer repeatedly brute-forcing live
tech-stack gotchas with no doc-lookup channel: `.fil` vs `.odb` for Abaqus
`*IMPERFECTION` (D007), `max_waiting_time=60` too short for Riks preprocessing
(D007), `except RuntimeError` not catching `CalledProcessError`/`TimeoutError`
(D003), `data.add()` not existing on `ExperimentData` (D003), store ordering by
completion time (D004). Each is a documentation question the agents could not
ask anyone — the literature_reviewer can only search *academic papers*
(Corpus/Semantic Scholar/OpenAlex/arXiv), not Abaqus or Python docs.

**Current state (the channel already exists — this is mostly capability+prompt,
not topology).**
- `agents/_graphs.py` already wires `datagenerator → literature_reviewer` and
  `implementer → literature_reviewer`, and `agents/datagenerator.py` already
  instructs `Delegate(target="literature_reviewer", …)`.
- BUT `agents/datagenerator.py:48` says *"Delegate for methodology, not for
  Python syntax"* — the exact opposite of consulting for an API gotcha.
- AND `agents/literature.py` has **no** general-web tool (no WebSearch/WebFetch);
  its toolset is academic-paper search only.

**Proposed changes (one agent, two modes — "general but sharp").**
1. **Add `WebSearch` + `WebFetch`** to the agent (general web covers Abaqus,
   Python, any tech stack — no per-tool MCP needed). FEATURES.md entry required
   in the same commit (tool catalog is enforced by
   `tests/agentic/test_features_documented.py`).
2. **Flip the guidance** in `agents/datagenerator.py` (and the implementer) so
   workers may consult for tooling/API/doc questions, not just methodology.
3. **Rename** `literature_reviewer` → `consultant` everywhere: the agent class
   `role`, the node name + edges in `_graphs.py`, every prompt reference
   (strategizer/datagenerator/implementer/critic), the KB-menu audience filter,
   and the milestone/gate text that special-cases `literature_reviewer` (e.g.
   `milestones.py` "literature_reviewer is never gated" and its tests). This is
   the bulk of the mechanical churn — grep `literature_reviewer` across `src/`
   and `tests/` first; ~dozens of sites.
4. **Two-mode prompt (the one real risk).** The current prompt is science-
   citation-heavy (corpus, falsification support). A doc lookup needs *different*
   rigor — the right answer + a source URL, fast — not a literature synthesis.
   The prompt must explicitly distinguish: (a) *literature mode* (academic claim
   → cite a paper from the corpus) vs (b) *docs mode* (API/tooling question →
   authoritative doc/source URL, concise). Without this the agent will
   over-academicize a one-line API question.

**Scope guard.** One agent, not a split — the web tools serve both modes and a
second node is churn without evidence the roles conflict. Revisit only if a run
shows the two modes degrading each other.

**Not §4.** This is agent capability/tooling + prompt, not science epistemics —
no science_monitor / charter / critic-criteria / budget change. Build under the
normal contract (headless test first: assert the renamed node + edges resolve,
the new tools appear in the catalog, and both preamble/guidance render; e2e
behavior-only last).
