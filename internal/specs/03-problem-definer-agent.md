# Spec 03 — ProblemDefinerAgent: a pre-strategizer intake stage

**Backlog #3.** Priority: medium. Status: spec.

## Goal
Turn a raw human problem into a high-signal, airtight brief **before** the
strategizer runs — pinning (a) the problem statement, (b) the ExperimentData
output-column schema + Domain, (c) hard-to-automate plumbing (evaluator
entrypoint, budgets) — so the strategizer spends budget on science, not on
inferring schema and reconciling config vs prose.

## Primary evidence — the seam already exists
There is a working **pre-pass precedent**, and it is the cleanest hook:
- `_review_problem_statement()` (`agent_runtime.py:703-769`) runs in `execute()`
  **before** the graph is built (`agent_runtime.py:500-503`), invokes the
  tool-less one-shot `ProblemStatementReviewerAgent` (`reviewer.py:86-93`), judges
  5 universal elements, **always** writes `debug/problem_statement_review.md`
  (`:734`), and — when interactive — prompts per-gap via `input()` and writes
  `debug/problem_statement_addendum.md` (`:764`). It is **advisory and
  non-blocking** (catches all exceptions, `:730`) and returns the augmented
  problem text.
- Config is loaded by `_load_study_config` (`agent_runtime.py:49-55`, YAML); the
  `evaluator` block → `_init_canonical_store` writes `run_config.json` keys
  `evaluator_entrypoint` / `evaluator_output_names` (`agent_runtime.py:107-111`);
  the DataGenerator agent can also register them at runtime via
  `register_evaluator_entrypoint` (`agent_runtime.py:117-172`,
  `routing.py:842-846`).
- **Gap (evidence):** output column names are *either* in `config.yaml` (pre-run)
  *or* discovered at runtime by the DataGenerator (`instrumented.py:352-361`
  requires `output_names` for bare-fn entrypoints). There is **no pre-run
  mechanism to establish the schema** before the graph runs — exactly the gap
  this agent fills.
- Agent class pattern to mirror: `Agent` base (`backends/base.py:126-186`:
  `system_prompt`, `tools`, `description`, `role`, `build_closure_tools`);
  `ProblemStatementReviewerAgent` (`reviewer.py:86-93`) is the minimal example.

## Design (DRY — extend the pre-pass, don't add a graph node)
Two seams exist: a **pre-pass** (advisory, `agent_runtime.py:500-503`) and a
**graph entry node** (`Graph.entry`, `graph_builder.py:92`). Recommend the
**pre-pass** — it already has the precedent, runs before the ledger schema is
fixed, and avoids re-plumbing the entry node (only the entry node gets the full
closure toolset, `graph_builder.py:68-77`, so making ProblemDefiner the entry
would demote the strategizer).

1. **Promote the reviewer to a definer** that *authors*, not just *flags*. Reuse
   `ProblemStatementReviewerAgent`'s structure and the `REVIEW_ELEMENTS`; extend
   its output to a structured brief: objective (min/max), Domain (vars+bounds+
   types), **output columns** (objective col, feasibility cols, units), evaluator
   entrypoint, budgets.
2. **Persist the brief as reproducible artifacts** (DRY with existing writers):
   write/upgrade `config.yaml` (evaluator block, budgets) and an enriched
   `PROBLEM_STATEMENT.md`, and pre-seed `run_config.json` schema keys via the
   existing `_init_canonical_store` / `register_evaluator_entrypoint` — so the
   ledger schema is fixed from turn one and the brief itself is auditable.
3. **Interactivity** reuses the now-fixed TTY-guarded prompt path (see the
   FollowUp headless fix): interactive → ask the human to confirm/fill the
   schema; non-interactive → proceed with best-inferred brief + record
   assumptions (never block a headless run, mirroring `:730`).
4. **Relationship to the advisory pass:** the ProblemDefiner *replaces*
   `_review_problem_statement` (superset), or wraps it — decide at build time;
   keep ONE pre-pass, not two.

## TDD plan (tests first)
1. `test_definer_writes_run_config_schema` — given a problem + evaluator, the
   pre-pass seeds `run_config.json` with `evaluator_output_names` before the graph
   runs (so the ledger schema is fixed turn-one).
2. `test_definer_noninteractive_never_blocks` — headless run proceeds with an
   inferred brief + recorded assumptions (parity with `:730`).
3. `test_definer_interactive_collects_schema` — with a stubbed TTY, gaps are
   filled from operator answers and written to the brief.
4. `test_definer_brief_is_reproducible_artifact` — `config.yaml` +
   `PROBLEM_STATEMENT.md` are (over)written and re-loadable by `_load_study_config`.
5. `test_single_prepass` — exactly one pre-pass runs (no double review).

## Risks / out of scope
- **Risk:** auto-writing `config.yaml` could clobber a hand-authored one — gate on
  "only fill missing keys" or write an addendum, never silently overwrite.
- **Risk:** schema inference for an opaque evaluator is hard; when unsure,
  defer the column schema to the DataGenerator (don't over-claim).
- **Out of scope:** making the ProblemDefiner the graph orchestrator;
  multi-objective schema negotiation.

## Done when
A pre-pass produces an airtight brief (problem + Domain + output schema +
plumbing), persists it as reproducible artifacts, fixes the ledger schema before
the graph runs, never blocks headless, and replaces the advisory review —
tests 1-5 green. KPI test: fewer SCIENCE_DRIFT/MILESTONE_BLOCK diagnostics
traceable to an under-specified brief on a re-run.
