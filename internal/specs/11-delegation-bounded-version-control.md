# Spec 11 — Delegation-bounded version control of the workspace

**Status:** spec. **Priority:** medium. **Depends on:** nothing.

## The problem, in one sentence

A delegation's account of what it changed is **prose the agent wrote about
itself**, and nothing on disk can contradict it.

## Evidence

Run `20260912T142229` (mathexpert_kinematic_matching_test), three independent
observations, all verified against the raw artifacts rather than a
retrospective's own account of them:

1. **A deliverable's file claims are unverifiable.** `_finish_ok`
   (`nodes/tools/routing/delegation.py:648`) records `deliverable=text` — the
   worker's own narration — as the permanent record of the delegation. No
   artifact records which files the worker actually created, modified or
   deleted. The reproduction gate proves `pipeline.ipynb` *runs*; it says
   nothing about whether a delegation's story matches its edits. This is the
   §4.5 faithfulness axis, structurally out of reach.

2. **Intermediate epistemic states are destroyed, and the loss is invisible.**
   D002's retrospective: *"Initial coefficient checks failed (INCONCLUSIVE) …
   Once included, all 6 confirmed."* `main_summary.json` records only the final
   state: `11 ASSERTED / 6 CONFIRMED`, zero INCONCLUSIVE. The agent re-ran the
   script from a fresh `Workspace`, so the failed check is not merely omitted
   from the summary — no artifact of the run contains it. A check that failed,
   prompted a correction, and then passed is the *most* epistemically
   interesting event in the delegation, and it leaves no trace. This defeated a
   direct audit: a reviewer cross-checking the strategizer's verdict comment
   against the summary JSONs concluded the cited INCONCLUSIVE was fabricated.
   It was not; the evidence had been overwritten.

3. **The workspace is shared across runs, and that is a science hazard.** The
   study's workspace lives at `study_dir/runs/math_workspace/`
   (`knowledge/entries/0011-symbolic-derivation-patterns.md`), outside the
   timestamped `runs/<ts>/` tree, so it survives `run.py`'s pre-run wipe. Before
   this run, `math_workspace/main.py` and `lcp_variant.py` held the **answers**
   from prior runs; they had to be archived by hand or the agents would have
   been handed the conclusion they were asked to derive.

## What to build

A git repository, initialised per run, committed once per delegation.

### Mechanism

- **Run start.** `run_setup.py` creates `runs/<ts>/workspace/` and runs
  `git init` in it (`git -c init.defaultBranch=main init`, local
  `user.name`/`user.email` set in that repo's config so a missing global git
  identity cannot fail the run). One empty root commit, so the first
  delegation's diff has a parent.
- **Delegation end.** In `WorkerSession._finish_ok` and `._finish_error`
  (`delegation.py:577`, `:760`), immediately before
  `node._delegation_log.record(...)`: `git add -A` then `git commit` with
  message `<delegation_id> <from_node> -> <to_node>`, allowing an empty commit
  so every delegation has exactly one commit whether or not it touched a file.
- **The record carries the sha.** `DelegationLog.record()` gains
  `workspace_sha: str | None = None`, additive and defaulting to `None` exactly
  as `phase` and `constraints` did.
- **Failure is never fatal.** Every git call is wrapped; any failure logs a
  warning, records `workspace_sha=None`, and the run continues. Version control
  is an observation instrument, not a dependency of the science.

### Why per-delegation is the right boundary

A delegation is already the system's unit of accountable work: it has one
`task`, one `deliverable`, one entry in the log, one entry in the ledger's
provenance chain. Making it also the unit of version control means the
question "what did this delegation actually do" has one answer in the same
granularity as every other record of it. Per-turn would fragment below the unit
anyone reasons about; per-run collapses above it.

### What it makes possible (not built here)

Downstream consumers become possible once the sha exists; none are in scope:

- The critic can diff a delegation's commit against its parent and compare the
  result to the deliverable's prose — evidence where there is now only
  testimony.
- Evidence (2) becomes recoverable: the script that produced an INCONCLUSIVE
  is in D002's commit even after a later commit overwrites it.
- A `WorkspaceDiff(delegation_id)` read-only tool, if the audit turns out to
  need it. Deliberately not specified here — build the record first, and let
  a real audit demand the reader.

### The workspace-location change is part of this

Moving the workspace under `runs/<ts>/` is not an incidental cleanup; a git
repo shared across runs would carry prior runs' answers into a new run's
history, making the spoiler problem worse rather than better. The handbook
entry's path must move with it. Agents keep continuity within a run (a later
delegation still reads the earlier one's file) and lose it between runs, which
is the correct behaviour for a study whose conclusion is the thing under test.

## Tests, named first (TDD)

Failing before, passing after:

- `test_workspace_repo_initialised_at_run_start` — `runs/<ts>/workspace/.git`
  exists and holds exactly one (root) commit.
- `test_each_delegation_produces_exactly_one_commit` — three delegations, one
  of which writes no file: `git rev-list --count HEAD` is 4 (root + 3).
- `test_delegation_record_carries_its_workspace_sha` — the sha on `D002`'s
  record resolves, and `git show --stat` on it names the file D002 wrote.
- `test_failed_delegation_is_still_committed` — `_finish_error` commits, so a
  crashed delegation's partial edits are inspectable rather than lost.
- `test_git_unavailable_does_not_fail_the_run` — `git` missing from PATH: the
  run completes, records carry `workspace_sha=None`, a warning is logged.
- `test_workspace_is_run_scoped` — two runs of one study get independent
  workspaces; run 2's does not contain run 1's files.

## Done when (KPI)

**Mechanism claim** (tests pass): every delegation record in a completed run
carries a resolvable `workspace_sha`, and `git rev-list --count HEAD` equals
the delegation count + 1.

**Behavioral claim** (must be measured on a re-run, not assumed): on a repeat
of `mathexpert_kinematic_matching_test`, a reviewer answering "which files did
D002 change" reads it from `git show --stat` rather than from the
deliverable's prose. This is an *auditability* claim; it does not predict a
change in gate outcome, gate attempts, or `ERROR_RETURN`, and should not be
reported as one. If it is later paired with a critic that reads the diff, the
KPI to watch is gate attempts on a run whose deliverable overstates its work —
but that is a different spec.

## Deliberately out of scope

- Committing anything the agent did not produce (the run's `debug/` tree is
  already append-only and separately durable).
- Giving agents any git tool. They never see the repo; the harness commits on
  their behalf. An agent that can rewrite the history recording its own work
  defeats the purpose.
- Capturing intermediate states *within* one delegation. Evidence (2) is
  mitigated (the script is recoverable per delegation) but not solved: a
  Workspace re-run inside a single delegation still overwrites its own earlier
  verdicts. Whether `write_summary` should carry a step's verdict *history* is
  an epistemic-record question and belongs to the user under CLAUDE.md §4.
