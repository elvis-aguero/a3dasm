# Agentic system — feature catalog

The single place that says **what the agentic system can do, why, and where it
lives.** Read this to get your bearings without reading code.

> **Contract (enforced):** every agent tool listed in an agent's `tools` set MUST
> appear in the "Tools" table below — `tests/agentic/test_features_documented.py`
> fails the build otherwise. Every new *capability* (tool OR infrastructure)
> MUST get an entry here in the same commit that adds it. The test can only
> enumerate tools; infrastructure features rely on this written contract.

Format per feature: **what** (plain language) · **why** · **where** (files) ·
**config** (if any) · **status**.

---

## A. Science & orchestration

### Hypothesis ledger
- **What:** the run's record of falsifiable hypotheses and their verdicts (OPEN /
  SUPPORTED / FALSIFIED / INCONCLUSIVE), append-only.
- **Where:** `hypothesis_ledger.py`; per-run file `debug/strategizer_notes/hypotheses.json`.
- **Status:** core.

### Falsification charter (the Popperian rules)
- **What:** the single binding text defining how a hypothesis may be tested and
  labelled (severity of the attempt, verdict follows the result, no goalpost-moving).
- **Why:** one shared standard both the strategizer and the critic cite.
- **Where:** `knowledge/charter.py`. **Status:** core (§4 user-owned).

### Live verdict validator (#9)
- **What:** when a hypothesis is closed, an independent referee checks — *live* —
  that the verdict obeys the charter, and nudges the strategizer if not.
- **Why:** the gate critic only checks at the end; this catches charter violations
  at the moment of assertion.
- **Where:** `verdict_validator.py` (judge logic); invoked by `nodes/strategizer.py`
  HypothesisUpdate via `node._run_verdict_validator`, which is defined in
  `nodes/critic_gate.py`. Runs on the **critic's** model (reuses the critic adapter),
  not the strategizer's — one refereeing standard, decoupled from the agent it judges.
- **Memory (anti-oscillation):** the judge is fed its own prior rulings on the SAME
  hypothesis (from the ledger `status_log`, via `_prior_rulings_digest`) with a
  justify-any-reversal guard, so a borderline verdict can't silently flip between
  calls. Mirrors the gate critic's prior-reviews digest.
- **Bounded budget:** the advisory call runs with a tight `idle_timeout=120s` +
  `retry_max=1` (NOT the run-wide 5×600s agent-turn budget). A hung CLI stream once
  froze a whole run for ~89 min here; on any timeout/failure the verdict simply
  stands (the call is advisory).
- **Config:** kill switch `F3DASM_VERDICT_VALIDATOR=0`. **Status:** advisory, non-blocking.

### Science monitor
- **What:** background rules that flag scientific drift and escalate repeated
  drift to the critic. Two provenance-integrity rules bracket the eval ledger
  from both directions: **UNLEDGERED_EVALS** (a delegation reported evals but
  wrote no attributable rows — evals that never reached the store) and
  **UNSTAMPED_ROWS** (the store gained rows with no provenance owner — the
  reverse: rows written outside get_evaluator() via the public
  ExperimentData.store() door, neither counted nor reproducible). Both warn-only.
  A third rule, **DUPLICATE_EVALUATION**, flags a delegation re-evaluating a
  design point already FINISHED, unchanged, in the ledger (real incident: a
  delegation re-sampled an identical seed=42 LHS design three times, 122 of
  160 rows pure waste — backlog #24). Counter-based, not level-triggered: fires
  once 3 NEW duplicate rows land since the last check, then resets; capped at
  2 nudges per delegation; rate-limited to one per 60s. `Wait()`'s poll loop
  also drains the monitor on every 10s tick (not just on the next tool call),
  so a nudge reaches a strategizer blocked waiting on a live campaign instead
  of surfacing only after the whole delegation (and its budget) is spent.
- **Where:** `science_monitor.py` (`_check_unledgered`, `_check_unstamped_rows`,
  `_check_duplicate_evaluations`); `instrumented.py` `unstamped_row_count`,
  `duplicate_eval_stats`; `routing.py` `Wait()`. **Status:** core (§4 user-owned).

### Process milestones
- **What:** a small backlog (assess-literature, oracle-ready, …) that gates the
  implementer until the strategizer resolves each (complete or skip).
- **Where:** `milestones.py`. **Status:** core.

### Delegation + inter-agent messaging
- **What:** the strategizer delegates work to specialist agents and they report back;
  agents can ask one clarifying question, send async messages, and report progress.
- **Where:** `nodes/tools/routing.py`, `nodes/strategizer.py`.
- **Tools:** `Delegate`*, `GetStatus`, `Wait`, `FollowUp`, `Confer`, `ReportEvals`.
  (*Delegate is injected dynamically, not in a static `tools` set.)
- **Status:** core.

## B. The deliverable (pipeline.ipynb)

### Notebook authoring + reproduction gate
- **What:** the single deliverable is a Jupyter notebook; the runtime re-executes it
  lazily and accepts it only if it runs cleanly, adds zero new oracle evals, and
  leaves the ledger unchanged. The printed `REPRODUCED:` headline is informational —
  the critic checks its provenance (it must trace to a real ledger row); the runtime
  no longer machine-matches it to an objective extremum (that wrongly rejected
  constrained optima — audit 20260624T021359).
- **Where:** `notebook_exec.py`, `nodes/tools/routing.py`, `nodes/strategizer.py`
  (`_reproduction_gate`).
- **Tools:** `AddPipelineCell`, `AddPipelineMarkdownCell`, `EditPipelineCell`,
  `DeletePipelineCell`, `ShowNotebook`, `WriteDeliverable`, `CheckDeliverable`.
- **Status:** core (the live deliverable).

### Per-cell notebook debugger (#13)
- **What:** run pipeline.ipynb against a *copy* of the ledger and get a per-cell
  pass/error trace, so a failing cell can be pinpointed instead of guessing.
- **Where:** `notebook_exec.py` `diagnose_notebook`, `RunPipelineCell` closure.
- **Tools:** `RunPipelineCell`. **Status:** done.

### Unified, SDK-compatible Bash surface: `Bash` + `BashOutput` + `KillShell` (#24)
- **What:** one tool SURFACE across every backend. `Bash(command, timeout?,
  run_in_background?, description?, dangerouslyDisableSandbox?)` runs foreground
  by default; a command that exceeds its timeout is **backgrounded, not killed**,
  and returned with a `bash_id` the agent polls via `BashOutput(bash_id)` and
  stops via `KillShell(bash_id)`. Matches the Claude-Agent-SDK Bash param and
  tool names so agents don't relearn behavior. The one deliberate deviation:
  auto-background is made **visible** (an `interrupted` notice + `bash_id`) so an
  agent never wakes up thinking a still-running job finished. Declared by the
  implementer, datagenerator, and debugger.
- **Two implementations, one surface** (the standard pattern here): on Claude the
  SDK executes Bash/BashOutput/KillShell natively (they are SDK built-ins — we
  now enable the two companions we had omitted from `NATIVE_TOOLS`); on
  ollama/vllm/openrouter the framework provides them via `_BashSession` +
  `_make_bash_tool`/`_make_bashoutput_tool`/`_make_killshell_tool`.
- **Safety:** framework-backgrounded children stay in the run's process group
  (no `start_new_session`), so the watchdog group-kill reaches them; the bg pid
  is best-effort registered in `governor_pids.jsonl`; `KillShell` is the only
  per-delegation teardown.
- **Where:** `backends/claude.py` (`NATIVE_TOOLS`), `backends/openai_compatible.py`
  (`_BashSession`, the three factories, `_native_tool_map`).
- **Status:** done. Supersedes the earlier `WaitForProcess` stopgap.

### Sandbox study-root anchor (`F3DASM_STUDY_ROOT`)
- **What:** the reproduction gate, `CheckDeliverable`, `RunPipelineCell`, and the
  scratch tool run against a *temp copy* of the ledger, so the store path has no
  relationship to the study repo. They now also inject `F3DASM_STUDY_ROOT` (a
  read-only anchor to the real study root) so a pillar cell can locate non-ledger
  repo resources (e.g. `bo/cei_core.py` for a surrogate self-check) deterministically
  instead of hand-rolling multi-candidate path search. Store isolation is unchanged —
  only the store is a copy; the study root is read-only reference code. The three
  duplicated sandbox-env blocks are unified in one `sandbox_env()` helper.
- **Where:** `notebook_exec.py` `sandbox_env`; call sites in `nodes/strategizer.py`
  (`_reproduction_gate`) and `nodes/tools/routing.py` (`RunPipelineCell`, scratch).
- **Status:** telemetry/ergonomics, not a new cap. Run 20260705T181941 friction.

### Output-column guidance fix
- **What:** notebook guidance requires naming the objective column EXPLICITLY. The
  earlier "first non-provenance output" auto-detect was unsafe: `output_names` is
  sorted, so with multiple outputs a constraint flag (e.g. `coilable`) sorts before
  the objective and gets silently picked (audit 20260624T021359). If derived, read
  `run_config['evaluator_output_names'][0]`, not column order.
- **Where:** `notebook_exec.py`. **Status:** done.

## C. Workers & ground truth

### Metered oracle (get_evaluator) + canonical ledger
- **What:** the one door to the registered ground-truth oracle; every evaluation is
  written to the canonical store with provenance, under a file lock.
- **Where:** `instrumented.py`. **Tools (worker scratch):** `RunScratch`, `ReportEvals`.
- **Status:** core.

### Design namespaces — multiple oracles + ledgers per run (#20, Axis 3)
- **What:** a run may carry more than one oracle, one per design parametrization the
  agent invents (open design-space discovery). `get_evaluator(namespace=None)` resolves
  `run_config["oracles"][namespace]` — its own oracle + its own isolated, protected
  store; `Delegate(..., namespace="…")` scopes a worker to a namespace (injected as
  `F3DASM_NAMESPACE`, so the agent's call site stays `get_evaluator()`); the
  datagenerator registers a namespace oracle without disturbing the canonical default.
  ADDITIVE: `namespace=None` is byte-for-byte the single-study path. Comparable-by-
  construction (a new design reuses the fixed objective evaluator; see
  `OPEN_DESIGN_SPACE_FRAMEWORK.md`).
- **Where:** `instrumented.py` (`get_evaluator`, `_effective_oracle_config`),
  `agent_runtime.py` (`register_evaluator_entrypoint(namespace=…)`), `backends/base.py`
  + `backends/claude.py` (`set_namespace`/`F3DASM_NAMESPACE`), `graph_state.py`
  (`Delegation.namespace`), `routing.py` (`Delegate` + registration handoff).
- **Report-time provenance:** `LedgerBreakdown()` (strategizer tool) shows per-experiment
  / per-delegation ledgered eval counts read live from the stores
  (`instrumented.ledger_breakdown`), so a writeup DERIVES counts from the ledger instead
  of hardcoding stale plan numbers (run 20260628T001710 hardcoded 70 polar evals; the
  ledger held 90 → UNGATED). It also reads `eval_budget` from run_config and prints
  `spent of budget — N remaining`, so the agent READS that number rather than hand-
  computing it and flipping spent↔remaining (run 20260628T130525 asserted "200 remain"
  with 200 spent of 300 → UNGATED). Read-only; spends no eval budget.
- **Multi-experiment load idiom:** `a3dasm.load_experiments()` (`instrumented.
  load_experiments`) loads every experiment store of a run as `{name: ExperimentData}`
  (default + each design experiment, at their nested paths). A namespaced run has N
  stores and no namespace column, so the single-study `from_file` idiom silently loads
  only the default; this is the one call pipeline.ipynb uses to load them all. Wired
  into the deliverable spec (`notebook_exec.py`) and the injected paths block
  (`agent_prompts.py`).
- **Tools:** `LedgerBreakdown`.
- **Status:** plumbing complete (branch `exp/open-design-space`); gated on the 2D
  experiment before the baseline study adopts it.

### Literature reviewer
- **What:** a specialist agent that searches papers (arXiv / Semantic Scholar) and
  returns findings; degrades to lexical search without the heavy extras.
- **Where:** `agents/literature.py`, backends. **Status:** core.

### Delegation-ID allocation fix (D002)
- **What:** delegation IDs are allocated *after* the milestone gate, so a blocked
  attempt no longer burns an ID (IDs stay contiguous).
- **Where:** `nodes/tools/routing.py`. **Status:** done.

## D. Resource governance (this is the big recent addition)

### Soft eval-budget nudge
- **What:** when the shared ledger crosses 80/100/150% of the eval budget, the
  *offender* (the running campaign) is nudged via its own output — capped at one per
  band. **Soft: never stops the campaign** (the eval budget is the agent's call).
- **Where:** `instrumented.py` `_flush` governor; budget plumbed via `agent_runtime.py`.
- **Config:** `eval_budget` (config.yaml / `F3DASM_EVAL_BUDGET`). **Status:** done.

### Hard memory cap (the one hard boundary)
- **What:** a 5-second watchman sums each delegation's process-tree **resident (RSS)**
  memory and kills the tree if it exceeds the cap. Real-usage based; verifies the
  process is still ours (start-time match) before killing, so a recycled PID is never
  hit. Per-delegation, absolute (not a share of system RAM), does not sum across
  delegations.
- **Where:** `studies/.../run.py` `_memory_watcher`; `watchdog_cleanup.py`
  (`check_memory_and_kill`, `_owned_pids`); `resource_backend.py`.
- **Config:** `mem_cap` (config.yaml / `F3DASM_MEM_CAP`); default 4 GiB. On SLURM set
  below the job's `--mem`. **Status:** done (cgroup-native HPC backend = future seam).

### Resource backend (OS abstraction)
- **What:** one interface (`set_self_limit` / `read_rss` / `kill` / `proc_start_time`)
  so memory/kill OS-specifics live in one place; psutil impl + stdlib fallback.
- **Where:** `resource_backend.py`. **Status:** done (Linux cgroup backend = future).

### Per-delegation resource telemetry
- **What:** `GetStatus` shows a delegation's eval count, current RSS, and **peak
  RSS** (the high-water across the watcher's ticks), so the strategizer can see a
  fat or fattening campaign (and `Confer` the implementer).
- **Where:** `nodes/tools/routing.py`, `watchdog_cleanup.py`
  `delegation_rss` / `delegation_peak_rss`.
- **Status:** done.

### Resource AWARENESS (telemetry, NOT enforcement)
- **What:** primes agents to be efficient with the things models ignore — time,
  RAM, disk, parallelism width — via two surfaces:
  - **Static envelope at delegation start:** a `<resources>`-style stanza in the
    worker/strategizer preamble — `~N CPU cores · RAM cap X GB (HARD — exceed it
    and your process is killed; stream/cache) · disk free Y GB · parallelize up to
    ~N ways, sized to RAM`.
  - **Measured peak RAM in the KPI footer:** `peak RAM (this delegation): Z GB of
    X GB hard cap`, the watcher's high-water — so memory cost travels with the
    result like wall-time already does.
- **Footprint (by design):** peak RAM rides the memory watcher's existing 5s poll
  (one `max()` per tick — no new poll/thread/I/O); the envelope is one
  `os.cpu_count()` + one `shutil.disk_usage` (O(1) `statvfs`, **never** a recursive
  `du`); the per-eval hot path is untouched (no per-eval RSS/disk stamping).
- **Where:** `watchdog_cleanup.py` `resource_envelope` / `delegation_peak_rss`
  (high-water recorded in `check_memory_and_kill`); `agent_runtime.py`
  `_resource_stanza`; `agent_prompts.py` `{resources}` placeholder;
  `instrumented.py` `delegation_footer` peak-RAM line.
- **Status:** awareness only — the hard memory cap stays the one enforced boundary.

### Per-delegation ledger KPIs auto-appended to the report
- **What:** when a delegation completes, a KPI footer is appended to the result
  the strategizer auto-receives (GetStatus/Confer/Done) — per-eval wall-time
  (median, max), this delegation's total eval wall-time, the ledger total, and —
  when a wall budget is set — the time remaining (telemetry, not a hard stop), so
  the median is actionable (≈ remaining / median = sims still affordable).
  Measured from the rows the delegation actually wrote, so budget planning runs
  on observed sim cost instead of an a priori per-sim estimate. Auto-delivered,
  not on-demand. Plain measurements only — interpretation is the strategizer's.
- **Where:** `instrumented.py` `RunStateSummary.{wall_per_delegation,
  delegation_footer}`; appended in `nodes/tools/routing.py`.
- **Status:** done.

### Framework-owned local LLM on a SLURM GPU node (vLLM)
- **What:** instead of a hosted API, the framework can own the LLM *behind the
  nodes* on a separate SLURM GPU allocation: it sizes the allocation from the
  checkpoint's own published metadata (parameter count + dtype, fetched
  weights-free — local HF cache first, then a single `requests` GET of the HF
  Hub model-info JSON; no `huggingface_hub`/`transformers` dependency), so a
  full HF id or a short alias gets correctly-sized GPUs/mem with **zero user
  config**. GPU-count derivation uses a built-in per-GPU VRAM table keyed by the
  cluster's exact Slurm gres names (the Oscar/Brown-CCV inventory; hardware
  drifts far slower than model releases), defaulting to `l40s` (48 GB,
  schedulable on the general `gpu` partition) when the study names no GPU, and
  emits a type-qualified `--gres=gpu:<name>:<n>` so Slurm grants the exact card
  the size was computed for. VRAM is sized as weight-bytes(serve quant) +
  **context-aware KV cache** (read from the model config — layer count, KV
  heads/head-dim, and the sliding/global attention split, at the served context
  length and `--kv-cache-dtype`; a flat multiplier is the fallback only when the
  config lacks those fields) + a small fixed overhead. The serve quant is a
  `runtime` knob whose default is **GPU-aware**: FP8 only on FP8-capable cards
  (Ada/Hopper/Blackwell — `l40s`/`h100`/`nvidia_rtx_pro_6000_blackwell`/
  `nvidia_b200`), else a safe BF16/Q4 path, so a zero-config Ampere/Turing run
  never asks for FP8 it cannot serve; an explicit `llm_quantization=fp8` still
  wins. This lets the power model `gemma-4-31b` (30.7B) fit a single 48 GB L40S
  at FP8 weights + FP8 KV (~44 GB) while keeping its full 256K context; the same
  quant is applied to the `vllm serve` launch so sizing and serve agree. A tiny
  built-in alias table maps `gemma-4`/`gemma-4-e4b` → the cheap default and
  `gemma-4-31b` → the max-power-for-48 GB variant. Optional, purely-override layers,
  most-explicit-wins: `llm_slurm` config fields > metadata-derived sizing >
  basename-matched family *serve-hints* (context length etc. that metadata
  can't publish) > conservative default. Metadata unavailable (offline, gated
  repo, unknown GPU) degrades to a loud warning + conservative default — never
  raises, never silently mis-sizes. Then submits a `vllm serve` job (reusing f3dasm's
  `SlurmCluster` + the plain `sbatch` idiom — a persistent server is NOT routed
  through the eval-oriented `Pipeline`/`SlurmExecutor`), waits for the granted
  node, polls `/v1/models` past the cold model load, publishes `VLLM_BASE_URL`
  so the existing vLLM adapter reaches it over the cluster network, and
  scancels the job on EVERY exit path (normal close, crash, and the watchdog's
  `os._exit` hard-kill via `reap_run_serve_job`). A build-time, leading-order
  throughput bound (decode is memory-bandwidth-bound; GPU/model-size/dtype/
  tensor-parallel are all config-known) warns loudly when a config is likely to
  choke — a nudge at config time, never a block. Same physics the token
  telemetry measures after the fact (parity). Phase 1: one allocation for the
  run's lifetime; Phase 2 (walltime chaining + a stable local proxy) is designed
  but deferred.
- **Config (all optional overrides):** `llm_slurm:` block — `enabled`
  (default off), `model`, `aliases` (short-name→canonical-HF-id map; wins over
  the built-in aliases), resource overrides
  (`gres`/`mem`/`time`/`cpus_per_task`/`vllm_args`), sizing/throughput inputs
  (`gpu_model`/`params_b`/`dtype_bytes`/`tensor_parallel`), `queue_timeout`/
  `serve_timeout`, and a nested `cluster:` (`partition`/`account`/`env_setup`/
  `env_vars`/`runner`). Metadata knobs live in the `runtime:` block via
  `settings.get_*`: `llm_metadata_fetch` (bool, default on — off = local-cache
  only, no network), `llm_metadata_timeout_s` (float, default 8), and
  `llm_quantization` (str; unset → GPU-aware default: FP8 on FP8-capable cards
  else BF16/Q4. Set `fp8`/`bf16`/`fp16`/`awq`/`gptq`/`q4`/…, or `auto` for the
  checkpoint's native dtype) — the serve dtype used for both VRAM sizing and the
  `vllm serve --quantization` flag; an explicit value wins over the GPU-aware
  default. Requires
  `backend: vllm`. Disabled → hosted-API runs are unchanged.
- **Where:** `agentic/slurm_llm.py` (aliases, metadata fetch, VRAM sizing,
  serve-hints, resolve, render/submit, wait, throughput bound, teardown,
  reaper); `agent_runtime.py`
  (`_maybe_start_slurm_llm` + the teardown `finally` in `execute()`);
  `studies/*/run.py` watchdogs (serve-job reap). Reuses
  `pipeline/resources.py` (`SlurmCluster`/`SlurmResources`) unchanged.
- **Status:** Phase 1 done, headless-tested; validate on a real GPU cluster
  before making it a default anywhere (greenfield + cluster-specific).

## E. Runtime safety

### Wall-clock watchdog + recursive reap (#11/#14)
- **What:** a hard wall-clock timer force-exits a stalled run; on exit it recursively
  kills every campaign process tree — including detached/new-session ones that a
  process-group kill misses.
- **Where:** `studies/.../run.py` `_watchdog`; `watchdog_cleanup.py` `reap_governor_pids`.
- **Config:** watchdog = 2× the run's time budget. Operational kill-switch
  `F3DASM_DISABLE_WATCHDOG=1` turns the wall-clock force-exit OFF (the memory-cap
  watcher stays on) — for long supervised runs. **Status:** done.

### Synthetic watchdog retrospective (#12)
- **What:** a watchdog kill leaves a labelled post-mortem so the analysis protocol
  isn't blind.
- **Where:** `watchdog_cleanup.py` `write_watchdog_retrospective`. **Status:** done.

### KB (handbook) entries
- **What:** curated knowledge the agents consult (incl. running on SLURM, pipeline
  patterns). **Where:** `knowledge/entries/`. **Status:** core.
- **Injected menu:** an audience-filtered, one-line-per-entry MENU
  (`KnowledgeBase.menu(audience)`) is injected into every agent's system prompt
  (`agent_runtime._kb_menu` → the `{knowledge}` placeholder in both preambles), so
  an agent always SEES the latent chapters it can pull — the same way it always
  sees its tool list — instead of only discovering one if it already thought to
  call `ConsultHandbook`. The descriptor is the entry `title`, capped to one terse
  line by a ≤100-char invariant (`test_knowledge_base.py`).

---

## Tools (every one must be documented above; the test enforces it)

| Tool | Feature |
|---|---|
| `Delegate` | Delegation (dynamically injected) |
| `GetStatus` · `Wait` · `FollowUp` · `Confer` · `ReportEvals` | Delegation + messaging + telemetry |
| `AddPipelineCell` · `AddPipelineMarkdownCell` · `EditPipelineCell` · `DeletePipelineCell` · `ShowNotebook` · `WriteDeliverable` · `CheckDeliverable` | Notebook authoring + reproduction gate |
| `RunPipelineCell` | Per-cell notebook debugger (#13) |
| `RunScratch` | Worker scratch execution against a ledger copy |
| `WriteNote` · `ReadNote` | Agent scratch notes |
| `RecallStore` · `QueryStore` | Canonical evaluation-store read (declaration-gated; shared verbatim across node types — strategizer, workers, and the critic). `QueryStore` accepts `where=` (a pandas `query()` expression over the joined inputs+outputs frame — compound feasibility predicates + arithmetic on input columns in one call) and `limit=` (lifts the 20-row default listing cap); bad `where` returns a column-listing ERROR, never raises (spec 09) |
| `HypothesisPropose` · `HypothesisUpdate` · `HypothesisList` · `HypothesisGet` · `LinkFalsificationAttempt` | Hypothesis ledger — read (List/Get) is declaration-gated to any node; mutate (Propose/Update/Link) is strategizer-only |
| `MilestoneList` · `MilestonePropose` · `MilestoneComplete` · `MilestoneSkip` | Process milestones (strategizer-only) |
| `BashOutput` · `KillShell` | Bash companions: poll / stop a backgrounded shell (#24) |
| `Read` · `Write` · `Edit` · `Bash` · `Glob` · `Grep` | Workspace file/shell primitives |
| `Done` | Close the run for the gate |
