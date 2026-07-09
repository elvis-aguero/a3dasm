# Spec 05 — KB entry: running an agentic study on SLURM

**Backlog #5.** Priority: lowest (docs) unless cluster scaling is near-term.
Status: spec. Deliverable is primarily a **KB entry + doc**, but the evidence
surfaced real correctness hazards that the KB must call out (and that may need
small code fixes first).

## Goal
Document how the agentic layer maps onto f3dasm's existing SLURM machinery, and
the **invariants that must hold** for the canonical ledger to stay correct under
cluster parallelism.

## Primary evidence — f3dasm already has the cluster machinery
- `Pipeline.run(mode="local"|"slurm", cluster=SlurmCluster, project_job=…,
  rootdir=…)` — `pipeline/pipeline.py:248-301`. `"slurm"` submits via `sbatch`;
  the orchestrator self-resubmits with step counters.
- Per-step resources: `SlurmResources` dataclass (`pipeline/resources.py:25-55`:
  `time, mem, cpus_per_task, nodes, max_array_size, max_concurrent, extra_sbatch`),
  attached via `Step.resources` (`pipeline/pipeline.py:77`) → rendered to
  `#SBATCH` directives.
- Lazy resume is **mode-agnostic**: `get_open_job()` skips non-OPEN samples
  (`experimentdata.py:1455-1480`); `evaluate_cluster` loops `get_open_job()` under
  lock until none remain (`datagenerator.py:228-300`). So a resumed campaign
  re-evaluates only OPEN rows on the cluster exactly as locally.

## Primary evidence — the hazards (what the KB must warn about)
The agentic runtime today is **local-only** (KB 0009 line 9: "our agentic runs
are local — ignore the cluster/Hydra machinery"; `container_runner.py` is
Docker/Colima only; `agent_runtime.py` never calls `pipeline.run`). The ledger
invariants assume a single host. Under SLURM they can break:

| Hazard | Evidence | Why it breaks |
|---|---|---|
| **FileLock is single-host advisory** | `instrumented.py:191` `with FileLock(self.lock_path)`; default `store_dir/experiment_data/.lock` (`:110`) | If the store is on per-node `/tmp` or `/scratch`, two nodes take *different* locks → concurrent `get_evaluator()` writes race → duplicate/corrupt ledger rows. **Requires the canonical store on a shared FS** (and a lock the shared FS honours). |
| **Env vars are process-local** | `F3DASM_RUN_CONFIG`/`F3DASM_DELEGATION_ID` set per-session (`backends/claude.py:411-417`); `_load_run_config` reads them (`instrumented.py`) | A SLURM job inherits env at spawn; if a delegation runs on a compute node without these exported, resolution falls back to cwd walk-up — which fails off the study tree. |
| **cwd walk-up assumption** | `_load_run_config` walk-up + `_resolve_delegation_id` from cwd name (`instrumented.py:474-489`) | A compute node may start in `/scratch/slurm.XXXX/`; the now-added `F3DASM_RUN_CONFIG` (spec done) mitigates, but must be exported into the job env. |
| **sys.path restoration is Pipeline-only** | `pipeline/run_step.py:100-112` restores `sys.path` from `.sys_path.json` | Agentic delegations don't use that handoff; a delegation subprocess on a compute node won't have `study_dir` importable. |
| **run_config.json reachability** | written to `run_dir/debug/run_config.json` (`agent_runtime.py:111`); SLURM orchestrator `count_open` reads `job_dir` (`pipeline/executors/slurm.py:602`) | Must be on shared FS readable from the submission host AND all compute nodes. |

## Design (KB entry + the invariants; small code changes flagged separately)
The KB entry (`knowledge/entries/00NN-running-on-slurm.md`) must state:
1. **The shared-FS invariant (load-bearing):** the canonical store, its `.lock`,
   and `run_config.json` MUST live on a filesystem shared + lock-honouring across
   all nodes. This is the single most important rule; the ledger's correctness
   depends on it. Cite `instrumented.py:191`.
2. **Export the resolution env vars** (`F3DASM_CANONICAL_STORE`,
   `F3DASM_RUN_CONFIG`, `F3DASM_DELEGATION_ID`) into the SLURM job env so
   `get_evaluator()` resolves without cwd luck. Use `SlurmCluster.env_vars`.
3. **Lazy resume works on-cluster** — re-running adds zero evals (cite the
   `get_open_job` loop); the reproduction gate's contract is unchanged.
4. **Resource declaration** — map study cost to `SlurmResources`
   (`time/mem/cpus_per_task`, `max_concurrent`), with an example `Step.resources`.
5. **What is NOT wired today** — the agentic runtime does not submit itself to
   SLURM; only the *evaluation* workload (the oracle/campaign run inside
   `pipeline.py`) would; the agent loop still runs on one host. Be explicit so a
   reader doesn't expect agent-on-cluster.

## Code changes this may require first (flag, don't assume)
- A guard/assert that, when a cluster mode is requested, the store + lock path
  resolve to a shared FS (fail fast with a clear message) — TDD: a test that a
  non-shared lock path raises.
- Ensure `F3DASM_RUN_CONFIG` etc. are surfaced for inclusion in
  `SlurmCluster.env_vars` (they're thread-locals today).

## TDD plan
1. `test_get_open_job_skips_finished_under_concurrent_readers` — two readers over
   one store skip FINISHED rows; no double-eval (proves lazy-resume invariant).
2. `test_filelock_path_is_explicit_and_configurable` — `lock_path` from
   `run_config` is honoured (so it can be pointed at shared FS); cite `:110`.
3. (if guard added) `test_cluster_mode_rejects_nonshared_store` — clear failure.
4. Doc test: the KB entry exists and the shared-FS invariant string is present
   (mirrors `test_charter.py`'s wording-pin approach for a load-bearing doc).

## Out of scope
Running the **agent loop itself** on SLURM (only the eval workload). Any new
executor. Hydra/multirun integration.

## Done when
The KB entry exists with the shared-FS invariant front-and-centre, the env-var
export recipe, a `SlurmResources` example, and the explicit "agent loop stays
local" note; lazy-resume + lock-path tests pass.
