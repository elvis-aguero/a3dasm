---
id: running-a-study-on-slurm
title: Running a study on SLURM (the deliverable pipeline, not the agent graph)
tags: [slurm, cluster, hpc, pipeline, parallel, resources, get_evaluator, canonical-store, filelock, resume]
audience: [strategizer, implementer]
---
SLURM runs the DELIVERABLE pipeline, not the agentic orchestration. The
strategizer/critic graph runs LOCAL on one host, and by default each node's
LLM is a hosted API (the `claude` CLI). What maps onto SLURM is the f3dasm
`Pipeline` your deliverable builds: the DoE → `get_evaluator()` → analyze
recipe (see [[pipeline-building-patterns]]). You change the `.run(...)` call and
per-step resources; the composition is identical to local mode.

> **Opt-in exception — the agent's *own* LLM served on a SLURM GPU node.** The
> framework can own a local model (vLLM) on a separate GPU allocation instead
> of a hosted API: it submits the `vllm serve` job, waits for the node + a ready
> server, points the backend at it over the cluster network, and scancels it on
> every exit path (normal/crash/watchdog). This is the LLM *behind* the nodes,
> not the deliverable pipeline. Enable it with the `llm_slurm:` config block
> (see the feature catalog); disabled by default, so the claim above holds
> unless you turn it on.

## The API (all top-level f3dasm exports)
```python
from f3dasm import Pipeline, Step, Loop, SlurmCluster, SlurmResources

Pipeline(name="solve", steps=[
    Step("create", block=create_experimentdata),                 # load-or-create
    Step("run", block=get_evaluator(), parallel=True,            # ← fans out as a job array
         resources=SlurmResources(time="02:00:00", mem="8G",
                                  cpus_per_task=4, max_array_size=200)),
    Step("analyze", block=analyze),
], orchestrator_resources=SlurmResources(time="00:30:00")).run(
    mode="slurm",
    cluster=SlurmCluster(partition="batch", account="my_acct",
                         runner="uv run",            # how python is launched
                         env_setup=["module load python"]),  # pre-run shell
    rootdir="/scratch/user",
)
```
In SLURM mode a single **self-resubmitting orchestrator** job walks the pipeline,
submitting ONE step (or one `Loop` iteration) at a time. `orchestrator_resources`
is tiny by default (10 min / 1 GB / 1 CPU) — it only issues `sbatch` calls.

## What `parallel=True` does
The run step fans out as a **SLURM job array** (`run_step.py` → `cluster_array`
mode): array task *k* processes `open_experiments[k::max_array_size]`, so the open
ledger rows are striped across tasks. Tune with `SlurmResources.max_array_size`
(cap 900) and `max_concurrent` (default 64). A non-parallel step on the cluster
runs as one job with file-lock serialization instead.

## Why the canonical store still works under parallelism
`get_evaluator()` writes FINISHED rows into the ONE canonical store with a
**FileLock** (`instrumented.py`), so concurrent array tasks serialize their writes
— no corruption. And because the create step is **load-or-create** and the run
step **skips FINISHED rows**, a resubmitted orchestrator or a re-run array job
recomputes nothing already done: the lazy resume that makes local runs cheap is
exactly what makes cluster array jobs and requeues safe. Still derive the headline
from the store, never hardcode (see [[evaluate-through-get-evaluator]]); still
never `data.store()` after `evaluator.call()` (it would overwrite FINISHED rows).

## What changes vs local mode (checklist)
- `.run(mode="slurm", cluster=SlurmCluster(...), rootdir=...)` instead of
  `mode="local"`.
- Per-step `resources=SlurmResources(...)` on the heavy run step; let the
  orchestrator default stand.
- `SlurmCluster.env_setup` / `env_vars` / `runner` must reproduce your local env
  on the compute node (module loads, `uv run`); a missing dep there fails the job,
  not the gate.
- Optional heavy deps (`torch`, `botorch`) must be guard-imported regardless —
  the array task hits the same `ModuleNotFoundError` trap as the gate.
- Everything else — Domain, sampler arm/call, `get_evaluator()`, Loop state on the
  ED, the four notebook pillars — is unchanged.
