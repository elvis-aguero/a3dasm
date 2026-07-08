---
id: pipeline-building-patterns
title: Building an f3dasm Pipeline (patterns from real studies)
tags: [pipeline, step, block, loop, datagenerator, sampler, experimentdata, domain, deliverable]
audience: [strategizer, implementer]
---
How to compose a data-driven solution as an f3dasm `Pipeline` — distilled from
the bessagroup `l2co_experiments` study suite. (Those run on SLURM via Hydra;
our agentic runs are **local** — ignore the cluster/Hydra machinery and keep the
composition idioms below. Build the pipeline in code, not YAML.)

## Canonical skeleton: LOAD-OR-CREATE → run → analyze
A pipeline is an ordered list of `Step`s. The deliverable shape is *load the
ledger if it exists else build the DoE, evaluate every OPEN row through the
oracle, derive the headline from the ledger*. The **run step is
`get_evaluator()`** — the one door to the registered ground-truth oracle. You do
NOT write your own `Oracle(DataGenerator)`; the datagenerator agent already built
and registered it, and `get_evaluator()` returns it as a ready Step block that
meters every call into the run's canonical store with provenance.

The create step is **LOAD-OR-CREATE** — this is what makes ONE script both
regenerate from scratch AND reproduce lazily. It is NOT two scripts and NOT a
contradiction: on a fresh machine with the shipped ledger it loads it (and the
run step skips every FINISHED row → zero new oracle evals, reproduces in
minutes); on an empty store the SAME code builds the DoE and the run step
evaluates it for real.

```python
import os
from f3dasm import Pipeline, Step, ExperimentData, create_sampler
from a3dasm import get_evaluator
from f3dasm.design import Domain

def create_experimentdata(project_dir):                 # Step 1: LOAD-OR-CREATE
    store = os.environ.get("F3DASM_CANONICAL_STORE", project_dir)
    try:                                                # shipped ledger present?
        if len(ExperimentData.from_file(project_dir=store).to_pandas()[1]) > 0:
            ExperimentData.from_file(project_dir=store).store(project_dir)
            return                                      # → run step skips FINISHED: 0 new evals
    except Exception:
        pass                                            # empty store → build it from scratch
    domain = Domain()
    domain.add_float("x0", -5.0, 5.0); domain.add_float("x1", -5.0, 5.0)
    domain.add_output("f")                              # declare outputs up front
    data = ExperimentData(domain=domain)
    sampler = create_sampler("random_sampler", seed=0)
    sampler.arm(data=data)                              # ARM before CALL (two-phase API)
    data = sampler.call(data=data, n_samples=50)
    data.store(project_dir)

def analyze(project_dir):                               # Step 3: derive headline FROM the ledger
    _, out = ExperimentData.from_file(project_dir=project_dir).to_pandas()
    print(f"REPRODUCED: {float(out['f'].min())}")       # derive it; NEVER hardcode

Pipeline(name="solve", steps=[
    Step(name="create",  block=create_experimentdata),         # load-or-create
    Step(name="run",     block=get_evaluator(), parallel=True),# ← the ONE oracle door (lazy)
    Step(name="analyze", block=analyze),                       # self-asserting headline
]).run(mode="local", project_job="run_001")
```

`get_evaluator()` must be called from inside your delegation workspace (it reads
the run config to resolve the registered source and the canonical store). Every
row it evaluates is ledgered automatically — derive your headline from that
canonical store (what the runtime reproduces from), never from hardcoded
numbers or a step's local scratch copy. The runtime reproduces by running THIS
script against the shipped ledger and asserting zero new evals — the
load-or-create create step is exactly what passes it. Derive your `REPRODUCED:`
headline from the ledger too: for a CONSTRAINED objective it is the best
FEASIBLE value (e.g. `out[out['coilable']==1]['f'].max()`), NOT necessarily an
objective extremum. The critic checks the headline's provenance; the runtime
does not machine-match the value, so a constrained optimum is a valid headline.

## The composition API (the actual idioms)
- **`Step(block=, name=, parallel=, kwargs=)`** wraps a callable, a `Block`, or a
  `DataGenerator`. A plain callable receives `project_dir` auto-injected.
- **`block_a >> block_b`** chains blocks into ONE step (run sequentially, shared
  state) — use for post-processing that builds on the previous block.
- **`Loop(n_iterations=N, steps=[...])`** repeats inner steps; carry state across
  iterations on the on-disk `ExperimentData` (`data.mark_all("open")` at the end
  of a round so the next iteration recomputes) — the active-learning / BO-rounds
  shape.
- **`Block` subclass** for ED→ED post-processing: `def call(self, data, **kw) -> ExperimentData`.
- **`DataGenerator` subclass** ONLY for your OWN off-ledger predictors (a fitted
  surrogate, a stub): heavy state in `__init__`, per-row work in `execute`. The
  ground-truth oracle is NEVER hand-written here — it is always `get_evaluator()`.
- **Stub a block** with a trivial body or fake return to draft a *working*
  skeleton first (the baseline-to-beat), then swap real blocks in. Swapping a
  block (a different sampler, surrogate, or optimizer) is how you test a
  hypothesis; the oracle run step stays `get_evaluator()` throughout.

## Domain / oracle / sampler
- `Domain()` + `add_float/add_int/add_category(name, ...)` for inputs;
  `add_output(name)` for every output column (custom objects:
  `add_output(name, to_disk=True, store_function=, load_function=)`). Outputs are
  **untyped** — there is **no `add_float_output`** (a common wrong guess); always
  `add_output(name)`.
- **Imports (a common slip):** `create_sampler`, `Pipeline`, `Step`,
  `ExperimentData` are **top-level** — `from f3dasm import create_sampler`.
  **Only `Domain` lives in `f3dasm.design`** (`from f3dasm.design import Domain`);
  `get_evaluator` is `from a3dasm import get_evaluator`. Importing
  `create_sampler` from `f3dasm.design` fails.
- Wrap an existing fn as a generator: `datagenerator(output_names="y")(fn)`.
- Samplers (`"random_sampler"`, `"latin_sampler"`, `"sobol_sampler"`,
  `"grid_sampler"` — the preferred names; bare `"random"` etc. are deprecated
  aliases): always `arm(data=)` then `call(data=, n_samples=)`.

## ExperimentData (the ledger)
The **one canonical ledger** is written by `get_evaluator()` — every true-oracle
row lands there, stamped with a `_delegation_id`, across all delegations in the
run. That store is the single source of truth for the eval count and the
headline. Read it with `ExperimentData.from_file(project_dir)` and query via
`.to_pandas()` / `.select(...)` / `.select_with_status("finished")`; the headline
is *derived* from those rows, never hardcoded. You may `.store(...)` your own
scratch ExperimentData (a surrogate's predictions, intermediate DoEs) wherever
you like — just don't confuse it with the canonical ledger.

## Pipeline archetypes ("use when")
1. **Sampling sweep** (create→evaluate→collect): a labelled dataset over a space.
2. **Surrogate-fit sweep** (grid over model configs → fit → score): compare surrogates.
3. **Iterative `Loop`** (create once, then loop run+update): the design depends on
   prior rounds — BO, active learning, on-policy RL.
4. **Independent-per-row** (one fit per row, parallel) + a joined evaluation.
5. **Baseline + method** spliced together: ship a method and its apples-to-apples
   baseline in one pipeline.
6. **Resume tail-only**: re-derive results from an existing run without recomputing.

## Anti-patterns the examples warn against
- Don't put heavy loads in `execute` — they run per-row; load once in `__init__`.
- Don't forget to `arm()` before `call()` — it's a required two-phase API.
- Don't carry `Loop` state in memory — persist to the ED, `mark_all("open")`.
- Don't hardcode the headline — derive it from the canonical store.
- Don't ship a read-only analysis script whose run step is a comment
  (`# in production this would call get_evaluator()`), and don't make a create
  step that ALWAYS rebuilds the DoE (it re-evaluates on a re-run → not lazy).
  Use the **load-or-create** create step above: it loads the shipped ledger if
  present (run step skips FINISHED rows → zero new evals, reproduces fast) and
  builds the DoE only on an empty store. One script, both behaviours — "lazy"
  and "regenerable" are the same code path, not opposites. A real
  `get_evaluator()` block costs nothing when the ledger is full but is what
  makes the script regenerate, not just summarise. The runtime executes it
  against the shipped ledger and asserts zero new evals.
- The oracle run step is ALWAYS `get_evaluator()` — never a hand-written
  `DataGenerator`, never the raw evaluator, never a redirected store. That one
  door is what makes the result ledgered and reproducible; see
  [[evaluate-through-get-evaluator]] and [[surrogates-are-off-ledger]].
- **Never call `data.store()` after `evaluator.call()`**. The
  `InstrumentedDataGenerator` behind `get_evaluator()` already writes FINISHED
  rows into the canonical store inside each `flush()`. Calling `data.store()`
  on the *local* `data` object afterwards overwrites those FINISHED rows with
  IN_PROGRESS — corrupting job statuses silently. The canonical store is
  write-only via `get_evaluator()`. If you need the updated outputs locally,
  reload: `data = ExperimentData.from_file(project_dir=canonical_store)`.
- **Never import a library without checking it is installed**. Heavy optional
  packages (`torch`, `botorch`, `gpytorch`, `jax`) are NOT in the base
  environment. Unconditionally importing them breaks the gate with a
  `ModuleNotFoundError` and forces a correction delegation. Check first:
  ```python
  import importlib.util
  if importlib.util.find_spec("torch") is None:
      # fall back to sklearn / scipy
  ```
  or use `try/except ImportError`. Never write `import torch` at the top of a
  notebook cell without a fallback — the gate runs in the same environment and
  will fail if the package is absent.

## When the deliverable is a notebook (`pipeline.ipynb`)
In notebook mode the single deliverable is `pipeline.ipynb` — the writeup AND the
lazily-reproducible recipe in one. Build it as valid nbformat v4 (never
hand-write fragile JSON). It is a **scientific narrative**: its spine is the
Popperian loop, its body mirrors the four f3dasm pillars, and each code cell
carries `metadata.name` = its pillar so the structure is machine-checkable.

Cell order:
1. `# Problem & objective` (md) — question, min/max, success criterion.
2. `## Hypotheses` (md) — registered hypotheses + falsifiable predictions.
3. WHY-explainer (md) + code `name="doe"` — Domain + sampler, **load-or-create**.
4. WHY-explainer (md) + code `name="data_generation"` — `get_evaluator()` only (lazy).
5. WHY-explainer (md) + code `name="ml"` — fit the surrogate.
6. WHY-explainer (md) + code `name="optimization"` — acquisition / BO loop.
7. `## Verdict & result` (md) + code `name="analysis"` — per-hypothesis
   SUPPORTED/FALSIFIED + WHY; derive the headline from the ledger and print
   `REPRODUCED: <value>`.

The four pillar cells are ALWAYS present; a pillar you did not run stays present
with its explainer stating "NOT executed (budget)" — never silently drop one.
Every WHY-explainer justifies the choice (cite the literature). The lazy +
zero-new-eval reproduction contract is binding: the runtime executes the
notebook against the shipped ledger and asserts zero new oracle evals + the
grounded `REPRODUCED:` line.
