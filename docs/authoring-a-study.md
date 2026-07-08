# Authoring a study folder — the agentic-f3dasm contract

A **study folder** is the single input to a run: `AgenticRun(study_dir=...)`.
The agent reads the folder, designs and runs experiments, and writes its results
back into the same folder. This page is the **contract** — the exact files the
runtime reads, the evaluator interface it expects, and the artifacts it
produces — so you can prepare a folder that runs correctly the first time.

For framework internals (architecture, tools, backends, HPC) see
[`../../README-agentic.md`](../../README-agentic.md); this page is the narrow
"what must my folder contain" reference.

---

## Minimal study folder

```
my_study/
  PROBLEM_STATEMENT.md      # REQUIRED — the brief the agent works from
  config.yaml               # recommended — declares model / budget / oracle
  workspace/                # optional — your evaluator + the agent's scratch space
    evaluator.py            #   (only if you ship the oracle as a callable/class)
```

Launch:

```python
from a3dasm import AgenticRun
AgenticRun(study_dir="my_study").execute()
```

Everything else under `my_study/` (`pipeline.ipynb`, `runs/…`) is
**produced by the run** — you do not author it.

---

## The input files

### `PROBLEM_STATEMENT.md` — REQUIRED

The runtime reads this file (`agent_runtime.py`) and injects it into the agents
that request it; it is the entire task definition, and the critic checks the
final deliverables against it. Write it precisely — vague briefs produce vague,
ungatable claims. Include:

- **Objective + explicit success criteria** — the headline number or claim the
  run must deliver (e.g. "maximise normalised buckling load; report the design
  vector and the ledgered value").
- **Design space** — every input variable with bounds, type (continuous /
  integer / categorical) and **units**.
- **Deliverables** — what the run must hand back beyond `pipeline.ipynb`
  (e.g. a mechanism explanation, a plot).
- **Resources** — datasets, solver paths, reference values, prior work to beat.
- **Validity / constraint gates** — feasibility flags, regimes of validity,
  noise gates. These are where reward-hacking happens; state them.

### `config.yaml` — optional (defaults apply), recommended

Every key has a default; an absent `config.yaml` is valid (claude backend,
no budget, honor-system evaluator). Keys the runtime actually **reads**:

| key | type | default | meaning |
|---|---|---|---|
| `model` | str | backend default | LLM id. `AgenticRun(model=...)` overrides. |
| `backend` | `claude` \| `ollama` | `claude` | which LLM backend |
| `budget` | `"HH:MM:SS"` or seconds | none (unlimited) | **soft** wall-clock (warn at 95/100%); a separate `RUN_BACKSTOP_MULTIPLE`× backstop guards runaway cost |
| `eval_budget` | int | none | **soft** cap on ground-truth evaluations |
| `required_deliverables` | list[str] | `[]` | EXTRA files that must exist before `Done()` is accepted (`pipeline.ipynb` is the single deliverable and is **always** required regardless — it is not listed here) |
| `evaluator` | block | none → honor-system | declares the oracle — see below |

> ⚠️ **`checkpoint_every` is a no-op.** It appears in `README-agentic.md`
> examples and several shipped configs, but the runtime does **not** read it
> today. Setting it does nothing. (Tracked as a doc/contract gap — see bottom.)

### `workspace/` — optional

Where your evaluator file lives (if you ship one) and where the agents do scratch
work. Not needed if you use a `lookup` pool or let the DataGenerator author the
oracle.

### `run.py` — optional

If present, it defines a custom graph topology / launch and is executed *instead*
of the default entrypoint (this is how the container runner launches a study that
ships its own topology). Most studies don't need it — the default graph is used.

---

## The evaluator contract (the oracle)

This is the most important contract. **`get_evaluator()` (no arguments) is the
only metered path to ground truth**: only rows produced through it are counted
against `eval_budget` and admitted to the canonical ledger. Surrogates,
acquisition functions and optimisers the agent builds are its *own* free
`DataGenerator`s — not routed through `get_evaluator()`, not metered. You
declare the oracle in one of these ways (decision order):

1. **Shipped callable** — a `**kwargs`-style function, one kwarg per input:
   ```yaml
   evaluator:
     entrypoint: "workspace/evaluator.py:evaluate_kw"   # path relative to study_dir
     output_names: [f]                                   # required for a bare callable
   ```
2. **`DataGenerator` subclass:**
   ```yaml
   evaluator:
     entrypoint: "workspace/data_generator.py:MyDataGenerator"
   ```
3. **Precomputed lookup pool** (probes resolve to the nearest pool row):
   ```yaml
   evaluator:
     lookup:
       pool: "experiment_data"            # path under study_dir
       input_columns: [x1, x2, x3]
       output_columns: [y]                # optional
   ```
4. **Agent-authored** — omit the `evaluator` block entirely *and* describe the
   oracle (a binary, dataset, or physics) in `PROBLEM_STATEMENT.md`; the
   DataGenerator agent writes a normalised `DataGenerator` and the runtime
   registers it mid-run. `get_evaluator()` then resolves it.
5. **Nothing** — no `evaluator` and nothing authored → an **honor-system**
   fallback where the agent self-reports eval counts (no instrumented ledger).
   Fine for exploration; **not** recommended for benchmarks you want gated.

`fidelity_column` (optional) names a column for multi-fidelity oracles.

---

## What the run PRODUCES (do not author these)

```
my_study/
  pipeline.ipynb                           ← THE single deliverable (study root): leading markdown cells hold the writeup; runnable cells reproduce the headline from the ledger (0 new evals). Stamped post-run with a trailing metadata cell + notebook metadata (model, run, gate_outcome)
  runs/<timestamp>/
    experiment_data/experiment_data/        ← canonical ledger: output.csv / input.csv / jobs.csv / domain.json
    debug/
      strategizer_notes/hypotheses.json     ← Popperian hypothesis trail
      delegation_log.jsonl                  ← every Delegate() + status
      critic_reviews/call_NNN.md            ← each critic audit + verdict
      diagnostics.jsonl                     ← ScienceMonitor rule firings
      retrospectives.jsonl                  ← per-node consistency notes
      run.log                               ← human-readable log
      run_status.json                       ← gate outcome (GATED / UNGATED / FAILED)
```

The gate outcome lives in `run_status.json` (and is mirrored into the notebook's
`agentic` metadata as `gate_outcome`). A `gate_outcome` of `UNGATED` (or `FAILED`
/ a `BUDGET EXCEEDED` banner) means the run did **not** pass the adversarial
critic — treat it as unaudited.

---

## Authentication

Runs use Claude via a **subscription OAuth token** (preferred) or an API key
(fallback). Export `CLAUDE_CODE_OAUTH_TOKEN` (from `claude setup-token`) before
launching; see the benchmarks tutorial's Authentication section for the
generate-on-laptop / use-on-cluster recipe.

---

## Pre-flight checklist

Before a long run, confirm:

- [ ] `PROBLEM_STATEMENT.md` exists and states **explicit** success criteria,
      the design space (bounds/types/units), and deliverables.
- [ ] `config.yaml` parses, and its `evaluator` block points at a real
      `file.py:attr` (path relative to study root) or a real `lookup.pool`.
- [ ] The oracle imports and runs on **one** sample without error.
- [ ] Auth: `CLAUDE_CODE_OAUTH_TOKEN` is exported and `api.anthropic.com` is
      reachable from where you'll run.
- [ ] You expect a `pipeline.ipynb` deliverable — it's the single deliverable,
      always required, the `Done()` gate refuses to close without it, and its
      runnable cells reproduce the headline from the ledger (0 new evals).

---

## Minimal worked example

A runnable, **tested** copy of this folder lives at
[`example_study/`](example_study/) and is executed end-to-end by
`tests/agentic/test_study_contract.py` (config loader + `get_evaluator()`), so
this example cannot silently drift from the runtime contract — if it does, the
test fails.

`my_study/config.yaml`:

```yaml
model: claude-haiku-4-5-20251001
backend: claude
eval_budget: 200
evaluator:
  entrypoint: "workspace/evaluator.py:evaluate"
  output_names: [y]
# pipeline.ipynb is the single deliverable (auto-required) — its leading
# markdown cells hold the writeup. Add required_deliverables only for EXTRA files.
```

`my_study/workspace/evaluator.py`:

```python
def evaluate(x1: float, x2: float) -> float:
    """One kwarg per input; returns the single output named in output_names."""
    return (x1 - 1.0) ** 2 + (x2 + 2.0) ** 2
```

`my_study/PROBLEM_STATEMENT.md` (sketch):

```markdown
# Minimise a 2-D quadratic
Objective: minimise y = (x1-1)^2 + (x2+2)^2 over x1,x2 ∈ [-5, 5].
Success: report argmin (x1*, x2*) and the ledgered y*, with pipeline.ipynb
reproducing y* from the canonical store (lazy, zero new evals).
Design space: x1, x2 — continuous, [-5, 5], dimensionless.
Deliverables: pipeline.ipynb (the writeup lives in its leading markdown cells).
```

Then `AgenticRun(study_dir="my_study").execute()`.

---

## Known doc/contract gaps (2026-06-10)

- **`checkpoint_every`** is documented and shipped in configs but **not read**
  by the runtime — a no-op key. Either wire it or drop it from docs/examples.
- **Lookup projection distance is invisible** — a probe that resolves to a far
  pool row is not flagged; projection distance is not stamped into provenance.
  Treat lookup results as approximate.
- **Fidelity is not stamped into provenance** today even when `fidelity_column`
  is set on the oracle — a headline can rest on low-fidelity rows without the
  ledger recording the tier.
