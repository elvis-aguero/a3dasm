# Authoring a study

A **study** is the single input to a run. It's a folder you prepare; a3dasm reads
it, does the work, and writes its results back into the same folder:

```python
from a3dasm import AgenticRun
AgenticRun(study_dir="my_study").execute()
```

This page shows what to put in that folder.

## What the folder contains

```
my_study/
  PROBLEM_STATEMENT.md   # required — the brief the agents work from
  config.yaml            # optional — the model, budgets, and how to evaluate a design
  workspace/             # optional — your evaluator, if you ship one
    evaluator.py
```

Everything else you'll see later (`pipeline.ipynb`, a `runs/` directory) is
**produced by the run** — you don't write it.

## `PROBLEM_STATEMENT.md` (required)

This is the whole task definition. The agents work from it and the critic checks
the result against it, so write it precisely — a vague brief produces a vague,
unverifiable answer. Cover:

- **Objective and success criteria** — the headline number or claim the run must
  deliver (e.g. "maximise the normalised buckling load; report the design and its
  value").
- **Design space** — every input variable, with its bounds, type (continuous,
  integer, or categorical), and **units**.
- **Deliverables** — anything the run must produce beyond the notebook (a plot, a
  mechanism explanation).
- **What "valid" means** — feasibility limits, regimes of validity, noise
  thresholds. State these explicitly; they're what keep the agents honest.

## `config.yaml` (optional)

Every setting has a default, so you can omit this file entirely. The settings you
can set:

| key | meaning | default |
|---|---|---|
| `model` | which language model to use | the backend's default |
| `backend` | `claude`, `ollama`, `openrouter`, or `vllm` | `claude` |
| `budget` | soft wall-clock limit, `"HH:MM:SS"` or seconds — a nudge, not a hard stop | none |
| `eval_budget` | soft cap on how many real evaluations the run may spend | none |
| `required_deliverables` | extra files that must exist before the run can finish | none |
| `evaluator` | how a design gets scored — see below | honor-system |

See [Configuring the backend](backends.md) for the `backend`/`model` details.

## How designs get evaluated (the evaluator)

The evaluator is your ground truth: the function that scores a design. It's the
one metered path — the only calls counted against `eval_budget`. (Surrogates and
optimisers the agents build on top are their own business and aren't metered.)
Declare it one of these ways:

**A function you ship** — one argument per input variable:

```yaml
evaluator:
  entrypoint: "workspace/evaluator.py:evaluate"   # path within the study folder
  output_names: [y]                               # names of what it returns
```

**A precomputed table** — each query resolves to the nearest row (results are
approximate, since a query may land between rows):

```yaml
evaluator:
  lookup:
    pool: "experiment_data"     # path within the study folder
    input_columns: [x1, x2]
    output_columns: [y]
```

**Described in the brief** — omit the `evaluator` block and describe the scoring
oracle (a binary, a dataset, a physics model) in `PROBLEM_STATEMENT.md`; the
agents write the evaluator themselves during the run.

**Nothing** — no evaluator and none described falls back to the honor system: the
agents self-report. Fine for exploring, not for a result you want verified.

## What the run produces

At the study root you get **`pipeline.ipynb`** — the deliverable. Its opening
cells are the write-up; its code cells reproduce the headline result. Each run
also writes a timestamped folder under `runs/` with the evaluation record, logs,
and a status file. See [Understanding a run's output](reading-a-run.md) for what's
in there and how to read it.

## Before a long run, check

- `PROBLEM_STATEMENT.md` states explicit success criteria, the design space
  (bounds, types, units), and any deliverables.
- `config.yaml` parses, and its `evaluator` points at a real file/attribute or a
  real lookup pool.
- Your evaluator imports and runs on one sample without error.
- Your backend is reachable (for the default Claude backend, you're logged in —
  see [Installation](installation.md)).

## A minimal worked example

A runnable copy of this folder ships in the repository at `studies/example_study`
(it's exercised by the test suite, so it can't drift from what the runtime
actually expects).

`config.yaml`:

```yaml
model: claude-haiku-4-5-20251001
backend: claude
eval_budget: 200
evaluator:
  entrypoint: "workspace/evaluator.py:evaluate"
  output_names: [y]
```

`workspace/evaluator.py`:

```python
def evaluate(x1: float, x2: float) -> float:
    """One argument per input; returns the output named in output_names."""
    return (x1 - 1.0) ** 2 + (x2 + 2.0) ** 2
```

`PROBLEM_STATEMENT.md`:

```markdown
# Minimise a 2-D quadratic
Objective: minimise y = (x1-1)^2 + (x2+2)^2.
Success: report the argmin (x1*, x2*) and the value y*, reproduced in
pipeline.ipynb from the run's evaluation record.
Design space: x1, x2 — continuous, in [-5, 5], dimensionless.
```

Then run it:

```python
from a3dasm import AgenticRun
AgenticRun(study_dir="studies/example_study").execute()
```
