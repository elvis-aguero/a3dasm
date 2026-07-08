# Minimise a 2-D quadratic (contract example)

This is the canonical minimal study folder for the agentic-f3dasm **study-folder
contract** (see `../authoring-a-study.md`). It is intentionally trivial — its job
is to be a *runnable, tested* reference, not a hard problem.

## Objective
Minimise `y = (x1 - 1)^2 + (x2 + 2)^2` over the design space below. The global
minimum is `y = 0` at `(x1, x2) = (1, -2)`.

## Success criteria
Report the argmin `(x1*, x2*)` and the **ledgered** `y*` in the deliverable
`pipeline.ipynb` — a notebook whose code cells form a lazy f3dasm Pipeline that
loads the canonical store and, re-executed, reproduces `y*` from ledgered rows
(derived, not a hardcoded number).

## Design space
| variable | type | bounds | units |
|---|---|---|---|
| `x1` | continuous | [-5, 5] | dimensionless |
| `x2` | continuous | [-5, 5] | dimensionless |

## Oracle
A shipped callable: `workspace/evaluator.py:evaluate` (one kwarg per input,
returns `y`). Declared in `config.yaml` under `evaluator.entrypoint`.

## Deliverables
- `pipeline.ipynb` — the single deliverable. Its leading markdown cells hold the
  writeup (the argmin, the ledgered `y*`, a one-line conclusion); its code cells
  are the lazy f3dasm Pipeline of the whole process. Re-executed, it reproduces
  `y*` from the canonical ledger (zero new evals) and asserts it.
