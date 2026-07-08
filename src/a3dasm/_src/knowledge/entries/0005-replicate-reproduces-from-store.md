---
id: pipeline-reproduces-from-store
title: pipeline.ipynb is the deliverable — lazy, and it reproduces the headline from the store
tags: [pipeline, reproducibility, deliverable, critic, headline, lazy, notebook]
audience: [strategizer, implementer]
---
The single deliverable is `pipeline.ipynb`: a human-readable f3dasm Pipeline
notebook of the whole data-driven process that ALSO reproduces the headline.
There is no `pipeline.py` and no `solution.md` — the notebook's markdown cells
ARE the writeup, its code cells ARE the runnable recipe. The runtime EXECUTES
the notebook lazily after the critic gate, asserting the headline re-derives
from the canonical ledger with ZERO new oracle evaluations. Hand-authored idxmin
scripts are gone — running the notebook IS the reproduction. (See
[[pipeline-building-patterns]] for the cell structure: the four f3dasm pillars +
the Popperian spine.)

Its code cells must be LAZY:
- **Oracle:** load the canonical store (`ExperimentData.from_file`) and reach the
  oracle only via `get_evaluator()`. f3dasm skips already-`FINISHED` rows, so a
  fresh run does the full campaign while a re-run evaluates nothing (see
  [[evaluate-through-get-evaluator]]).
- **Heavy non-oracle blocks** (a fitted GP/NN/RF surrogate, costly analysis):
  f3dasm's row-laziness does NOT cover these — cache-or-load them yourself.
  Persist the artifact (`Domain.add_output(name, to_disk=True, store_function=,
  load_function=)`, or a plain "load if the file exists else fit+save" guard) so
  a re-run does not refit. See [[surrogates-are-off-ledger]].

And SELF-REPRODUCING:
- Derive the headline from ledgered rows in a final analysis cell and print it
  exactly as `REPRODUCED: <value>` — never hardcode the answer; deriving it IS
  the reproduction.
- Read the store from the `F3DASM_CANONICAL_STORE` env var when set (the runtime
  gate sets it), else a self-locating path — never a brittle cwd-relative guess
  (that is what broke past runs).
