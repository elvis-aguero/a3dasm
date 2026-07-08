---
id: evaluate-through-get-evaluator
title: Run ground-truth evaluations through get_evaluator()
tags: [evaluation, get_evaluator, canonical-store, provenance, ledger]
audience: [implementer, datagenerator]
---
Every ground-truth (true-oracle) evaluation must go through
`get_evaluator()` so it is written to the canonical ExperimentData store with
provenance — a `_delegation_id` stamp — and counts toward the run's evaluation
ledger. Numbers produced any other way cannot anchor a headline and are not
reproducible from the store.

```python
from a3dasm import get_evaluator
gen = get_evaluator()
data = gen.call(data, mode="sequential")
gen.flush()
```

Do NOT compute the objective in a private loop and write your own CSV/.h5, and
do NOT call `ExperimentData.store()` directly against the run store — both
produce rows without provenance (or no canonical rows at all). If you reported
evaluations but none landed in the canonical store, you bypassed
`get_evaluator()`; re-run them through it.

**Exception — the datagenerator's one-sample validation.** A datagenerator
validates its freshly-authored generator by calling it directly (e.g.
`gen.call(sample)`), NOT through `get_evaluator()`. At that point the source
is not yet registered, so `get_evaluator()` would resolve nothing. This rule
is about the *implementer* reaching the *registered* source — once a source is
registered, every true-oracle call goes through `get_evaluator()`.
