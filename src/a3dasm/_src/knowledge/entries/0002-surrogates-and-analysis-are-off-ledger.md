---
id: surrogates-are-off-ledger
title: Surrogates, samplers, and analysis are free and stay off-ledger
tags: [surrogate, sampler, optimizer, metering, off-ledger, expressivity]
audience: [implementer]
---
Only true-oracle calls go through `get_evaluator()` and count as evaluations.
Everything you build and run yourself — surrogate models (sklearn/botorch GPs,
random forests), acquisition functions, samplers, optimizers, clustering,
backtracking, plots — runs on your own `DataGenerator`s / artifacts and is
**not** metered. That is intentional: explore, fit, and iterate as freely as
you like.

The line is simple: reaching the canonical ground-truth source is metered;
modelling, searching, and reasoning on top of it is free. So fit a thousand
surrogate predictions if you want — only the points you actually validate
against the oracle (via `get_evaluator()`) consume the evaluation budget.
