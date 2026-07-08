---
id: author-and-register-a-source
title: When no source is shipped, author and register one via the datagenerator
tags: [datagenerator, registration, canonical-source, get_evaluator, setup]
audience: [strategizer, datagenerator]
---
If the study ships no canonical ground-truth source (no evaluator entrypoint
or lookup pool in `run_config`), one must be authored before evaluations can
be ledgered. Delegate to the `datagenerator` agent: it conforms whatever the
problem provides — a compiled binary, an external solver, a dataset with a
quirky convention, raw physics, or a plain-language spec — into one validated
f3dasm `DataGenerator`, and writes a `registration.json` manifest.

The runtime reads that manifest on delegation completion and points the
canonical entrypoint at it, so the next `get_evaluator()` resolves it — no
manual config edit. After that, implementers reach the source the normal way
(through `get_evaluator()`) and their evaluations are ledgered with provenance.

Until a source is registered, every evaluation lands off-ledger; the
strategizer will be reminded to register one.
