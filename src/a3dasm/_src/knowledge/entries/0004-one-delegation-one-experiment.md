---
id: one-delegation-one-experiment
title: One delegation is one bounded experiment
tags: [delegation, granularity, strategizer, campaign]
audience: [strategizer]
---
A single `Delegate()` call is ONE bounded experiment — a single sweep, fit,
optimisation pass, or falsification probe that a worker finishes in a few tool
calls. Do not hand a worker an entire multi-phase campaign in one call
("sample, fit a surrogate, run BO, then multi-start, then falsify").

A campaign is a SEQUENCE of delegations you steer between, reading each report
before choosing the next. A falsification probe is always its own delegation
with `is_falsification_attempt=True`. Giant delegations are uninterruptible,
hide their progress, and blow the time budget — keep each one small enough to
fail fast and inform the next.
