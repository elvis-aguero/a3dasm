# Trivial three-node UI smoke test

This is not a science problem. It exists solely to exercise the strategizer
→ implementer → critic graph live, for watching in the read-only run viewer.

## Task

Delegate to `implementer`: write a small Python file `add.py` in the
workspace containing a function `add(a, b)` that returns `a + b`, then
write and run a tiny script that calls it on a couple of example pairs and
prints the results, confirming it works.

Once the implementer reports back, delegate to `critic` for a normal gate
check on this trivial result, then call `Done()`.

There is no data pipeline, no evaluator, no deliverable notebook.
