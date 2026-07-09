# Understanding a run's output

When a run finishes, look in the study folder. Two things tell you almost
everything: the deliverable, and whether it passed review.

## The deliverable: `pipeline.ipynb`

`pipeline.ipynb`, at the study root, is the result. Open it: the opening cells
are the write-up (what was tried, what was found, and why), and the code cells
reproduce the headline number from the run's own evaluation record — so you can
re-run the notebook and get the same answer, not take a claim on faith.

## Did it pass review? `run_status.json`

Before a run is allowed to finish, an adversarial critic reviews the deliverable
and the notebook is re-executed to confirm it reproduces its headline. The
outcome is recorded in `runs/<timestamp>/run_status.json` (and mirrored in the
notebook's metadata):

- **GATED** — passed. The result held up to the critic and reproduces.
- **UNGATED** / **FAILED** — did not pass. Treat the result as unaudited.

If you only check one thing, check this.

## The run folder

Each run writes a timestamped directory:

```
runs/<timestamp>/
  run_status.json     the outcome above
  run.log             a readable log of what happened
  experiment_data/    every real evaluation, on the record
  debug/              detailed traces (see below)
```

`experiment_data/` is the authoritative evaluation record — every measured design
and its result. It's what the notebook re-derives the answer from, and what the
evaluation budget is counted against.

## When a result looks off

If a run came back UNGATED, or the answer surprises you, `runs/<timestamp>/debug/`
holds the detail, in rough order of usefulness:

- **`retrospectives.jsonl`** — each agent's own end-of-run notes: the call it was
  least sure about, and anything that tripped it up. Usually the fastest way to
  see why a run went the way it did.
- **`critic_reviews/`** — the critic's write-ups and verdicts. If a run was
  UNGATED, this says what the critic objected to.
- **`diagnostics.jsonl`** — automatic flags raised during the run (e.g. a claim
  made without enough evidence).
- **`delegations/`** — the full transcript of each task the hub handed to a
  specialist. Open one only when the notes above point you at it.

## Comparing runs over time

`studies/run_ledger.csv` appends one row per run — its outcome, how many
evaluations it used, wall-clock, and the headline value — so you can see whether
successive runs are actually improving rather than just changing.
