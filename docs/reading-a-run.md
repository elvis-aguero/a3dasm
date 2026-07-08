# Reading a run

Every run writes a forensic trail under
`studies/<study>/runs/<timestamp>/debug/`. When you want to understand what a run
actually did (or why it did something surprising), read these in order. This is
the analysis protocol; the first source is the highest signal.

## 1. `retrospectives.jsonl` — read this first

Each node writes a first-person retrospective after the run closes. One JSON
object per line, with a `text` field in four sections:

- **CONSISTENCY** — rule contradictions the agent noticed in its own work.
- **DECISION** — the most uncertain strategic choice it made, and why.
- **FRICTION** — rules, APIs, or contracts that were counterintuitive or forced
  guessing. A FRICTION entry often names the root cause of a bug that never shows
  up in a traceback.
- **BLOCKED** — capability gaps that stopped the agent doing its job.

```python
import json
from pathlib import Path

debug = Path("studies/my_study/runs/<timestamp>/debug")
for line in (debug / "retrospectives.jsonl").read_text().splitlines():
    entry = json.loads(line)
    print(entry["source_id"], entry["role"])
    print(entry["text"])
```

These are ephemeral: a new run wipes `runs/`. Read them before launching the
next run.

## 2. `diagnostics.jsonl` — the science monitor's firings

Rule events like `UNLEDGERED_EVALS`, `UNSTAMPED_ROWS`, `BUDGET_WARN`,
`HYPOTHESIS_*`. Cross-reference against the retrospectives: a monitor event with
no matching FRICTION entry means the agent did not notice it, which is itself a
finding.

## 3. `critic_reviews/call_NNN.md` — the gate attempts

If the deliverable went through more than one gate attempt, read them in order.
The sequence shows whether the agent was fixing real issues, the critic misfired,
or the same problem was patched superficially and resurfaced.

## 4. Delegation transcripts, targeted

`delegations/<ID>/` holds the full transcript for each delegation. Only open the
ones the steps above flagged. Reading every transcript with no prior hypothesis
produces surface-level, wrong diagnoses; a transcript is evidence for a question
you already have.

## KPIs

For run-over-run comparison, `studies/run_ledger.csv` records one row per run:
gate outcome, evaluations used vs budget, wall-clock, and the headline value. A
change that improves no KPI without degrading another is noise, not progress.

## The deliverable itself

`studies/<study>/pipeline.ipynb` is the scientific output. The reproduction gate
proves it runs and reproduces its headline; it cannot prove the science is good.
Read it as a skeptical peer would: does it explain *why* the result holds
(mechanism tested against the run's own evidence), and is every number and claim
grounded in the run's data?
