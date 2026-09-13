# How a run is kept honest

An autonomous run can talk itself into a conclusion. Most of a3dasm's
machinery exists to stop that. This page is the inventory: every check
between an agent deciding something and that decision counting, what each
one does, and which ones you can turn off.

The vocabulary matters, because the two kinds behave completely differently:

- A **hard** check *refuses*. The agent asks to do something and is told no,
  synchronously, and the thing does not happen.
- A **soft** check *speaks*. The agent is told something and carries on. It
  is advice, not control.

Soft checks are deliberately soft. A rule that halts a run every time it
looks suspicious produces a system that cannot finish; a rule that says what
it noticed produces one that can be audited. Which of the two a given check
is, is the first thing to know about it.

## Before the run: the problem statement review

Before the loop starts, your `PROBLEM_STATEMENT.md` is reviewed against the
elements of a well-posed problem — objective, design space, success
criteria, validity, deliverables.

This is **advisory**. It always writes a report and it *never* blocks a run.
If you are running interactively it may ask you up to `max_ask` clarifying
questions; set `review_statement=False` on `AgenticRun` to skip it entirely.

A vague brief is the single most common cause of a disappointing run, and
this is the cheapest place to catch one — but the decision stays yours.

## During the run: refusals at the tool boundary

These are hard. The agent calls a tool and the tool returns an error instead
of doing the thing.

**A hypothesis cannot be marked SUPPORTED without a recorded attempt to
falsify it.** This is the charter's central rule, enforced where it cannot be
argued with — in the code path, not in a prompt. An agent that tries to
close a hypothesis as supported having never attacked it gets a refusal.

**Evidence must cite a delegation that actually ran.** You cannot support a
claim by referring to work that did not happen.

**A delegation must be anchored to hypotheses that exist.** Work is attached
to the question it is meant to answer, which is what makes every evaluation
traceable to a claim afterwards.

**The implementer is unreachable while the process backlog is open.** The
strategizer has a short list of process milestones — engage with the
problem, consider where it might be wrong, get the oracle right — and cannot
delegate experiments until each is either done or explicitly skipped with a
stated reason. Skipping is allowed; skipping silently is not.
Turn the whole mechanism off with `runtime: milestones_enabled: false`.

## During the run: the referee on closing verdicts

When the strategizer closes a hypothesis (SUPPORTED, FALSIFIED or
INCONCLUSIVE), an independent referee judges that verdict against the *same*
charter text the critic uses: was the registered prediction actually given a
severe test, and does the verdict follow from the result?

This is **soft** — the ruling comes back as advice attached to the update,
and it escalates if the same hypothesis keeps getting flagged. Disable it
with `F3DASM_VERDICT_VALIDATOR=0`.

## During the run: the monitors

Three checks run continuously and never block. Each reaches the agent as an
in-band note, marked so it is distinguishable from tool output:

- **unledgered evaluations** — results being reasoned about that never made
  it onto the record;
- **duplicate evaluations** — the same design measured twice, which wastes
  budget and can inflate an apparent effect;
- **unstamped rows** — evaluations on the record without the delegation that
  produced them.

They are reported, not enforced, and they all land in
`runs/<timestamp>/debug/diagnostics.jsonl` for you to read afterwards.

## Closing the run: the ladder

`Done()` is not one check. It is a sequence, and the first one that holds
returns immediately — nothing after it runs:

1. **Nothing closes mid-flight.** Delegations still running? The close is
   declined (softly — the agent is offered the options: keep working, or wait
   and read the results).
2. **The exit interview.** Each agent records what it was least sure about
   and what got in its way, before anything closes. This is what ends up in
   `retrospectives.jsonl`.
3. **The milestone backlog must be resolved.** Hard, same mechanism as above,
   same `milestones_enabled` switch, same `MilestoneSkip` escape.
4. **`Done()` is two-shot.** The first call warns and lists everything still
   unmet; only a second call closes. A run cannot end on a single impulse.
5. **The deliverable must reproduce — before a critic sees it.** A
   `pipeline.ipynb` that cannot execute bounces straight back to the
   strategizer rather than spending a critic turn on it. Bounded, so a
   persistently broken notebook still lets the run end (UNGATED).

Then, and only then:

6. **The adversarial critic gate.** A critic reads the deliverable against
   the problem statement and the charter and returns a verdict. **A run
   closes GATED only on a critic PASS.** Anything else sends control back to
   the strategizer.

## The reproduction gate, in detail

Step 5 above is the check that makes the notebook you get back trustworthy,
so it's worth stating exactly:

`pipeline.ipynb` is executed end to end in a hermetic sandbox copy of the
evaluation ledger, and it must

- **exit cleanly** — no errors;
- **add zero new oracle evaluations** — it re-derives from the record rather
  than re-measuring, which is why it is cheap to re-run and why it cannot
  quietly produce a different answer each time;
- **rewrite none** — it cannot alter the record it is reading;
- **agree with itself** — if the notebook prints both the number it computes
  and the number its prose claims, they must match within tolerance. This one
  is deliberately lenient: if either marker is absent it stays silent, so it
  adds no new way to fail, and only catches a deliverable that contradicts
  itself.

Turn the notebook deliverable off entirely with
`runtime: pipeline_deliverable: false`, for a study whose output is not a
notebook. The reproduction gate goes with it.

## The shared charter

Two agents adjudicate a run — the strategizer proposing verdicts, the critic
attacking them — and they are given the *identical* numbered text of the
falsification charter, as is the verdict referee. That is what lets one cite
"Charter §3" and the other defer to the same words, instead of negotiating
over what falsification means.

The charter is one file, injected verbatim, and its wording is pinned by the
test suite. It is not restated anywhere else.

## Where the outcome is written

`runs/<timestamp>/run_status.json`:

- **GATED** — every check above passed, including the critic.
- **UNGATED** / **FAILED** — it did not. The result is unaudited; treat it
  as a draft.

See [Understanding a run's output](reading-a-run.md) for what to read when
it comes back UNGATED, and [Troubleshooting](troubleshooting.md) for the
common causes.

## Summary of the switches

| What | Switch | Default |
|---|---|---|
| Pre-run problem-statement review | `review_statement=False` on `AgenticRun` | on (advisory) |
| Interactive prompting | `interactive=False` on `AgenticRun` | on |
| Process milestones (both the delegation block and the close gate) | `runtime: milestones_enabled` | `true` |
| Notebook deliverable + reproduction gate | `runtime: pipeline_deliverable` | `true` |
| Verdict referee | `F3DASM_VERDICT_VALIDATOR=0` | on |

There is no switch for the charter refusals or the critic gate. Those are
the system.
