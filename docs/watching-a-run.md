# Watching a run

A run is autonomous, not silent. While it works you can watch what every
agent is doing, and the run can stop and ask you a question. This page
covers both.

Neither is required: a run with nobody watching behaves exactly the same,
and a question nobody answers times out and the agent proceeds on its own
judgment.

## The live viewer

The viewer is a read-only web view of a study's runs: the graph, each
delegation's full transcript, the evaluation ledger as it fills, and the
deliverable as it is written. It ships as an optional extra.

```bash
pip install "a3dasm[viewer]"
python -m a3dasm.viewer my_study
```

Then open <http://127.0.0.1:8765>. It binds to localhost only and has **no
authentication** — don't put it on a public interface.

```bash
python -m a3dasm.viewer my_study --host 0.0.0.0 --port 9000  # only on a network you trust
```

The viewer is a separate process from the run, so start it in its own
terminal. It reads the run's `debug/` directory, which means
`runtime: debug: true` must be set in `config.yaml` for there to be
anything to watch. It works on a finished run too — the same view, not
streaming.

From Python, if you already have the `AgenticRun` object:

```python
run = AgenticRun(study_dir="my_study")
run.serve_viewer()   # blocking; run it in a separate process from execute()
```

The difference is small but real: `serve_viewer()` hands the viewer this
run's live `Graph` object, while the CLI has no in-memory graph and
recovers one from the study's own `build_graph()` (or falls back to the
default graph).

### What it shows

- **The graph** — every node, its resolved backend and model, and the
  delegation edges between them.
- **Delegations** — each one's task, status, and full transcript, with
  a3dasm's own injected text marked `<a3dasm-note>` so you can tell what the
  tool returned from what the runtime said to the agent.
- **The ledger** — every real evaluation as it is recorded, and the oracle's
  state.
- **Vitals** — budget, spend, and progress.
- **The deliverable** — `pipeline.ipynb` as it is being authored.

## When the run asks you something

The entry node can ask the operator one clarifying question per delegation,
with `FollowUp`. When it does, the run **blocks** until it gets an answer or
times out — `runtime: followup_wait_s` (default `600`, ten minutes). After
the timeout the agent proceeds with its best judgment; an unanswered
question is not a failure.

You can answer from either place, whichever gets there first:

- **the terminal**, if the run is attached to a TTY — type the answer;
- **the viewer**, which has an answer box on the run's page.

Both write to the same on-disk channel in the run's `debug/` directory, so
what was asked, what was answered, and what went unanswered all end up in
the run record rather than scrolling past in a terminal.

Two things follow from that design:

- A headless run (no TTY) is never deaf: the viewer still reaches it. But
  with no TTY *and* no viewer, every `FollowUp` will simply wait out its
  timeout. Set `interactive=False` on `AgenticRun` to turn the prompting off
  entirely for an unattended run.
- The viewer also sends a heartbeat saying a human is actually looking,
  which is how the run distinguishes "someone is about to answer" from "this
  is a stall".

## Sending a note, unprompted

You don't have to wait to be asked. The viewer can queue a **note** to the
entry node — a correction, a constraint you forgot to write down, a "stop
chasing that branch". The agent picks it up on its next tool call, marked as
coming from the operator rather than from a tool.

Notes are one-way and asynchronous: there is no guarantee about *when* the
agent reads one, only that it will, and that it lands in the run record.

## Agents talking to each other

For completeness, since you will see it in transcripts: agents also message
each other mid-flight with `Confer`, which returns immediately and never
blocks either side. The strategizer uses it to steer a delegation that is
*already running* rather than waiting for a wrong result and re-delegating.
A running delegation gets the message prefixed onto its next tool result; an
idle node's message waits until that node next checks in.

You don't drive this — it's between the agents — but it explains messages
appearing in a transcript that the agent never asked for.

## Next

- [Understanding a run's output](reading-a-run.md) — what to read once it
  finishes.
- [How a run is kept honest](how-a-run-is-kept-honest.md) — the checks a run
  has to pass before it can close.
