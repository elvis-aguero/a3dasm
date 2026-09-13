# Troubleshooting

Ordered by when you'll hit them: before the run starts, while it runs, and
when the result isn't what you wanted.

## Before the run

**`claude` isn't authenticated.** The most common first-run failure, and it
shows up as the model call failing rather than as a clear auth error. Check
the CLI works on its own first:

```bash
claude   # should open a session, not prompt you to log in
```

If it doesn't, either log in once interactively or set an API key:

```bash
export ANTHROPIC_API_KEY=sk-...
```

See [Installation](installation.md). For a non-Claude backend, see
[Customizing a run](customizing-a-run.md#reference-the-available-backends).

**The evaluator entrypoint doesn't resolve.** `evaluator.entrypoint` is
`path/to/file.py:attribute`, relative to the study directory — a path and a
colon, not a dotted module path. Two other ways it fails:

- it resolves to a plain function but `output_names` is missing — a callable
  evaluator must declare the names of what it returns;
- it resolves to something that is neither a callable nor a `DataGenerator`
  subclass.

All three raise at startup with the offending value in the message. Check it
yourself before a long run:

```bash
python -c "import sys; sys.path.insert(0,'my_study'); from workspace.evaluator import evaluate; print(evaluate(0.0, 0.0))"
```

**A `config.yaml` key is ignored.** Anything inside `runtime:` that isn't a
known knob is reported as a warning at startup rather than silently
defaulting — so if a setting isn't taking effect, read the startup output
first. Note the nesting: run knobs go **inside** `runtime:`, while `model`,
`backend`, `budget`, `eval_budget`, `budget_usd`, `evaluator` and
`required_deliverables` are top-level.

## While it runs

**The run seems to hang.** Check, in order:

1. Is it waiting on a question? The entry node can ask you one, and it
   blocks for `runtime: followup_wait_s` (default 600s) before proceeding on
   its own. If the run has no TTY and nobody is watching in the viewer, it
   will simply wait out the timeout. See
   [Watching a run](watching-a-run.md#when-the-run-asks-you-something), or
   pass `interactive=False` for a fully unattended run.
2. Is a model call stalled? `runtime: llm_stream_idle_timeout` (default
   600s) abandons a call after that much stream silence.
3. Is it just working? Long delegations are normal. Open the
   [viewer](watching-a-run.md#the-live-viewer) and look, rather than
   guessing — that is what it is for.

**The run halted on cost.** `budget_usd` is a hard ceiling and it is
**resumable**: raise it and resume from the run directory.

```python
AgenticRun(
    study_dir="my_study",
    budget_usd=25.0,                       # was 10.0
    resume_from="my_study/runs/<timestamp>",
).execute()
```

Resuming replays the run's checkpoint, so the run directory must have been
written with `runtime: debug: true`. On a backend that reports no per-call
cost (Ollama, for instance) the ceiling is inactive and you get one warning
rather than a halt.

**The run halted on repeated errors.** After
`runtime: max_consecutive_errors` (default 12) consecutive *crashes* against
the same target, the run stops. Slow delegations and critic disagreement do
not count — only hard failures do. Read `run.log` for what was failing; this
usually means a broken evaluator or an unreachable backend, not a science
problem. Set it to `0` to disable.

**Wall-clock overran the budget.** `budget` is soft — it nudges. The hard
backstop is `runtime: run_backstop_multiple` (default `2.0`), after which
the run is force-closed.

## When the result isn't what you wanted

**It came back UNGATED.** The run finished but did not pass review. The
result is unaudited — don't quote it. In order of usefulness:

1. `runs/<timestamp>/debug/critic_reviews/` — what the critic actually
   objected to. This is the answer most of the time.
2. `runs/<timestamp>/debug/retrospectives.jsonl` — each agent's own account
   of what it was least sure about.
3. `runs/<timestamp>/debug/diagnostics.jsonl` — flags raised during the run.

See [How a run is kept honest](how-a-run-is-kept-honest.md) for what each
check is testing, so you know which one it failed.

**The notebook won't reproduce.** A run whose `pipeline.ipynb` cannot
execute cleanly bounces back to the strategizer, and after a bounded number
of attempts the run closes UNGATED anyway. Run it yourself to see the real
error — the notebook re-derives from the evaluation record, so it needs that
record present and readable.

**The answer is vague, or answers a different question.** This is almost
always the brief. The agents work from `PROBLEM_STATEMENT.md` and the critic
judges against it, so an unstated success criterion is an unjudged one. Go
back to [Authoring a study](authoring-a-study.md#problem_statementmd-required)
and state the objective, the design space with bounds/types/units, and what
counts as valid — explicitly. The pre-run review exists to catch this; read
its report rather than skipping past it.

**The numbers look self-reported.** If you declared no evaluator and didn't
describe one in the brief, the run falls back to the honor system and the
agents self-report. That's fine for exploring and worthless for a result you
want to defend. Declare an evaluator — see
[How designs get evaluated](authoring-a-study.md#how-designs-get-evaluated-the-evaluator).

**The same design was evaluated twice.** Reported as a diagnostic, not
blocked. It costs budget and can inflate an apparent effect; check
`diagnostics.jsonl` if a result looks stronger than it should.

## Reading the record

Everything above lives under `runs/<timestamp>/`, and almost all of it
requires `runtime: debug: true`. If you plan to debug a run at all, set it
before you start — it cannot be recovered afterwards.

See [Understanding a run's output](reading-a-run.md) for the full layout.
