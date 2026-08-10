# a3dasm

Agentic Data-driven Design and Analysis of Structures and Materials.

You write one file describing an engineering design or data problem — the
objective, the design space, what counts as valid. a3dasm runs a team of LLM
agents that decide what to try, build the code to evaluate it, run real
experiments, review their own conclusions before accepting them, and hand you
back a notebook that reproduces the result end to end.

It builds on [f3dasm](https://github.com/bessagroup/f3dasm) for the
data-driven primitives (`ExperimentData`, `Domain`, `DataGenerator`, the
`Pipeline`); a3dasm is the agentic layer on top and carries no copy of f3dasm
core.

## Install

```bash
pip install "a3dasm @ git+https://github.com/elvis-aguero/a3dasm.git"
```

You'll also need a model to drive the agents — by default, the
[Claude CLI](https://docs.claude.com/en/docs/claude-code):

```bash
npm install -g @anthropic-ai/claude-code
claude   # first run prompts you to log in
```

## Quick start

The only required input is a `PROBLEM_STATEMENT.md` in the study directory.

```python
from a3dasm import AgenticRun

report = AgenticRun(
    study_dir="studies/my_study",
    model="claude-haiku-4-5-20251001",
).execute()
print(report)
```

See the [Quickstart](https://elvis-aguero.github.io/a3dasm/notebooks/quickstart/)
for a worked example, start to finish.

## Documentation

<https://elvis-aguero.github.io/a3dasm/> — or run `mkdocs serve` locally.

## License

BSD-3-Clause.
