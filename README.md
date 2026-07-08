# a3dasm

Agentic Data-driven Design and Analysis of Structures and Materials.

a3dasm is a graph of LLM agents that solves data-driven engineering problems.
A hub strategizer delegates to specialist nodes (literature reviewer, data
generator, implementer, critic) on an open-loop architecture, keeps a Popperian
hypothesis ledger, and produces a reproducible `pipeline.ipynb` deliverable that
must pass a reproduction gate before a run is considered closed.

a3dasm builds on [f3dasm](https://github.com/bessagroup/f3dasm) for the
data-driven primitives (`ExperimentData`, `Domain`, `DataGenerator`, the
`Pipeline` and its SLURM execution). f3dasm is a pinned dependency; a3dasm adds
the agentic layer on top and carries no copy of f3dasm core.

## Install

The repository is private for now, so install from Git:

```bash
pip install "a3dasm @ git+ssh://git@github.com/bessagroup/a3dasm.git@v0.1.0"
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

## Documentation

Built with MkDocs. Run `mkdocs serve` locally, or see the hosted docs once
published.

## License

BSD-3-Clause.
