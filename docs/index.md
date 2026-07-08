# a3dasm

Agentic Data-driven Design and Analysis of Structures and Materials.

a3dasm is a graph of LLM agents that solves data-driven engineering problems. A
hub strategizer runs an open loop over specialist nodes (literature reviewer,
data generator, implementer, critic), maintains a Popperian hypothesis ledger,
and produces a reproducible `pipeline.ipynb` that must pass a reproduction gate.

It builds on [f3dasm](https://github.com/bessagroup/f3dasm) for the data-driven
primitives and adds the agentic orchestration on top.

- New here? Start with [Installation](installation.md) and the
  [Quickstart](notebooks/quickstart.ipynb).
- Want the full picture of what the system can do? See the
  [Feature catalog](FEATURES.md).
- Building your own study? See [Authoring a study](authoring-a-study.md).
