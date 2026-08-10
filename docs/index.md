# a3dasm

**You write one file. It runs the research.**

a3dasm takes a plain-language description of an engineering design or
data-driven problem — the objective, the design space, what counts as a
valid answer — and runs a team of LLM agents that decide what to try, write
the code to evaluate it, run real experiments, check their own conclusions
before accepting them, and hand you back a notebook that reproduces the
result end to end.

That's the whole interface: one input file, one notebook back. Everything
in between — what to try first, how to test it, when to believe it — is the
system's job, not yours.

## Get started

1. [Installation](installation.md) — the package, plus a model to drive the
   agents. Two commands.
2. [Quickstart](notebooks/quickstart.ipynb) — a real problem in, a real
   answer out, a couple of minutes start to finish.
3. [Core concepts](concepts.md) — once you've seen it run once, what the
   pieces are actually called and why.
4. [Authoring a study](authoring-a-study.md) — write your own problem.

## Under the hood

a3dasm is a graph of agents — a hub **strategizer** delegating to
specialists (**literature reviewer**, **data generator**, **implementer**,
**critic**) — that keeps every claim honest with a Popperian **hypothesis
ledger** and closes every run with a **reproducible notebook**: checked end
to end, before the run is allowed to close, not just written up afterward.

It builds on [f3dasm](https://github.com/bessagroup/f3dasm) for the
data-driven primitives and adds the agentic orchestration on top.
