# Core concepts

a3dasm runs a small team of language-model agents to solve a data-driven
engineering problem end to end. You give it a written problem statement; it
decides what to try, runs real evaluations, argues with itself about whether the
result holds up, and hands back a notebook that reproduces the answer. This page
explains the pieces and the words a3dasm uses for them.

## The graph and the open loop

The agents are nodes in a graph. One node, the **strategizer**, is the hub: it
reads the problem, decides what to do next, and hands work to the specialists. It
runs an **open loop**, meaning it is not a fixed pipeline of "step 1, step 2, step
3". After every piece of work comes back, the strategizer looks at the current
state and chooses the next move. The loop ends when the strategizer declares the
work done and that decision survives review.

The specialists it delegates to:

- **literature reviewer** — finds and reads relevant papers.
- **data generator** — turns a way of evaluating a design into a metered oracle
  the rest of the system can call.
- **implementer** — writes and runs the actual code (sampling, surrogates,
  optimisation) against that oracle.
- **critic** — an adversarial reviewer that tries to find holes in a claimed
  result before it is accepted.

## Delegation

When the strategizer hands work to a specialist, that is a **delegation**. Each
delegation has an id (`D001`, `D002`, …), a task description, and a report that
comes back. Delegations are the unit of work and the unit of accounting: every
real evaluation is attributed to the delegation that produced it, which is what
lets a3dasm tell you exactly where each number came from.

## The hypothesis ledger and the falsification charter

a3dasm does science, so it tracks **hypotheses** explicitly. A hypothesis is a
claim with a testable criterion, a prediction, a prior, and (as evidence comes
in) a verdict. These live in the **hypothesis ledger**.

The rules for what counts as real evidence live in the **falsification charter**.
The charter is Popperian: a hypothesis cannot be marked SUPPORTED without a
recorded attempt to falsify it. This is the epistemic backbone. It exists so the
system cannot quietly talk itself into a conclusion the evidence does not carry.

## The canonical evaluation ledger

Every real oracle evaluation is written once, under a lock, to a shared
**canonical ledger** (an `ExperimentData` store), and stamped with the delegation
that produced it. This store is the single source of truth for "what was actually
measured". It is protected: a stray write that would shrink it, or reset a
completed evaluation, is refused. The headline number in the final deliverable
must trace back to rows in this ledger, or the deliverable cannot reproduce it.

## The deliverable and the reproduction gate

The output of a run is a Jupyter notebook, `pipeline.ipynb`. It is not a summary
written after the fact; it *is* the work. Its markdown cells hold the writeup and
its code cells rederive the headline result from the canonical ledger.

Before a run is allowed to close, the notebook goes through the **reproduction
gate**: it is executed end to end in a clean sandbox, and the number it produces
is checked against the number the run claims. A run that cannot reproduce its own
headline does not pass. This is why the notebook you get back runs as-is.

## Backends

The agents are driven by a language model through a **backend**. a3dasm ships
several: the Claude CLI (default), any OpenAI-compatible endpoint, Ollama,
OpenRouter, and vLLM (including a mode that serves a model on a SLURM GPU node the
framework owns for the run). The backend is a configuration choice; the graph and
the science do not change with it.

## Resource governance

Long autonomous runs need guardrails. a3dasm separates two kinds:

- **Soft budgets** (the evaluation budget) nudge the strategizer when it is
  spending heavily, but never hard-stop the science.
- **The one hard cap** is per-delegation memory (host safety), enforced by a
  watchdog that also reaps runaway processes and force-exits a stalled run.

## What you provide, what you get

You provide one file: `PROBLEM_STATEMENT.md` in a study directory (plus a
`config.yaml` if you want to set the backend, budgets, or the evaluator). You get
back `pipeline.ipynb` — the reproducible answer — alongside the run's evaluation
record and logs. See [Authoring a study](authoring-a-study.md) to set one up,
[Understanding a run's output](reading-a-run.md) for what comes back, and the
[Quickstart](notebooks/quickstart.ipynb) to run one.
