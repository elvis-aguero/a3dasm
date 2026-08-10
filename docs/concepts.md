# Core concepts

If you've just run the [Quickstart](notebooks/quickstart.ipynb), you've already
seen a3dasm work end to end: one problem statement in, one reproducible
notebook out. This page names the pieces that made that happen, so you can
reason about what a run is doing, and write a better problem statement for
your own, rather than treating it as a black box.

<figure class="a3l" markdown="0">
<style>
  .a3l {
    --a3l-ink: var(--md-default-fg-color);
    --a3l-dim: var(--md-default-fg-color--light);
    --a3l-accent: var(--md-accent-fg-color);
    --a3l-line: color-mix(in srgb, var(--md-default-fg-color) 20%, transparent);
    --a3l-agent-fill: color-mix(in srgb, var(--md-default-fg-color) 4%, var(--md-default-bg-color));
    --a3l-store-fill: color-mix(in srgb, var(--md-accent-fg-color) 8%, var(--md-default-bg-color));
    margin: 1.6em 0;
  }
  .a3l .a3l-scroll { overflow-x: auto; }
  .a3l svg { display: block; width: 100%; min-width: 950px; height: auto; }
  .a3l .node-title { font-family: var(--md-text-font, sans-serif); font-weight: 600; font-size: 12.5px; fill: var(--a3l-ink); letter-spacing: 0.01em; }
  .a3l .blurb      { font-family: var(--md-text-font, sans-serif); font-size: 12px; font-style: italic; fill: var(--a3l-dim); }
  .a3l .tag        { font-family: var(--md-code-font, monospace); font-size: 9.5px; fill: var(--a3l-dim); }
  .a3l .gate-label { font-family: var(--md-text-font, sans-serif); font-weight: 600; font-size: 12px; letter-spacing: 0.05em; fill: var(--a3l-accent); }
  .a3l line, .a3l path.arrow { stroke: var(--a3l-dim); stroke-width: 1.4; fill: none; }
  .a3l line.trail, .a3l path.trail { stroke: var(--a3l-accent); stroke-width: 1.4; fill: none; }
  .a3l .agent   { fill: var(--a3l-agent-fill); stroke: var(--a3l-line); stroke-width: 1.2; }
  .a3l .store   { fill: var(--a3l-store-fill); stroke: var(--a3l-accent); stroke-width: 1.2; stroke-dasharray: 4 3; }
  .a3l .gateline{ stroke: var(--a3l-accent); stroke-width: 1.6; stroke-dasharray: 2 4; }
  .a3l .icon    { fill: none; stroke: var(--a3l-ink); stroke-width: 1.8; stroke-linecap: round; stroke-linejoin: round; }
  .a3l .icon-fill { fill: var(--a3l-ink); stroke: none; }
  .a3l .legend { display: flex; gap: 20px; margin-top: 14px; font-size: 12.5px; color: var(--a3l-dim); }
  .a3l .legend span { display: inline-flex; align-items: center; gap: 7px; }
  .a3l .legend i { width: 20px; height: 0; border-top: 2px solid; display: inline-block; }
  .a3l .legend .dim i { border-color: var(--a3l-dim); }
  .a3l .legend .trail-key i { border-color: var(--a3l-accent); }
  .a3l figcaption { font-size: 13px; color: var(--a3l-dim); margin-top: 10px; }
</style>
<div class="a3l-scroll">
<svg viewBox="0 0 1000 680" role="img"
  aria-label="The strategizer delegates to four specialists: literature reviewer, data generator, implementer, and critic, and loops on their reports. The implementer's evaluations and the strategizer's hypothesis verdicts feed two ledgers. Both ledgers are checked at a reproduction gate before the deliverable, pipeline.ipynb, is allowed out; failing the gate sends control back to the strategizer instead of ending the run.">
  <defs>
    <marker id="a3l-arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="11" markerHeight="11" orient="auto-start-reverse">
      <path d="M0,0 L10,5 L0,10 z" fill="var(--a3l-dim)" />
    </marker>
    <marker id="a3l-arrow-accent" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="11" markerHeight="11" orient="auto-start-reverse">
      <path d="M0,0 L10,5 L0,10 z" fill="var(--a3l-accent)" />
    </marker>

    <g id="a3l-icon-strategy">
      <line class="icon" x1="16" y1="16" x2="9" y2="9"/>
      <line class="icon" x1="16" y1="16" x2="23" y2="9"/>
      <line class="icon" x1="16" y1="16" x2="23" y2="23"/>
      <line class="icon" x1="16" y1="16" x2="9" y2="23"/>
      <circle class="icon" cx="9" cy="9" r="2.4"/>
      <circle class="icon" cx="23" cy="9" r="2.4"/>
      <circle class="icon" cx="23" cy="23" r="2.4"/>
      <circle class="icon" cx="9" cy="23" r="2.4"/>
      <circle class="icon-fill" cx="16" cy="16" r="3.4"/>
    </g>

    <g id="a3l-icon-book">
      <path class="icon" d="M4,9 C9,6 13,7 16,10 C19,7 23,6 28,9 L28,23 C23,20 19,21 16,24 C13,21 9,20 4,23 Z"/>
      <line class="icon" x1="16" y1="10" x2="16" y2="24"/>
    </g>

    <g id="a3l-icon-flask">
      <path class="icon" d="M12,4 L20,4 M13,4 L13,12 L6.5,25.5 C5.7,27.4 6.7,29 9,29 L23,29 C25.3,29 26.3,27.4 25.5,25.5 L19,12 L19,4"/>
      <line class="icon" x1="9.6" y1="21" x2="22.4" y2="21"/>
      <circle class="icon-fill" cx="14" cy="25" r="1.3"/>
      <circle class="icon-fill" cx="18.5" cy="26.5" r="1"/>
    </g>

    <g id="a3l-icon-code">
      <path class="icon" d="M13,8 L5.5,16 L13,24"/>
      <path class="icon" d="M19,8 L26.5,16 L19,24"/>
    </g>

    <g id="a3l-icon-check">
      <circle class="icon" cx="14" cy="14" r="9"/>
      <line class="icon" x1="20.4" y1="20.4" x2="28" y2="28"/>
      <path class="icon" d="M10,14.5 L13,17.5 L18.5,10.5"/>
    </g>

    <g id="a3l-icon-notebook">
      <path class="icon" d="M8,4 L20,4 L26,10 L26,28 L8,28 Z"/>
      <path class="icon" d="M20,4 L20,10 L26,10"/>
      <line class="icon" x1="12" y1="15" x2="21" y2="15"/>
      <line class="icon" x1="12" y1="19.5" x2="21" y2="19.5"/>
    </g>
  </defs>

  <rect class="agent" x="380" y="18" width="240" height="100" rx="3"/>
  <use href="#a3l-icon-strategy" x="484" y="32"/>
  <text class="node-title" x="500" y="76" text-anchor="middle">STRATEGIZER</text>
  <text class="blurb" x="500" y="92" text-anchor="middle">reads, decides, delegates</text>

  <path class="arrow" marker-end="url(#a3l-arrow)" d="M 612 34 C 660 18, 660 60, 616 66" />

  <rect class="agent" x="60"  y="168" width="200" height="100" rx="3"/>
  <use href="#a3l-icon-book" x="144" y="182"/>
  <text class="node-title" x="160" y="226" text-anchor="middle">LITERATURE REVIEWER</text>
  <text class="blurb" x="160" y="242" text-anchor="middle">finds cited work</text>

  <rect class="agent" x="288" y="168" width="200" height="100" rx="3"/>
  <use href="#a3l-icon-flask" x="372" y="182"/>
  <text class="node-title" x="388" y="226" text-anchor="middle">DATA GENERATOR</text>
  <text class="blurb" x="388" y="242" text-anchor="middle">builds the oracle</text>

  <rect class="agent" x="516" y="168" width="200" height="100" rx="3"/>
  <use href="#a3l-icon-code" x="600" y="182"/>
  <text class="node-title" x="616" y="226" text-anchor="middle">IMPLEMENTER</text>
  <text class="blurb" x="616" y="242" text-anchor="middle">writes, runs, measures</text>

  <rect class="agent" x="744" y="168" width="196" height="100" rx="3"/>
  <use href="#a3l-icon-check" x="826" y="182"/>
  <text class="node-title" x="842" y="226" text-anchor="middle">CRITIC</text>
  <text class="blurb" x="842" y="242" text-anchor="middle">hunts for holes</text>

  <line x1="160" y1="168" x2="425" y2="118" marker-end="url(#a3l-arrow)" marker-start="url(#a3l-arrow)"/>
  <line x1="388" y1="168" x2="468" y2="118" marker-end="url(#a3l-arrow)" marker-start="url(#a3l-arrow)"/>
  <line x1="616" y1="168" x2="532" y2="118" marker-end="url(#a3l-arrow)" marker-start="url(#a3l-arrow)"/>
  <line x1="842" y1="168" x2="590" y2="118" marker-end="url(#a3l-arrow)" marker-start="url(#a3l-arrow)"/>

  <rect class="store" x="140" y="352" width="260" height="62" rx="3"/>
  <text class="node-title" x="270" y="378" text-anchor="middle">HYPOTHESIS LEDGER</text>
  <text class="tag" x="270" y="396" text-anchor="middle">OPEN → SUPPORTED / FALSIFIED</text>

  <rect class="store" x="560" y="352" width="300" height="62" rx="3"/>
  <text class="node-title" x="710" y="378" text-anchor="middle">EVALUATION LEDGER</text>
  <text class="tag" x="710" y="396" text-anchor="middle">append-only · locked</text>

  <path class="trail" marker-end="url(#a3l-arrow-accent)" d="M 502,118 L 502,300 L 270,300 L 270,352" />

  <line class="trail" x1="640" y1="268" x2="700" y2="352" marker-end="url(#a3l-arrow-accent)"/>

  <line class="trail" x1="900" y1="268" x2="900" y2="480" marker-end="url(#a3l-arrow-accent)"/>

  <line class="gateline" x1="60" y1="480" x2="940" y2="480"/>
  <text class="gate-label" x="500" y="460" text-anchor="middle">REPRODUCTION GATE</text>

  <line class="trail" x1="270" y1="414" x2="270" y2="480" marker-end="url(#a3l-arrow-accent)"/>
  <line class="trail" x1="710" y1="414" x2="710" y2="480" marker-end="url(#a3l-arrow-accent)"/>

  <rect class="agent" x="420" y="536" width="160" height="100" rx="3"/>
  <use href="#a3l-icon-notebook" x="484" y="550"/>
  <text class="node-title" x="500" y="594" text-anchor="middle">pipeline.ipynb</text>
  <text class="blurb" x="500" y="610" text-anchor="middle">reproduces the answer</text>
  <line class="trail" x1="500" y1="480" x2="500" y2="536" marker-end="url(#a3l-arrow-accent)"/>

  <path class="arrow" marker-end="url(#a3l-arrow)"
    d="M 420,580 C 260,640 40,600 40,480 L 40,90 C 40,40 280,20 378,45" />
</svg>
</div>
<div class="legend">
  <span class="dim"><i></i>delegate, report, retry</span>
  <span class="trail-key"><i></i>the evidence trail</span>
</div>
<figcaption>Every specialist reports back to the strategizer the same way; only the evidence trail is checked before the notebook is allowed out.</figcaption>
</figure>

## The graph and the open loop

The agents are nodes in a graph. One node, the **strategizer**, is the hub: it
reads the problem, decides what to do next, and hands work to the specialists. It
runs an **open loop**, meaning it is not a fixed pipeline of "step 1, step 2, step
3". After every piece of work comes back, the strategizer looks at the current
state and chooses the next move. The loop ends when the strategizer declares the
work done and that decision survives review.

The specialists it delegates to:

- **literature reviewer**: finds and reads relevant papers.
- **data generator**: turns a way of evaluating a design into a metered oracle
  the rest of the system can call.
- **implementer**: writes and runs the actual code (sampling, surrogates,
  optimisation) against that oracle.
- **critic**: an adversarial reviewer that tries to find holes in a claimed
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
back `pipeline.ipynb` (the reproducible answer) alongside the run's evaluation
record and logs. See [Authoring a study](authoring-a-study.md) to set one up,
[Understanding a run's output](reading-a-run.md) for what comes back, and the
[Quickstart](notebooks/quickstart.ipynb) to run one.
