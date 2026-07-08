# Open Design-Space Discovery — Scientific Framework Spec

> Branch: `exp/open-design-space` (epistemic change, isolated from `dev/confer`).
> This is a **framework spec**, not an implementation plan. The implementation
> plan (TDD tasks) comes *after* the decisive 2D experiment validates the
> direction. Nothing here is proven — it is a hypothesis about how to reframe
> the task, to be tested cheaply before any code is written.
>
> Status as of 2026-06-26: spec approved; awaiting the 2D experiment (step 2).
> Grounded in three codebase audits run 2026-06-26 (epistemic-contract,
> block-vs-nudge, abstraction-mutability).

## Context

**Why.** PI feedback: prescribing a fixed 14D box and asking the agent to beat a
baseline by 15% wastes the one thing an LLM uniquely brings. BO already solves a
14D box. The frontier is *design representation* — inventing new low-dimensional
parametrizations (e.g. top/bottom rings as ellipses with a parametrized phase
offset), each motivated by literature / physics / memory, each studied for
breakthrough potential. The reframe: **"here is the 7D baseline; propose new,
low-dimensional-ish designs and study each for signs of a breakthrough."**

**Why a spec first.** This changes the epistemic contract (CLAUDE.md §4, user-owned),
so it earns its own branch and a written framework before code. The three axes the
user named — epistemics, ruleset, abstraction levels — were audited against the
current codebase.

**Intended outcome.** A scientific framework where a single self-directed agent can
invent and study *new* design spaces over a run, while (a) single-design problems
stay byte-for-byte as they are today, and (b) the provenance floor that makes results
checkable stays intact.

## Governing principle (the user's steer)

**Additive with a default that equals current behavior.** A one-design study must use
zero new concepts and zero new tools — it *is* today's system. Every mechanism below
activates only when the agent opens more than one design space. The simplest, most
general framing wins; no apparatus is imposed on problems that don't need it.

---

## Axis 1 — Epistemics: a claim is scoped to its design namespace

This is a *clarification* of the existing Charter, not new machinery. (User's call:
the per-namespace idea is the one closest to blessing; the "new claim primitive" and
"portfolio layer" options were rejected as overfit / expressivity-hindering.)

- **The Charter, the 4 statuses (OPEN/SUPPORTED/FALSIFIED/INCONCLUSIVE), and the
  verdict validator are untouched.** §2/§3 severity already means "the search had power
  over *the space the claim ranges over*" (`charter.py:62-68`). That space is now
  explicitly **the namespace**, not "the problem." With one namespace this reading is
  identical to today.
- **A new parametrization = a new namespace.** Proposing one is exploration; the
  results *within* it are ordinary hypotheses, tested by the existing machinery, scoped
  to that namespace. No new status, no new lifecycle.
- **Cross-design claims are ordinary hypotheses too** — and the *existing* critic
  criterion 4 (over-generalization, `critic.py:64-74`) already governs them: you may not
  assert across namespaces what you tested in one. The synthesis is **emergent only when
  >1 namespace exists**; it is not a mandatory artifact.
- **The one genuinely new obligation:** a cross-namespace comparison must rest on a
  comparable metric. Axis 3 makes that structural, not policed.

Net epistemic change: near-zero new contract. The space the Charter already references
becomes a first-class, multipliable thing.

## Axis 2 — Ruleset: per-block decision, keep-or-nudge

The block-vs-nudge audit established the floor that **stays HARD** (provenance +
host-safety): schema validation, graph connectivity, nbformat validity, the
per-delegation memory cap, reproduction-must-run, and closing-a-hypothesis-requires-
evidence (Popperian provenance). All unchanged — these have a strong reason.

For each remaining creativity-constraining block, decide **per block**: keep it
blocking *only if there is a strong reason*, otherwise convert to a **two-shot confirm
nudge** (the existing pattern in `CancelDelegation` `routing.py:1307` and `Done()`
`routing.py:1479`): first call emits a message and refuses; an identical second call
proceeds. **The message regenerates per feature and per time-sensitivity** — a
time-sensitive feature (e.g. near a budget/watchdog limit) gets a terse, urgent
second-shot message and a short reconsider window; a non-time-sensitive one (e.g.
deliverable shape) gets a fuller explanatory message.

| Block | Site | Strong reason to keep blocking? | Decision |
|---|---|---|---|
| max-3-OPEN-hypotheses | `hypothesis_ledger.py:205` | No — it was closure *discipline*, not safety; exploring several designs needs >3 open | **Two-shot nudge.** Low time-sensitivity → full message ("you have N open; usual ceiling is 3 — re-call to confirm you want to track more") |
| process milestones (`oracle-ready`, …) | `routing.py:383`, `1449` | Partial — they ensure literature/oracle gates happen, but a nudge keeps the prompt while allowing skip | **Two-shot nudge, recurring per namespace.** Medium time-sensitivity |
| 5-pillar deliverable structure | `AddPipelineCell` phase enum, `routing.py:2409` | No — structure was convenience; a new-design notebook may not fit doe/data_generation/ml/optimization/analysis | **Two-shot nudge.** Low time-sensitivity → full message |

Each conversion is re-checked against this test at implementation time; any block found
to have a strong (provenance/host-safety) reason stays hard. Parsimony check
(CLAUDE.md §2): "let the cooperating agent proceed after confirming intent" is a general
principle, not a patch for one observed failure — a philosopher nods.

## Axis 3 — Abstraction: the namespace primitive (comparable-by-construction)

The mutability audit's headline: **"one space, one oracle" is a config choice, not an
abstraction limit.** Minimal, additive touch-points (default namespace = today; old
runs unaffected):

- `Delegation` gains optional `namespace: str | None = None` (`graph_state.py:64`).
  `None` → default → current path.
- `run_config`: single `evaluator_entrypoint` → `oracles: {default: {...}, <ns>: {...}}`,
  with a back-compat read of the old field (`agent_runtime.py:119`, `137-203`).
- `get_evaluator(namespace=None)` resolves the oracle for the namespace
  (`instrumented.py:532`).
- Per-namespace store dir + its own `.f3dasm_protected` sentinel (`instrumented.py:98`,
  `experimentdata.py:911`). **Provenance preserved**: rows stay `_delegation_id`-stamped;
  the delegation log stays global; each namespace ledger is independently auditable.
- The DataGenerator agent becomes **re-invokable per namespace**; its registration writes
  `oracles[<ns>]` instead of overwriting `default` (`routing.py:939`).

**Comparable-by-construction (user's choice).** A new namespace's oracle =
*(the agent's design→geometry mapping)* composed with the **shared, fixed
objective + feasibility evaluator**. A "15% beat" is then automatically on the same
ruler — comparability falls out of architecture, with no enforcement rule. The agent
*may* swap the objective evaluator (the PI's "touch the datagenerator" freedom), but
then it must **narrate why the new number is still comparable** — checked by the critic
(criterion 4 / CLAUDE.md §4.5 faithfulness), never blocked.

## What does NOT change (expressivity guarantee)

- One-namespace studies: identical to today — no new tools, no new concepts.
- Charter, 4 statuses, verdict validator, the provenance + host-safety floor: untouched.
- The reproduction gate still binds the deliverable (per namespace's ledger).

## Critical files (when implementation begins, post-experiment)

- `src/f3dasm/_src/agentic/graph_state.py` — `Delegation.namespace`
- `src/f3dasm/_src/agentic/agent_runtime.py` — `oracles` registry + back-compat
- `src/f3dasm/_src/agentic/instrumented.py` — `get_evaluator(namespace)`, per-ns store
- `src/f3dasm/_src/experimentdata.py` — per-namespace sentinel (additive guard)
- `src/f3dasm/_src/agentic/hypothesis_ledger.py` — max-open → confirm nudge
- `src/f3dasm/_src/agentic/nodes/tools/routing.py` — milestone + pillar nudges, ns plumbing
- `src/f3dasm/_src/agentic/agents/datagenerator.py` — re-invokable per namespace
- `docs/agentic/FEATURES.md` — new "design namespace" capability entry (same-commit rule)

## Verification — the decisive test BEFORE any code

The framework is an untested hypothesis. The cheap decisive test (no code change — uses
today's single-study machinery): stand up the PI's **ellipse-with-phase 2D design as ONE
namespace** in the current system and run the agent. Watch two things:

1. **Capability** — can it explore the low-D space competently (sample → surrogate →
   optimize) and find good feasible designs?
2. **Honesty** — does it report truthfully whether the design beats the baseline on the
   *comparable* metric, including "no, it didn't, but the idea was worth trying"?

If it cannot get past one hand-built design, iterate there — the multi-namespace
primitive is wasted until the single-design loop is honest and competent. Only after this
holds do we implement Axis 3 (and the Axis 2 nudges). Axis 1 needs no code beyond Axis 3.

## Sequencing

1. **This spec** — conceptual framework + touch-point map (now).
2. **2D single-namespace experiment** — no code; the decisive cheap test. Problem
   statement lives in the benchmark repo (`f3dasm-agentic-benchmarks`), not here.
3. **If it holds** — implementation plan (TDD): Axis 3 namespace primitive + Axis 2
   nudge conversions. Each block→nudge and the namespace plumbing get headless
   regression tests first, e2e last (per the headless-smoke-before-e2e rule).
4. **§4 record** — Axes 1 and 2 are epistemic-contract decisions the user made in this
   session; logged in this doc and the BACKLOG.

---

## Axis 3 eval-accounting — brittleness post-mortem + the bulletproof redesign (2026-06-27)

**What happened.** The first namespace design gave each namespace its OWN physical
store (`experiment_data/<ns>/`). Then every place that counts evals had to remember to
aggregate across stores. It didn't, in *seven* places, discovered one validation run at
a time (commits cea08d8b→d27a33ae): run total, KPI ledger, the unledgered-evals bounce,
per-delegation reconciliation, GetStatus poll, cancel two-shot, critic budget. Then n=5
datapoints exposed an *eighth* failure: the namespace can be chosen at the worker call
site (`get_evaluator(namespace='ring')`) independently of `Delegate(namespace=…)`, so the
registry-keyed fix is blind whenever the two diverge (run 20260627T045747, D004).

**Root cause (not the symptoms).** The DATA MODEL (N physical stores) did not match the
question the whole codebase asks ("how many evals in this run / by this delegation?").
The single-store invariant the code was built on was silently violated. Every aggregation
helper is a band-aid: it makes the *right* read AVAILABLE but leaves the *wrong* read
(`from_store(canonical)`) still present and still LOOKING correct — so the next consumer
is born blind. **A mechanism is bulletproof only when the wrong usage is impossible or
loud, not when the right usage is merely available.**

**The bulletproof redesign: namespace is a COLUMN, not a directory.** Collapse the N
stores back to ONE canonical store; stamp every eval with a `_namespace` provenance
column, exactly as the instrumented layer already stamps `_delegation_id`/`_source`/`_ts`.
Then:
- Run total = `len(store)` — every existing canonical-store reader is correct with ZERO
  changes. The naive read becomes the RIGHT read (the brittleness is inverted, not
  patched).
- Per-delegation count = rows where `_delegation_id==D`; per-namespace view = filter
  `_namespace==X`. Off-ledger guard, repro gate, budget: all read the one store, all
  namespace-aware for free. No subdir iteration, no `oracles[*].store_dir` redirection, no
  registry dependency, no "remember to aggregate."
- Selection path no longer matters: whether the namespace came from `Delegate(namespace=)`
  or `get_evaluator(namespace=)`, the eval lands in the one store stamped `_namespace`.

**Why THIS is not automatically bulletproof either (the adversarial half of the spec):**
- **Heterogeneous input schema.** Different namespaces have different design variables
  (cartesian x,y vs polar r,θ). One store ⇒ either a UNION input schema (sparse/NaN rows,
  Domain grows as namespaces appear — against f3dasm's "Domain fixed once") or a serialized
  `_design` blob column (stable schema, but inputs aren't first-class for the ledger).
  **Decisive test before committing:** does `ExperimentData(canon) + ExperimentData(batch)`
  with DISJOINT input columns union cleanly, or raise? If it unions → sparse columns are
  fine (counting is schema-agnostic; surrogate fitting uses the implementer's own local
  typed working data, not the ledger). If it raises → use the `_design` blob (stable
  schema). Either way the failure is LOUD (merge error) and contained in ONE place, not a
  silent undercount spread across consumers.
- **Backward-compat.** Single-namespace runs = one store, `_namespace="default"`. Existing
  studies unaffected (`_namespace` is additive provenance like `_delegation_id`). Verify no
  reader assumes the column's absence.
- **No f3dasm core change.** `_namespace` is stamped by the agentic InstrumentedDataGenerator
  via the existing `extra_provenance` mechanism — core `ExperimentData` is untouched (modulo
  the union-vs-blob test above).

**Decision rule going forward (the lesson):** prefer designs where the invariant the code
already assumes stays TRUE, over designs that add a parallel structure every consumer must
learn about. When a parallel structure is unavoidable, the spec must enumerate the blind
paths it creates and make them loud — before writing code, not after the third validation
run.

---

## Resolution — the experiment primitive + interaction philosophy (2026-06-27, implemented)

The "bulletproof redesign" section above proposed collapsing to a SINGLE store
with a `_namespace` column. **That was rejected as muddy** (hoping heterogeneous
`ExperimentData` merge gracefully, or a `_design` blob, is not a clean
abstraction — a philosopher frowns). The landed model is cleaner:

**The experiment primitive.** An experiment = its own clean `ExperimentData` + a
registered oracle + a provenance name. The run is the *collection* of experiments.
Honesty is intrinsic per-experiment (claims trace to that experiment's own clean
store); no merge, no blob, no global store. The few genuinely-global totals
(the cost/budget sum) iterate the registered collection.

**Accounting is provenance-based, not experiment-guessing.** A delegation's evals
are counted by its `_delegation_id` stamp across every store, so the count is
correct no matter HOW the experiment was selected (`Delegate(namespace=)` or
`get_evaluator(namespace=)` at the call site). This deleted the brittle namespace
threading and fixed the off-ledger false-positive at its root (commit 02f227f5).

**Comparison stays the agents' judgment, not our machinery.** We provide the
substrate; the comparable path (reuse the registered objective) is made the
low-friction default, not an enforced invariant.

**Interaction philosophy (the guard taxonomy).** Three modes, chosen by
reversibility-first: PROCEED+TIP (easily reversible) · CONFIRM/two-shot
(reversible-but-weighty, or catch-before for not-easily-reversible) ·
PRECONDITION-BLOCK (impossible until the input/world changes). Every message
states what + why + the next step — a helpful collaborator's tip, never a
bureaucratic refusal. §4 decisions implemented this session: SUPPORTED-without-
falsification → two-shot CONFIRM (with written justification); off-ledger →
corrective PROCEED+TIP (no rerun dance; gate is the floor); UNGATED close →
discloses the objections, not the attempt count; max-open & custom-phase →
PROCEED+TIP. Commits 02f227f5 → 25bf2014.

**Still open (the real viability question):** multi-experiment fan-out vs.
wall-clock — how the strategizer budgets/decomposes across experiments so a
3-experiment run doesn't blow the watchdog. The n=5 batch showed 2 GATED / 1
UNGATED / 2 watchdog-killed (one a literature-reviewer hang, one overwork). This
is the design fork that decides whether the direction scales; it is the user's
call and is NOT addressed by the accounting/interaction work above.
