# Spec 10 — MathExpert: a symbolic-derivation verification node

**Backlog #25.** Priority: medium. Status: spec.

## Goal
Give an agent dedicated, mechanically-verified symbolic math **by default**.
The failure mode this fixes: agents do numerics fine but rarely carry a
modeling decision through the algebra it implies — an early choice cascades
into dozens of derived steps an LLM's own arithmetic can't be trusted on.
MathExpert is a delegate node that authors and runs a small, SymPy-backed
Python script recording a derivation as a **sequential document** — the
same shape a published derivation already has (Galeano-Rios, Milewski &
Vanden-Broeck 2017, "Non-wetting impact of a sphere onto a bath...", §2.1–2.30d,
was used to stress-test this design end to end). It supports three real
usage modes, which turn out to be one mechanism, not three:
- **point queries** against an existing transcribed derivation ("what's the
  coefficient of the viscous term in eq. 2.20c"),
- **transcription** of a published derivation into a checked, re-runnable
  script,
- **counterfactual extension** ("what if we don't drop the high-Reynolds
  term — how does the model change") by copying the script and editing one
  assumption.

MathExpert is **opt-in** (a delegate target, not a mandatory ledger gate)
and its verified derivations are **citable, prose-level evidence** in the
hypothesis ledger — nothing mechanically enforced, per explicit user
decision.

## Primary evidence — the seams already exist

**Agent base contract** (`src/a3dasm/_src/backends/base.py:143-273`): a
subclass sets `system_prompt`, `tools: frozenset[str]`, `role`,
`description` (required), optionally `report_sections`. Concrete skeleton
to mirror directly (not just structurally — the tool *set*, not just the
pattern): `F3dasmImplementerAgent` (`src/a3dasm/_src/agents/implementer.py:518-571`)
— `tools = frozenset({"Bash","Edit","Read","Write","Glob","Grep",
"ReportEvals","RecallStore","QueryStore","OracleStatus","HypothesisList",
"HypothesisGet","BashOutput","KillShell","ReadProblemStatement"})`. It does
**not** override `build_closure_tools`, so it inherits the base default
(read-only corpus lookup, `base.py:222-233`) for free — MathExpert should do
the same: no closure-injection machinery at all (see Design §1, a deliberate
simplification from two earlier drafts of this spec).

**Graph wiring**: `src/a3dasm/_src/agents/_graphs.py` — `Edge("source",
"target")` tuples; existing fan-in precedent for "a specialist any worker
might need mid-task" is `literature_reviewer` (`Edge("strategizer",...)`,
`Edge("datagenerator",...)`, `Edge("implementer",...)`), and `implementer`
itself has one *outgoing* edge to `literature_reviewer`, which is what
grants it `Delegate`/`Wait`/`Reply`/`FollowUp`/`RecallHistory`
(`base.py:163-167`: topology-injected to any node with an outgoing edge).
MathExpert mirrors both shapes (Design §4).

**Delegate mechanism** (`routing.py:824`): `Delegate(target, intent,
expected_report, hypothesis_ids=None, ...)`. The milestone gate only fires
for `target_role == "implementer"` (`:927-928`) — MathExpert is exempt, zero
new plumbing. `hypothesis_ids` linkage (`:860-899`) is the existing,
unmodified mechanism the ledger-citation requirement rides on.

**Persistence location convention** — `LiteratureCorpus`
(`literature_corpus.py:553-581`) and its wiring
(`backends/base.py:243-247`, `study_dir/"runs"/"lit_reviewer_notes"`, a
sibling of the timestamped run directories, surviving every run of the
study). MathExpert's workspace directory mirrors this exact convention
(`study_dir/"runs"/"math_workspace"/`) but **not** the FileLock/JSONL-trace
mechanism itself — see Design §2 for why that was dropped in favor of
something simpler.

**`tests/test_features_documented.py`** enumerates only statically-declared
`.tools`. Since MathExpert declares only native tools (`Bash`/`Edit`/etc.,
identical in kind to the implementer's), this test covers it with **zero
special-casing** — a direct benefit of dropping closure-injected tools.

**External grounding** (surveyed across this spec's iterations):
- [`sympy-mcp`](https://github.com/sdiehl/sympy-mcp) — named, stateful,
  handle-based tool calls, not raw code or bare strings. Informed the
  *vocabulary* (Design §3) even though the final interface is a Python
  library, not MCP tool calls.
- SymPy's own docs (`docs.sympy.org/.../gotchas.html`,
  `.../simplify/simplify.html`): `.equals()` returns `True`/`False`/`None`
  — equality is undecidable in general. This is why `check_equals` must
  report a genuine three-valued verdict (Design §3).
- A SymPy mailing-list thread on saving expressions notes SymPy Live's own
  fix for pickling problems: store the expression and **re-evaluate before
  every future use**. This validated "durability via replay, not
  serialization" as the right direction; the final design (§2) simplifies
  replay further — the replayed artifact is the `.py` source itself,
  executed, not a custom log format interpreted by bespoke code.
- **Primary domain evidence**: walking Galeano-Rios et al. (2017) §2.1–2.30d
  directly (not summarized from memory) surfaced that roughly half of a real
  derivation is pure algebra (substitution, linear combination, sufficiency-
  checking of a proposed particular solution against a target PDE) and the
  other half is physics-justified moves SymPy cannot itself validate:
  linearization (a perturbation truncation), an order-of-magnitude rescaling
  argument (§2.19), disregarding a higher-order term given a stated Reynolds-
  number regime (§2.20), an operator *defined* by its properties rather than
  an explicit formula (the Dirichlet-to-Neumann map, §2.22–2.23), and a
  free-boundary/complementarity problem structure (§2.28–2.30d) that is a
  modeling choice, not an equation. This evidence directly shaped the
  `ASSERTED` verdict (Design §3) and the explicit non-goals (Risks).

## Design

### 1. No closure-injected tools — MathExpert authors and runs a script
`MathExpertAgent.tools = frozenset({"Bash", "Edit", "Write", "Read", "Glob",
"Grep", "RecallStore", "QueryStore", "HypothesisList", "HypothesisGet",
"ReadProblemStatement"})` — the same *kind* of tool set as the implementer,
not a bespoke closure vocabulary. No `build_closure_tools` override (two
earlier drafts of this spec built one; dropped — see Risks for why keeping
it would have been the wrong call). This is a direct simplification: no new
tool-injection machinery, and the existing FEATURES.md/test enforcement
already covers it.

### 2. A durable, sequential, file-per-edition workspace
New dependency: `sympy>=1.13` in `pyproject.toml` (unbounded floor,
matching this project's convention for a normal sync-time dependency — see
`f3dasm>=2.4.0`; the one place this repo pins an upper bound, `_EMBED_WITH`,
is about an ephemeral subprocess env with an observed breaking release, a
different situation).

New first-party library, `src/a3dasm/_src/math_dsl.py`, exposing a
`Workspace` class, re-exported publicly as `a3dasm.Workspace` (mirroring how
`get_evaluator` is re-exported for agent-authored pipeline scripts) so a
derivation script imports it the same way any other agent-authored script
imports a3dasm's own API: `from a3dasm import Workspace`. A derivation is a
**`.py` script** written against it,
one file per "edition" (namespace), under `study_dir/runs/math_workspace/`
— e.g. `main.py` (a transcribed paper, as published) and `high_re.py` (a
counterfactual extension). **Durability and revision are both filesystem
operations, not a bespoke mechanism**: a later delegation `Read`s an
existing edition, optionally re-`Bash`-runs it to confirm it still executes
and to regenerate its outputs, then `Edit`s it to extend the derivation
further. Revising an assumption is `cp main.py high_re.py`, editing the one
`ws.assume(...)` call, and rerunning — no `Fork`/branch/graph-edit
primitive exists. Two editions sharing a prefix and diverging after one
`assume()` call are two ordinary files a normal diff already compares.

This is the collapse of an idea this spec went through two heavier versions
of: a JSONL call-log replayed by bespoke code, and a DAG of named nodes with
dependency edges and staleness propagation. Both were dropped once it was
clear a real derivation is linear (each step cites only *earlier* material,
never later — true of essentially any paper, since that's what makes it
readable top to bottom) — a DAG models a more general dependency structure
than the domain has, and forces a translation step a human autoring a
derivation would never naturally produce. The `.py` script already *is* the
sequential document; running it top to bottom already *is* the replay.

### 3. The `Workspace` vocabulary — exactly two categories, not three
Every method call appends one ordered, typed record to an internal list
(`self._steps`) — this list, rendered, **is** the sequential document. Not
raw free-form SymPy: the vocabulary is fixed and narrow specifically so the
audit trail survives even though the artifact is executable code.

An earlier draft of this spec had three step kinds — `define` (an ansatz),
`assume` (a physical claim), `equation` (a derived relation, unchecked,
tagged with a `method=` string) — and a KB entry telling the agent how to
tell them apart. That's a design smell, not a teaching opportunity: needing
prose to distinguish two API calls means the API drew the line in the wrong
place. The real distinction is not "ansatz vs. physical claim vs. derived
relation," it's just **checked vs. not checked** — so there are exactly two
categories, and which one you're in is obvious from which method you called:

- **`assume(name, statement, expr=None, symbolic_effect=None)`** — anything
  not mechanically checked, full stop. Verdict is *always* `"ASSERTED"`.
  Covers what the earlier draft split across two methods: a defining
  relation/ansatz (pass `expr`, e.g. `u = v + w`) and a pure physical claim
  licensing a later truncation (pass `symbolic_effect`, e.g. "drop terms
  `O(eps**2)`"). SymPy never adjudicates either — an ansatz choice and a
  scaling argument are both physics judgment, not algebra.
- **Everything else is a real SymPy computation, correct by construction**
  (`check_equals`, `check_holds`, `check_dimensions`, `truncate_series`,
  `solve_ode`) — SymPy actually computed the result deterministically; a
  verdict (`CONFIRMED`/`REFUTED`/`INCONCLUSIVE`) only enters when the
  computed result is checked *against an independent target* (a claimed
  simplification, a paper's published equation, a target PDE a proposed
  solution should satisfy).

Combining prior equations (e.g. "take the x-derivative of (2.11a), the
y-derivative of (2.11b), add them, and use (2.10b)") is not a third category
either — it's ordinary SymPy arithmetic on already-declared expressions
(`.diff()`, `+`, `.subs()`), with `check_equals` available whenever you want
to confirm the result against a specific target. No `equation()` escape
hatch is needed for it.

```python
class Workspace:
    def __init__(self, name: str): ...
    def symbols(self, names: str, **assumptions): ...          # e.g. real=True
    def function(self, name: str, *args): ...

    def assume(self, name, statement: str, expr=None,
               symbolic_effect: str | None = None): ...
        # The ONLY unverified-claim primitive. Verdict is ALWAYS "ASSERTED".
        # `expr`: a defining relation/ansatz being introduced (e.g. the
        # Helmholtz decomposition, or a proposed particular solution).
        # `symbolic_effect`: what a scaling/truncation argument licenses,
        # e.g. "drop terms O(eps**2)", applied afterward via truncate_series.

    def check_equals(self, name, lhs, rhs, derived_from=()) -> str:
        ...
        # Verdict-bearing: computes (lhs - rhs).simplify(); "CONFIRMED" if 0,
        # "REFUTED" if .equals() is False, else "INCONCLUSIVE" with the
        # residual attached. Never coerces the third case into either pole.
        # This is how the algebraic CONSEQUENCE of an assume() step gets
        # checked, even though the assumption itself is ASSERTED, not proven.

    def check_holds(self, name, predicate, assumptions=None, derived_from=()): ...
    def check_dimensions(self, name, lhs, rhs, derived_from=()): ...
        # Same three-valued verdict, different SymPy subsystem
        # (assumptions/ask() engine; sympy.physics.units) — a domain-
        # constraint claim and a unit-homogeneity claim are not equality
        # claims and CheckEquals cannot express either.

    def truncate_series(self, name, expr, small_param, order, derived_from=()): ...
        # Perturbation expansion + truncation — the linearization step,
        # the high-Re truncation. Not expressible via check_equals/
        # substitution alone (needed, and missing, in this spec's first draft).

    def solve_ode(self, name, expr, func, ics=None, derived_from=()): ...
        # dsolve-based integration with an initial/boundary condition —
        # algebraically distinct from algebraic solving (needed for exactly
        # the §2.17-style step: integrate in time, apply "at rest initially").

    def coefficient(self, expr, term): ...
        # Thin wrap of SymPy's .coeff() — answers "what's the coefficient of
        # X" via GetNode-equivalent + this, not a bespoke query language.

    def render_latex(self, path: str) -> None: ...
        # Walks self._steps in order, emits one LaTeX-rendered equation per
        # step with its rationale — the sequential document a human or
        # another agent reads directly.

    def write_summary(self, path: str) -> None: ...
        # A flat {step_name, type, verdict, latex} table as plain JSON — the
        # interoperability contract: another agent parses this without
        # importing SymPy or reading the .py source at all. "Trust the
        # process" means trusting THIS artifact's verdict column, not
        # re-deriving anything.
```

An operator defined only by its properties (the Dirichlet-to-Neumann map,
§2.22-2.23) is introduced via `assume(..., expr=...)`, its rationale
documenting the defining property (linear, decay-respecting) —
its existence/explicit form is never something SymPy is asked to verify;
any later algebra that substitutes it into an equation is still checked
normally, treating it as an opaque function symbol.

**No `nondimensionalize` primitive** — an earlier draft of this spec had
one, dropped after empirically testing the premise it relied on. Rescaling
an *independent variable* already inside a built `Derivative` does not
reapply the chain rule via `.subs()`:
```python
sp.Derivative(eta(t), t).subs(t, (L/V)*tp).doit()
# -> Subs(Derivative(eta(t), t), t, L*tp/V)   -- inert, not simplified
```
SymPy fails safe here (it never produces the *wrong* value, it just refuses
to simplify) — but that means nondimensionalization isn't a transform a
bespoke method could apply to an already-derived equation; it requires
declaring functions of the *new* scaled variables from the start and
re-differentiating (`sp.diff(eta((L/V)*tp), tp)` does correctly produce the
chain-rule factor). That's a fresh re-derivation, not a distinct mechanical
operation — exactly how the paper itself does it (§2.14 declares new
dimensionless variables and characteristic scales; §2.15 re-derives the
boundary conditions from §2.9-2.13, it does not rescale them in place). So
nondimensionalizing, when needed, composes from existing primitives:
`symbols`/`function` for the new scaled quantities, `assume()` for the
choice of characteristic scales (a genuine modeling pick, matching the
paper's own "we take ρ as the characteristic density..."), then ordinary
`diff`/`check_equals`/`check_dimensions` to confirm the regrouped result.
No new primitive needed — same reasoning that cut `equation()` earlier.

### 4. Graph wiring — NOT added to `_default_graph()`
Corrected from an earlier draft of this spec, which said to add MathExpert
to `_default_graph()`'s standard topology. Checked the actual precedent for
"a shipped, tested Agent subclass not every study needs": `DebuggerAgent`
(`src/a3dasm/_src/agents/debugger.py`) is exported publicly
(`a3dasm.__init__.__all__`) but never instantiated inside `_default_graph()`
— confirmed by grep, it appears nowhere else in `_src`. A user opts into it
by building their own `Graph`, the documented pattern in
`docs/customizing-a-run.md`. Silently adding a 6th node to the default
topology would change every existing study's graph without anyone asking
for it — the literal opposite of "opt-in." `MathExpertAgent` follows the
same precedent: exported from `a3dasm.__init__`, never added to
`_default_graph()`.

For a custom graph that does include it, mirror `literature_reviewer`'s fan-in
shape — `Edge("strategizer","math_expert")`, `Edge("implementer","math_expert")`,
`Edge("datagenerator","math_expert")` — and give it one outbound edge,
`Edge("math_expert","literature_reviewer")`, mirroring `implementer`'s own
lateral edge (a real derivation may need to check a claimed formula against
the corpus); that outbound edge is what grants `Delegate`/`Wait`/`Reply`/
`FollowUp`/`RecallHistory` per the existing topology rule — no
special-casing needed, only present in a graph that wires it that way.

### 4.5. Know-how lives in the KB, not in the agent's own code or this spec
`src/a3dasm/_src/knowledge/entries/0011-symbolic-derivation-patterns.md`
(`audience: [math_expert]`, consultable via `ConsultHandbook`) carries the
distilled literature/internet survey (`sympy-mcp`'s vocabulary shape,
AlphaGeometry's engine-verifies/agent-proposes split, SymPy's three-valued
verdict semantics, autoformalization's real-world brittleness numbers) plus
this project's own design bets (the `.py`-per-edition model, collapsing
"ansatz / physical claim / unchecked derived relation" into the single
`assume()` primitive) with their rationale — each explicitly
labeled as literature-established or project-specific, so a future agent
extending this node knows which parts are safe to treat as settled practice
and which are open to revision on new evidence. This spec records *why the
mechanism is shaped this way*; the KB entry records *how to use it well*.
Per CLAUDE.md §5, this is a hand-documented infra feature (the enumeration
test only covers native `.tools`, not KB content) — the KB entry must ship
in the same commit as the `MathExpertAgent`/`math_dsl` code, or the
know-how it captures has nowhere to land at runtime.

### 5. Ledger citation
Unchanged from the earlier round of this spec: the delegation's report cites
the edition's `.tex`/`.py` path (filesystem-addressable, no new
serialization), attached via the existing `hypothesis_ids` `Delegate`
linkage. Prose-level only, nothing mechanically enforced.

## TDD plan (tests first)
1. `test_workspace_records_steps_in_call_order` — steps appear in `_steps`
   in the order the methods were called, each typed and carrying its
   rationale/verdict.
2. `test_check_equals_reports_three_valued_verdict` — `CONFIRMED` for a true
   identity, `REFUTED` for a false one, `INCONCLUSIVE` for a SymPy-
   undecidable comparison, never coerced.
3. `test_assume_always_records_asserted_never_adjudicated` — `assume(...)`
   always yields verdict `"ASSERTED"` regardless of its `symbolic_effect`
   content; the physical claim is never evaluated by the library.
4. `test_render_latex_emits_one_equation_per_step_in_order` — the rendered
   `.tex` contains one entry per recorded step, in call order, each paired
   with its rationale text.
5. `test_write_summary_is_plain_json_no_sympy_import_needed` — the summary
   file round-trips through `json.load` with no SymPy objects in it (the
   interoperability contract: a consuming agent never needs SymPy).
6. `test_rerunning_the_same_script_reconstructs_identical_steps` — executing
   an edition `.py` twice produces byte-identical `_summary.json` output
   (validates "durability via rerun" is actually deterministic, not
   incidentally so).
7. `test_check_equals_verdict_is_reproducible_across_process_runs` — a
   borderline claim SymPy can only decide via `.equals()`'s internal
   randomized numerical sampling reports the *same* verdict across 5
   separate process runs of the same script. Caught for real during
   implementation, not a hypothetical: `Abs(u).equals(u)` for a
   no-assumptions symbol `u` flips between `REFUTED` and `INCONCLUSIVE`
   across separate process runs (confirmed even with `PYTHONHASHSEED`
   fixed) because `sympy.core.random` is unseeded by default — a real,
   silent threat to test 6's own guarantee for any script that happens to
   hit a borderline claim. Fixed by seeding `sympy.core.random` at
   `Workspace.__init__` with a fixed constant.
9. `test_truncate_series_and_solve_ode_are_distinct_from_check_equals` — both
   exist as separate `Workspace` methods with their own recorded step type,
   not silently routed through equality-checking.
10. `test_math_expert_native_tools_documented_in_features_md` — mirrors the
    implementer's existing enforced requirement.
11. `test_math_expert_wired_into_a_custom_graph_mirrors_literature_reviewer_fanin`
    — build a small custom `Graph` (not `_default_graph()`) with the four
    edges from Design §4; assert `Graph.__post_init__` accepts it (every
    edge endpoint declared) and the outbound edge grants MathExpert
    `Delegate` per the existing topology rule.

## Risks / out of scope
- **Two concurrent MathExpert delegations editing the same edition file** is
  not guarded (no FileLock on the `.py` itself). This mirrors how the
  implementer's own scripts aren't specially guarded either — the canonical
  ledger has `FileLock` because silent numeric drift there is catastrophic;
  ordinary source-file edits carry the same collision risk everywhere else
  in this codebase and aren't singled out. Not a new risk category.
- **`Bash`-executing the edition script is the same trust boundary the
  implementer already has** — not a new risk category, but also not
  something to casually extend to a node that doesn't already warrant it.
- **A previously-considered closure-tool design was dropped deliberately**:
  it would have meant reinventing tool-injection machinery this codebase
  already has a trusted alternative to (native `Bash`/`Edit`/`Write`), for
  no benefit once the artifact itself (the `.py` file) already provides the
  audit trail. Noted here so a future reader doesn't rebuild it without
  re-litigating why it was dropped.
- **A DAG-of-named-nodes design was also considered and dropped** — see
  Design §2. Noted for the same reason.
- **Out of scope, explicitly deferred:**
  - Making MathExpert a *mandatory* gate before a hypothesis can be marked
    SUPPORTED — CLAUDE.md §4 territory, a separate decision from this
    spec's opt-in tool.
  - Verifying an operator's existence/explicit form (Dirichlet-to-Neumann-
    map-style objects) — always `ASSERTED`, never derived.
  - Verifying a free-boundary/complementarity problem's structure (e.g.
    §2.28-2.30d's contact-set conditions) — a modeling choice about problem
    *type*, not an equation; genuinely outside what a SymPy checker can
    adjudicate.
  - Justifying an order-of-magnitude/scaling argument (why a term is
    `O(Re^{-3/2})`, why a boundary layer scales as `Re^{1/2}`) — MathExpert
    can only check the algebraic consequence of such a claim once asserted,
    never originate or validate the claim itself.
  - Autoformalization / formal-proof-assistant integration (Lean/Coq
    lineage) — the aspirational direction raised at the start of this spec's
    discussion, not the v1 target.

## Done when
`MathExpertAgent` exists as a graph node with the fan-in/fan-out edges above;
`math_dsl.Workspace` implements the vocabulary in Design §3 with the four-
valued verdict (`CONFIRMED`/`REFUTED`/`INCONCLUSIVE`/`ASSERTED`) correctly
separating "is the algebra right" from "is the physical claim right"; a
script written against it renders a sequential `.tex` document and a plain-
JSON summary; copying, editing, and rerunning a `.py` edition is the sole
revision mechanism, with no bespoke fork/versioning code anywhere; verdicts
are reproducible across reruns of the same script, including for claims only
decidable via SymPy's own randomized numerical fallback — tests 1-11 green.

This is a new capability, not a regression fix, so there is no prior-run KPI
baseline to beat. The real behavioral questions — does a strategizer
actually reach for it, does transcribing a real paper hold up past the
2.1-2.30d slice already walked here, does it measurably reduce algebra-
related critic rejections — are deferred to a first observation on a real
e2e study, per CLAUDE.md's "headless smoke before e2e" rule: these 9 tests
are the headless smoke; an e2e run checks behavior, it does not hunt for
plumbing bugs.
