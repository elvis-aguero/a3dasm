---
id: symbolic-derivation-patterns
title: Symbolic derivation patterns for MathExpert
tags: [mathexpert, sympy, derivation, verification, assumption, latex, worked-example]
audience: [math_expert]
---
How to write a derivation script against `math_dsl.Workspace`. Attribution
for each pattern below — literature-established vs. this project's own
bet — is in the final section.

## Check what you can; report honestly what you can't
Call `check_equals` for any equality claim — don't assert it in prose.
SymPy's `.equals()` returns `True`, `False`, or `None`; a `None`
(`INCONCLUSIVE`) must be reported as inconclusive, never softened into
"essentially confirmed."

For a claim it can't resolve symbolically, `.equals()` falls back to
sampling random numerical values — verified directly that this makes the
*same* claim report a different verdict (`REFUTED` one run, `INCONCLUSIVE`
the next) across separate reruns of the identical script, which would break
the whole "rerun to reconstruct state" durability model. `Workspace` seeds
SymPy's own RNG at construction so this can't happen; nothing you need to do
about it, but it's why the verdict is trustworthy run to run.

## `Workspace` has two categories: checked, and `assume()`
An ansatz, a physical claim, and an unverified derived relation are all
unadjudicable by SymPy — all three are `assume()`. Everything else
(`check_equals`, `check_holds`, `check_dimensions`, `truncate_series`,
`solve_ode`) is a real computation; a verdict appears only when the result
is checked against an independent target.

Nondimensionalizing is not a primitive of its own — rescaling an
independent variable already inside a built `Derivative` does not reapply
the chain rule via `.subs()` (verified directly: it returns an inert,
unsimplified `Subs(...)` wrapper). Declare functions of the new scaled
variables from the start instead, `assume()` the choice of characteristic
scales, and re-derive with ordinary `diff`/`check_equals`/`check_dimensions`
— the same pattern the paper itself uses (§2.14-2.15 re-derives the
boundary conditions from scratch, it does not rescale them in place).

## Worked example: verifying a particular-solution choice
Galeano-Rios et al. (2017), eq. (2.9b) — after decomposing the velocity
field via the Helmholtz split `u = ∇φ + ∇×Ψ` — reads

```
∇(φ_t) + w_t = ∇(-p̃/ρ) + ν Δw,   z ≤ 0.                          (2.9b)
```

The paper then states: *"We note that (2.9b) is satisfied if we choose*

```
φ_t = -p̃/ρ,                                                       (2.10a)
w_t^i = ν Δw^i,   i = 1, 2, 3;  z ≤ 0."                            (2.10b)
```

This is a **choice**: infinitely many other splits of the sum would also
satisfy (2.9b), so (2.10a,b) is `assume()`, not `check_equals()`.
Substituting it back into (2.9b) and confirming it reproduces the equation
is a separate, checkable step. Here is the `Workspace` script for exactly
that excerpt, `Δ_H = ∂_xx + ∂_yy` computed as a real Laplacian (the paper's
own (2.12) notation) from `w³`'s actual spatial dependence, not stood in for
by an unrelated symbol:

```python
from a3dasm import Workspace

ws = Workspace("galeano_rios_2017_excerpt")

x, y, t = ws.symbols("x y t", real=True)
rho, nu = ws.symbols("rho nu", positive=True)
phi = ws.function("phi", x, y, t)
w3 = ws.function("w3", x, y, t)
p_tilde = ws.function("p_tilde", x, y, t)
delta_w3 = w3.diff(x, 2) + w3.diff(y, 2)  # Delta_H(w3), the real Laplacian

# (2.9b), as published — LHS and RHS kept separate, nothing checked yet.
eq_2_9b_lhs = phi.diff(t) + w3.diff(t)
eq_2_9b_rhs = -p_tilde / rho + nu * delta_w3

# (2.10a),(2.10b): the authors' CHOICE. Not derivable from (2.9b) alone —
# a genuine modeling pick — so it's assume(), never check_equals().
ws.assume(
    "particular_solution_choice",
    statement=(
        "(2.9b) is satisfied if we choose phi_t = -p_tilde/rho and "
        "w3_t = nu*Delta(w3); this is A sufficient split, not the only "
        "one the sum could take -- the choice itself is a physical/"
        "methodological pick, (2.9b) alone does not force it."
    ),
    expr={"phi_t": -p_tilde / rho, "w3_t": nu * delta_w3},
)

# Sufficiency IS mechanically checkable: substitute the chosen split back
# into (2.9b)'s LHS and confirm it reproduces the RHS.
substituted_lhs = (-p_tilde / rho) + nu * delta_w3
ws.check_equals(
    "eq_2_10_satisfies_eq_2_9b",
    lhs=substituted_lhs,
    rhs=eq_2_9b_rhs,
    derived_from=["particular_solution_choice"],
)
# -> "CONFIRMED" (verified by actually running this script, not asserted):
#    substituting (2.10a),(2.10b) into (2.9b) reproduces it exactly. The
#    CHOICE stays ASSERTED forever; only its consequence was ever checked.

ws.render_latex("excerpt.tex")
```

The pattern to repeat for every "we choose/note that X holds if Y" sentence
in a paper: exactly one `assume()` (the choice), exactly one `check_equals()`
(its consequence), nothing else.

## `truncate_series` and `solve_ode` exist because substitution can't do this
- **`truncate_series`**: perturbation/asymptotic truncation (linearizing
  around `η=0`, dropping a higher-order-in-`1/Re` term). Hand-simplifying
  and hoping the small terms disappear is the failure mode this prevents.
- **`solve_ode`** (with `ics=`): integrating a governing equation in time and
  applying an initial condition — (2.17)'s actual mechanism. Treating this
  as algebraic solving silently drops the data that pins down the solution.

## The symbolic engine verifies; you propose the creative step
Your job in a delegation is choosing *what* ansatz, assumption, or
particular-solution guess to try; the library only checks whether that
choice's consequence holds. It will not find a derivation path for you.

## An operator defined by its properties, not a formula, is still `assume()`
Galeano-Rios et al.'s Dirichlet-to-Neumann map `N(φ)` is never given a
closed form — it's defined by *being* the linear operator with a stated
property (solves Laplace's equation in the half-space, respects the decay
condition). Introduce it with `assume(name, statement=..., expr=N_phi)`;
later algebra substituting it into an equation is still checked normally.

A free-boundary or complementarity-style problem structure (a contact set
whose extent is itself unknown, solved for) is a modeling choice about
problem *type*, not an equation — name it as out of reach in the report,
don't force it through `check_equals`.

## Transcribing a published derivation
Name each step after the source's own equation numbering (`eq_2_6a`,
`eq_2_20c`) — this is what makes a later point query ("what's the
coefficient of the viscous term in eq. 2.20c") answerable by exact
reference. A derivation with two dozen or more equations is realistically
not one delegation: the workspace is a plain `.py` file under
`study_dir/runs/math_workspace/`, so a later delegation just reads it,
reruns it to confirm it still executes, and continues.

## Extending or revising a derivation
Copy the edition file (`cp main.py high_re.py`), edit the one `assume()`
call whose premise changed, rerun. Steps before the edit stay byte-identical
between the two files; a diff of the two `.tex` renders shows exactly what
changed downstream. Keep the old edition — it's the control the new one is
compared against.

## What's established literature vs. this project's own bet
The narrow, named, stateful vocabulary traces to `sympy-mcp`; the
engine-verifies/agent-proposes split traces to AlphaGeometry; the
three-valued verdict traces to SymPy's own documented `.equals()` semantics.
The `.py`-file-per-edition durability model, and collapsing every unchecked
claim — ansatz, physical assumption, unverified derived relation — into the
single `assume()` primitive, are this project's own bets: two heavier
alternatives (a JSONL trace log, a dependency DAG, a three-way step
taxonomy) were tried against a real paper first and each added a distinction
that needed prose to explain rather than one obvious from the API. If a
future run finds this leanness wrong, that's a finding to fold back into
this entry, not to silently override.
