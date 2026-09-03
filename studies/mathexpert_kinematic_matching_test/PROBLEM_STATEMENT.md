# MathExpert validation: transcribe, then counterfactually extend, a published model

This is a symbolic-derivation study, not a data-driven design study. There is
no evaluator and no `ExperimentData` ledger. This run has no critic in its
graph and will not produce `pipeline.ipynb` — that is expected, not a
failure; the run will end `UNGATED`, and that label is fine here.

## Source paper

Gabbard, Aguero, Cimpeanu, Kuehr, Silver, Barotta, Galeano-Rios & Harris,
**"Drop rebound at low Weber number"**, *Journal of Fluid Mechanics*, vol.
1019 (2025). Open access. arXiv: `2505.00902`.

The paper's own LaTeX source (fetched from arXiv, unmodified) is at
`workspace/DropRebound_JFM.tex` — read it directly, do not re-fetch or guess
from the title/abstract. The model section is `\subsection{Kinematic match
model formulation}` (`\label{subsec:Formulation}`), specifically
`\subsubsection{The kinematic match}`
(`\label{subsubsec:thekinematicmatch}`) for the four contact-set conditions,
each with its own `\label`:
- `eqn:km_contact_amplitudes` — the geometric matching condition inside the
  contact region, $\theta \le \theta_c$
- `eqn:km_non-contact_amplitudes` — strict inequality outside the contact
  region
- `eqn:km_pressure_amplitudes` — zero pressure outside the contact region
- `eqn:km_contact_derivative` — the tangency condition at the contact
  boundary $\theta_c$

The two evolution equations the contact pressure couples into are
`eqn:nd_dot_U_l_v01` (surface-mode dynamics) and `eqn:nd_dot_v` (center-of-mass
dynamics) — both already non-dimensionalized in the same subsection. Its
model is a direct extension of Galeano-Rios, Milewski & Vanden-Broeck (2017),
"Non-wetting impact of a sphere onto a bath..." (*JFM* 826), which is already
known to this system; the 2025 paper calls its own version of that contact
framework the "KM method."

## Task 1 — Transcribe

Delegate to `math_expert`: read `workspace/DropRebound_JFM.tex`'s KM
formulation (labels above) and transcribe its governing equations — the
surface/pressure spherical-harmonic expansions and the four contact-set
conditions — into a `Workspace` script, `runs/math_workspace/main.py`,
naming each step after the paper's own `\label{...}` tags (e.g.
`eqn_km_contact_amplitudes`). Confirm the script actually executes and
renders `main.tex` / `main_summary.json`. Check whatever algebraic steps the
paper's own derivation makes explicit; record anything that is a modeling
choice rather than a derived step via `assume()` (verdict `ASSERTED`), not
as if it were checked.

## Task 2 — Counterfactual extension

Delegate to `math_expert` again (read `main.py` first; write a **new
edition**, `runs/math_workspace/lcp_variant.py`, by copying and extending
`main.py` — never edit it in place):

> How does the model change if we drop the kinematic matching condition —
> a hard geometric/pressure constraint over an implicitly-determined contact
> set — for a linear complementarity problem (LCP) on a pressure-gap
> formulation instead, admitting arbitrary contact sets?

Revise the relevant assumption step(s) to state the new formulation, re-derive
whatever changes as a mechanical consequence, and report which equations
from Task 1 stay identical, which change, and which part of the new
structure has no mechanically-checkable consequence under the current
`Workspace` vocabulary (a free-boundary/complementarity problem's own
well-posedness is out of MathExpert's reach — name that honestly rather than
forcing it through `check_equals`).

## What "done" looks like

A final reply summarizing what `main.py`/`lcp_variant.py` contain, which
steps in each are `ASSERTED` vs. mechanically checked, and a direct
comparison of the two editions' equation sets.
