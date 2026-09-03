"""Workspace: the vocabulary a MathExpert derivation script is written against.

See internal/specs/10-math-expert-agent.md for the design rationale. Two
categories only, not three: `assume()` records anything not mechanically
checked (verdict always ``"ASSERTED"`` — an ansatz, a physical claim, an
unverified derived relation are all the same thing from SymPy's point of
view); every other method is a real SymPy computation, correct by
construction, with a verdict (``CONFIRMED``/``REFUTED``/``INCONCLUSIVE``)
appearing only when a computed result is checked against an independent
target.

A derivation script instantiates one ``Workspace``, calls its methods in
order, and (usually) ends with ``render_latex``/``write_summary``. Each
mutating call appends one record to ``self._steps`` — that list, rendered,
*is* the sequential document; re-running the script from top to bottom is
the entire "durability" story (no separate trace/replay format).
"""
from __future__ import annotations

import json

import sympy as sp
import sympy.core.random as _sp_random
from sympy import ask
from sympy.physics.units.systems.si import SI

# check_equals/check_holds can fall back to SymPy's own randomized numerical
# sampling (inside .equals()/ask()) for a claim it can't decide symbolically.
# That RNG is unseeded by default, so the SAME script can report a different
# verdict on a rerun -- silently breaking the durability guarantee this
# design depends on (verified directly: `Abs(u).equals(u - 0)` for a
# no-assumptions symbol `u` flipped between REFUTED and INCONCLUSIVE across
# separate process runs, even with PYTHONHASHSEED fixed). Seeding SymPy's own
# RNG makes it reproducible run to run for a script with the same call order.
_DETERMINISM_SEED = 0


def _latex(expr) -> str:
    return sp.latex(expr)


class Workspace:
    """Records a derivation as a sequential list of typed steps."""

    def __init__(self, name: str) -> None:
        self.name = name
        self._steps: list[dict] = []
        _sp_random.seed(_DETERMINISM_SEED)

    # -- vocabulary: declaring symbols/functions -----------------------
    def symbols(self, names: str, **assumptions) -> tuple[sp.Symbol, ...]:
        syms = sp.symbols(names, **assumptions)
        if isinstance(syms, sp.Symbol):
            return (syms,)
        return tuple(syms)

    def function(self, name: str, *args):
        return sp.Function(name)(*args)

    # -- the one unverified-claim primitive -----------------------------
    def assume(
        self,
        name: str,
        statement: str,
        expr=None,
        symbolic_effect: str | None = None,
    ) -> None:
        """Record a claim SymPy never adjudicates. Verdict is always
        ``"ASSERTED"`` — an ansatz (pass `expr`), a scaling/truncation
        argument (pass `symbolic_effect`), or a bare physical claim."""
        latex = None
        if expr is not None:
            if isinstance(expr, dict):
                latex = r" \\ ".join(f"{k} = {_latex(v)}" for k, v in expr.items())
            else:
                latex = _latex(expr)
        self._steps.append({
            "name": name,
            "type": "assume",
            "verdict": "ASSERTED",
            "statement": statement,
            "symbolic_effect": symbolic_effect,
            "latex": latex,
            "derived_from": [],
        })

    # -- verified computations -------------------------------------------
    def check_equals(self, name: str, lhs, rhs, derived_from=()) -> str:
        """CONFIRMED if (lhs - rhs) simplifies to / is proven equal to 0;
        REFUTED if proven not equal; INCONCLUSIVE if SymPy can't decide
        (``.equals()`` returning `None` — never coerced to either pole)."""
        residual = sp.simplify(lhs - rhs)
        if residual == 0:
            verdict = "CONFIRMED"
        else:
            decided = (lhs - rhs).equals(0)
            if decided is True:
                verdict = "CONFIRMED"
            elif decided is False:
                verdict = "REFUTED"
            else:
                verdict = "INCONCLUSIVE"
        self._steps.append({
            "name": name,
            "type": "check_equals",
            "verdict": verdict,
            "latex": f"{_latex(lhs)} = {_latex(rhs)}",
            "residual": str(residual) if verdict != "CONFIRMED" else None,
            "derived_from": list(derived_from),
        })
        return verdict

    def check_holds(self, name: str, predicate, assumptions=None, derived_from=()) -> str:
        """Same three-valued verdict as check_equals, via SymPy's assumptions
        engine (`ask`) instead of equality — a domain/inequality claim is not
        an equality claim and check_equals cannot express it."""
        result = ask(predicate, assumptions) if assumptions is not None else ask(predicate)
        verdict = "CONFIRMED" if result is True else "REFUTED" if result is False else "INCONCLUSIVE"
        self._steps.append({
            "name": name,
            "type": "check_holds",
            "verdict": verdict,
            "latex": _latex(predicate),
            "residual": None,
            "derived_from": list(derived_from),
        })
        return verdict

    def check_dimensions(self, name: str, lhs, rhs, derived_from=()) -> str:
        """Unit-homogeneity check via the dimension SYSTEM's own equivalence
        (`equivalent_dims`), not raw subtraction of dimensional expressions —
        two dimensionally-equal quantities can have syntactically different
        dimensional expressions (a named unit like `newton` reduces to a
        symbol, a built-up expression reduces to `length*mass/time**2`)."""
        dimsys = SI.get_dimension_system()
        d_lhs = SI.get_dimensional_expr(lhs)
        d_rhs = SI.get_dimensional_expr(rhs)
        verdict = "CONFIRMED" if dimsys.equivalent_dims(d_lhs, d_rhs) else "REFUTED"
        self._steps.append({
            "name": name,
            "type": "check_dimensions",
            "verdict": verdict,
            "latex": f"[{_latex(lhs)}] = [{_latex(rhs)}]",
            "residual": None,
            "derived_from": list(derived_from),
        })
        return verdict

    def truncate_series(self, name: str, expr, small_param, order: int, derived_from=()):
        """Perturbation/asymptotic truncation — not expressible via
        check_equals/substitution alone. A computation, not a checked claim:
        verdict is None."""
        truncated = expr.series(small_param, 0, order).removeO()
        self._steps.append({
            "name": name,
            "type": "truncate_series",
            "verdict": None,
            "latex": _latex(truncated),
            "residual": None,
            "derived_from": list(derived_from),
        })
        return truncated

    def solve_ode(self, name: str, expr, func, ics=None, derived_from=()):
        """Integration with an initial/boundary condition (`dsolve`-based) —
        algebraically distinct from algebraic solving. A computation, not a
        checked claim: verdict is None."""
        solved = sp.dsolve(expr, func, ics=ics) if ics is not None else sp.dsolve(expr, func)
        self._steps.append({
            "name": name,
            "type": "solve_ode",
            "verdict": None,
            "latex": _latex(solved),
            "residual": None,
            "derived_from": list(derived_from),
        })
        return solved

    # -- query utility, not a recorded step -----------------------------
    def coefficient(self, expr, term):
        return expr.coeff(term)

    # -- interoperability output -----------------------------------------
    def render_latex(self, path: str) -> None:
        """One block per step, in call order — the sequential document a
        human or another agent reads directly."""
        lines = []
        for step in self._steps:
            lines.append(f"% {step['name']} ({step['type']}, {step['verdict']})")
            if step["type"] == "assume":
                lines.append(step["statement"])
                if step["symbolic_effect"]:
                    lines.append(f"% effect: {step['symbolic_effect']}")
            if step["latex"]:
                lines.append(f"\\[{step['latex']}\\]")
            lines.append("")
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

    def write_summary(self, path: str) -> None:
        """A flat {name, type, verdict, latex} table as plain JSON — the
        interoperability contract: another agent parses this without
        importing SymPy or reading the .py source at all."""
        rows = [
            {
                "name": s["name"],
                "type": s["type"],
                "verdict": s["verdict"],
                "statement": s.get("statement"),
                "latex": s["latex"],
                "derived_from": s["derived_from"],
            }
            for s in self._steps
        ]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=2)
