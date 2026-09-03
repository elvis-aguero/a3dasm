"""Tests for math_dsl.Workspace — written before implementation (TDD red phase).

Covers spec 10 (internal/specs/10-math-expert-agent.md) tests 1-7: the
Workspace library layer, no agent/graph involved yet.
"""

from __future__ import annotations

import json

import sympy as sp

from a3dasm._src.math_dsl import Workspace


def test_workspace_records_steps_in_call_order():
    ws = Workspace("t1")
    x = ws.symbols("x", real=True)[0]
    ws.assume("a1", "an assumption", expr=x + 1)
    ws.check_equals("c1", x + 1, 1 + x)

    names = [s["name"] for s in ws._steps]
    types = [s["type"] for s in ws._steps]
    assert names == ["a1", "c1"]
    assert types == ["assume", "check_equals"]


def test_check_equals_reports_three_valued_verdict():
    ws = Workspace("t2")
    x = ws.symbols("x", real=True)[0]

    confirmed = ws.check_equals("true_identity", (x + 1) ** 2, x**2 + 2 * x + 1)
    refuted = ws.check_equals("false_identity", x, x + 1)
    # A genuinely SymPy-undecidable comparison: two totally abstract,
    # undefined functions of x can never be numerically evaluated against
    # each other, so `.equals()` returns None deterministically (unlike
    # Abs(u) vs u, verified separately to be flaky -- see
    # test_check_equals_verdict_is_reproducible_across_process_runs).
    g, h = sp.Function("g")(x), sp.Function("h")(x)
    inconclusive = ws.check_equals("undecidable", g, h)

    assert confirmed == "CONFIRMED"
    assert refuted == "REFUTED"
    assert inconclusive == "INCONCLUSIVE"


def test_assume_always_records_asserted_never_adjudicated():
    ws = Workspace("t3")
    x = ws.symbols("x", real=True)[0]
    # A statement that is physically absurd / self-contradictory — assume()
    # must never try to evaluate it, only record it.
    ws.assume("nonsense", "x equals x plus one, asserted anyway", expr=sp.Eq(x, x + 1))
    assert ws._steps[-1]["verdict"] == "ASSERTED"

    ws.assume("plain", "a plain physical claim with no expr")
    assert ws._steps[-1]["verdict"] == "ASSERTED"


def test_render_latex_emits_one_equation_per_step_in_order(tmp_path):
    ws = Workspace("t4")
    x = ws.symbols("x", real=True)[0]
    ws.assume("a1", "first step", expr=x + 1)
    ws.check_equals("c1", x + 1, 1 + x)
    ws.check_equals("c2", x, x + 1)

    out = tmp_path / "derivation.tex"
    ws.render_latex(str(out))
    text = out.read_text()

    assert text.index("a1") < text.index("c1") < text.index("c2")
    assert "ASSERTED" in text
    assert "CONFIRMED" in text
    assert "REFUTED" in text


def test_write_summary_is_plain_json_no_sympy_import_needed(tmp_path):
    ws = Workspace("t5")
    x = ws.symbols("x", real=True)[0]
    ws.assume("a1", "first step", expr=x + 1)
    ws.check_equals("c1", x + 1, 1 + x)

    out = tmp_path / "summary.json"
    ws.write_summary(str(out))

    # Round-trips through plain json.load with no sympy import anywhere.
    with open(out) as f:
        data = json.load(f)

    assert [row["name"] for row in data] == ["a1", "c1"]
    assert data[0]["type"] == "assume"
    assert data[0]["verdict"] == "ASSERTED"
    assert data[1]["verdict"] == "CONFIRMED"
    assert all(isinstance(v, (str, type(None), list)) for row in data for v in row.values())


def test_rerunning_the_same_script_reconstructs_identical_steps(tmp_path):
    script = tmp_path / "edition.py"
    script.write_text(
        "from a3dasm._src.math_dsl import Workspace\n"
        "ws = Workspace('rerun')\n"
        "x = ws.symbols('x', real=True)[0]\n"
        "ws.assume('a1', 'a step', expr=x + 1)\n"
        "ws.check_equals('c1', x + 1, 1 + x)\n"
        "ws.write_summary(str(__import__('pathlib').Path(__file__).parent / 'summary.json'))\n"
    )
    import subprocess
    import sys

    subprocess.run([sys.executable, str(script)], check=True, cwd=tmp_path)
    first = (tmp_path / "summary.json").read_text()
    subprocess.run([sys.executable, str(script)], check=True, cwd=tmp_path)
    second = (tmp_path / "summary.json").read_text()

    assert first == second


def test_check_equals_verdict_is_reproducible_across_process_runs(tmp_path):
    """Regression: check_equals's fallback to SymPy's .equals() can numerically
    sample to decide a claim it can't resolve symbolically, and that RNG is
    unseeded by default -- verified directly that `Abs(u).equals(u)` for a
    no-assumptions symbol `u` flips between REFUTED and INCONCLUSIVE across
    separate process runs, even with PYTHONHASHSEED fixed. Workspace seeds
    SymPy's own RNG (sympy.core.random) at construction so the SAME script
    reports the SAME verdict every rerun -- the durability guarantee this
    whole design depends on."""
    script = tmp_path / "borderline.py"
    script.write_text(
        "from a3dasm._src.math_dsl import Workspace\n"
        "import sympy as sp\n"
        "ws = Workspace('borderline')\n"
        "u = sp.Symbol('u')\n"
        "verdict = ws.check_equals('c1', sp.Abs(u), u)\n"
        "print(verdict)\n"
    )
    import subprocess
    import sys

    verdicts = {
        subprocess.run(
            [sys.executable, str(script)], check=True, cwd=tmp_path,
            capture_output=True, text=True,
        ).stdout.strip()
        for _ in range(5)
    }
    assert len(verdicts) == 1, f"verdict changed across reruns: {verdicts}"


def test_truncate_series_and_solve_ode_are_distinct_from_check_equals():
    ws = Workspace("t6")
    eps = ws.symbols("eps", positive=True)[0]
    expr = sp.cos(eps) + sp.sin(eps)

    truncated = ws.truncate_series("linearize", expr, eps, order=2)
    assert truncated == 1 + eps
    assert ws._steps[-1]["type"] == "truncate_series"
    assert ws._steps[-1]["verdict"] is None  # a computation, not a checked claim

    t = ws.symbols("t")[0]
    f = ws.function("f", t)
    ode = sp.Eq(sp.Derivative(f, t), -f)
    solved = ws.solve_ode("decay", ode, f, ics={sp.Function("f")(0): 1})
    assert ws._steps[-1]["type"] == "solve_ode"
    assert ws._steps[-1]["verdict"] is None
    assert solved.rhs == sp.exp(-t)


def test_check_holds_reports_three_valued_verdict():
    ws = Workspace("t7")
    x = ws.symbols("x", real=True)[0]
    y = sp.Symbol("y")  # deliberately no assumptions

    confirmed = ws.check_holds("positive_square_plus_one", sp.Q.positive(x**2 + 1))
    inconclusive = ws.check_holds("unknown_sign", sp.Q.positive(y))

    assert confirmed == "CONFIRMED"
    assert inconclusive == "INCONCLUSIVE"


def test_check_dimensions_uses_dimension_system_equivalence_not_raw_subtraction():
    from sympy.physics.units import kilogram, meter, newton, second

    ws = Workspace("t8")
    m, a = sp.symbols("m a")
    force_expr = m * kilogram * a * meter / second**2

    consistent = ws.check_dimensions("force_matches_newton", force_expr, newton)
    inconsistent = ws.check_dimensions("mass_is_not_a_force", m * kilogram, newton)

    assert consistent == "CONFIRMED"
    assert inconsistent == "REFUTED"


def test_coefficient_is_a_thin_wrap_not_a_recorded_step():
    ws = Workspace("t9")
    x = ws.symbols("x", real=True)[0]
    steps_before = len(ws._steps)

    coeff = ws.coefficient(3 * x**2 + 5 * x, x)

    assert coeff == 5
    assert len(ws._steps) == steps_before
