"""CI gate: the canonical f3dasm idioms injected into agent prompts must
EXECUTE against the installed f3dasm. If an f3dasm API moves, this fails at
build time — so a broken API example can never ship to an agent (and no runtime
introspection is needed).

This is the mechanism behind the single-source design: F3DASM_CORE_IDIOMS lives
in one place, the prompts compose it in, and this test validates it.
"""
from __future__ import annotations

from a3dasm._src.knowledge.idioms import F3DASM_CORE_IDIOMS


def _extract_code(text: str) -> str:
    """Pull the runnable code block (lines indented >= 4 spaces) and dedent it.
    Commented lines (e.g. get_evaluator, which needs a live run) stay comments.
    """
    return "\n".join(
        line[4:] for line in text.splitlines() if line.startswith("    ")
    )


def test_core_idioms_execute_against_installed_f3dasm():
    code = _extract_code(F3DASM_CORE_IDIOMS)
    ns: dict = {}
    exec(compile(code, "<F3DASM_CORE_IDIOMS>", "exec"), ns)  # noqa: S102
    # The idioms must produce what they claim:
    assert ns["X"].shape[0] == 8, "latin sampler should yield 8 rows"
    assert ns["X"].shape[1] == 2
    assert len(ns["cand_data"]) == 2, "candidate construction must work"


def test_idioms_contain_the_verified_correct_forms():
    """Positive guard: the verified idioms (whose absence caused the D002
    thrash) must remain present. The exec test above already guarantees no
    broken CODE can ship; this guards the pedagogy from being gutted."""
    required = [
        "X, y = data.to_numpy()",
        "create_sampler(",
        "sampler.call(data=data",
        "ExperimentData.from_data(",
        "_input_data=",
        "get_n_best_output(5",
    ]
    for token in required:
        assert token in F3DASM_CORE_IDIOMS, f"verified idiom missing: {token}"
