"""Item B — pre-run problem-statement HITL reviewer (LLM, ephemeral session).

The reviewer judges a problem statement against the FIVE universal elements of
a well-posed statement — and NOTHING benchmark-specific.  It is advisory
(always writes a report, never blocks an autonomous run) with an optional
interactive-refine step that appends human clarifications.
"""
from __future__ import annotations

import json

from a3dasm._src.agent_runtime import AgenticRun
from a3dasm._src.reviewer import (
    REVIEW_ELEMENTS,
    REVIEWER_SYSTEM_PROMPT,
    ProblemStatementReviewerAgent,
    parse_review,
    review_gaps,
)

# Benchmark-specific vocabulary the rubric must NEVER hard-code (overfitting to
# the handful of studies we happen to have).
_BENCHMARK_TERMS = [
    "surferbot", "supercompressible", "coilable", "wave_steepness",
    "black_box_8d", "interfacial", "locomotion",
]


class _FakeAdapter:
    """Stand-in LLM: returns a canned response regardless of input."""

    def __init__(self, response: str) -> None:
        self._response = response
        self.model = "fake"
        self.last_usage: dict = {}
        self.calls: list = []

    def invoke(self, messages):
        self.calls.append(messages)
        return self._response


def _run(tmp_path, interactive: bool) -> AgenticRun:
    study = tmp_path / "study"
    study.mkdir()
    (study / "PROBLEM_STATEMENT.md").write_text("# trivial\n")
    return AgenticRun(study_dir=study, interactive=interactive)


# --- overfit guard (deterministic, no LLM) ----------------------------------

def test_rubric_is_general_and_parsimonious():
    """The reviewer prompt must enumerate the 5 universal elements and must
    NOT bake in any benchmark-specific term."""
    # exactly the five universal elements
    keys = [k for k, _ in REVIEW_ELEMENTS]
    assert keys == [
        "objective", "design_space", "ground_truth", "validity", "deliverable",
    ]
    low = REVIEWER_SYSTEM_PROMPT.lower()
    for k in keys:
        assert k.split("_")[0] in low, f"rubric missing element '{k}'"
    for term in _BENCHMARK_TERMS:
        assert term not in low, f"rubric overfits to benchmark term '{term}'"


# --- parsing -----------------------------------------------------------------

def test_parse_review_handles_code_fenced_json():
    raw = (
        "Here is my review:\n```json\n"
        + json.dumps({
            "elements": [
                {"element": "objective", "status": "ok", "note": "clear"},
                {"element": "validity", "status": "gap", "note": "no gates"},
            ],
            "summary": "mostly fine",
        })
        + "\n```\nDone."
    )
    review = parse_review(raw)
    assert len(review["elements"]) == 2
    gaps = review_gaps(review)
    assert [g["element"] for g in gaps] == ["validity"]


def test_parse_review_unparseable_yields_no_gaps():
    """A non-JSON reply must NOT manufacture gaps (advisory, never blocks)."""
    review = parse_review("I could not produce JSON, sorry.")
    assert review_gaps(review) == []


# --- advisory behaviour (mocked LLM) ----------------------------------------

def test_noninteractive_writes_report_and_proceeds_unchanged(tmp_path):
    run = _run(tmp_path, interactive=False)
    debug = tmp_path / "debug"
    debug.mkdir()
    adapter = _FakeAdapter(json.dumps({
        "elements": [
            {"element": "validity", "status": "gap",
             "note": "no feasibility gate is defined"},
        ],
        "summary": "needs a validity gate",
    }))

    problem = "Minimise drag of a widget."
    out = run._review_problem_statement(problem, debug, adapter=adapter)

    # proceeds unchanged (non-interactive never blocks, never edits)
    assert out == problem
    report = (debug / "problem_statement_review.md").read_text()
    assert "validity" in report
    assert "feasibility gate" in report


def test_interactive_refine_appends_clarifications(tmp_path, monkeypatch):
    run = _run(tmp_path, interactive=True)
    # interactive now requires a real TTY (a headless run is forced
    # non-interactive so input() can't block — see test_headless_interactive).
    # Under pytest there is no TTY, so force it on to exercise the refine path.
    run._interactive = True
    debug = tmp_path / "debug"
    debug.mkdir()
    adapter = _FakeAdapter(json.dumps({
        "elements": [
            {"element": "validity", "status": "gap",
             "note": "no feasibility gate"},
            {"element": "deliverable", "status": "gap",
             "note": "deliverable unstated"},
        ],
        "summary": "two gaps",
    }))

    answers = iter(["feasible iff stress < yield", ""])  # 2nd gap skipped
    monkeypatch.setattr("builtins.input", lambda *a, **k: next(answers))

    problem = "Minimise mass."
    out = run._review_problem_statement(problem, debug, adapter=adapter)

    assert out != problem
    assert "feasible iff stress < yield" in out
    assert "validity" in out
    # only the answered gap is appended; the skipped one is not
    assert "deliverable unstated" not in out
    assert (debug / "problem_statement_addendum.md").exists()


def test_complete_statement_no_gaps_no_prompt(tmp_path, monkeypatch):
    run = _run(tmp_path, interactive=True)
    debug = tmp_path / "debug"
    debug.mkdir()
    adapter = _FakeAdapter(json.dumps({
        "elements": [
            {"element": e, "status": "ok", "note": "present"}
            for e, _ in REVIEW_ELEMENTS
        ],
        "summary": "well posed",
    }))

    def _no_input(*a, **k):
        raise AssertionError("input() must not be called when there are no gaps")

    monkeypatch.setattr("builtins.input", _no_input)

    problem = "A fully specified problem."
    out = run._review_problem_statement(problem, debug, adapter=adapter)
    assert out == problem
    assert (debug / "problem_statement_review.md").exists()
    assert not (debug / "problem_statement_addendum.md").exists()


def test_reviewer_failure_never_blocks_the_run(tmp_path):
    run = _run(tmp_path, interactive=False)
    debug = tmp_path / "debug"
    debug.mkdir()

    class _Boom:
        model = "x"
        last_usage: dict = {}

        def invoke(self, messages):
            raise RuntimeError("LLM down")

    problem = "Some statement."
    # must return the original problem, not raise
    out = run._review_problem_statement(problem, debug, adapter=_Boom())
    assert out == problem


def test_reviewer_agent_is_ephemeral_and_toolless():
    agent = ProblemStatementReviewerAgent()
    assert agent.reset_on_checkpoint is True
    assert agent.tools == frozenset()
    assert agent.role == "reviewer"
    assert agent.system_prompt == REVIEWER_SYSTEM_PROMPT
