"""Wet smoke test: the docs tutorials actually run end to end, for real.

Runs the exact problem shape taught in two docs tutorials — the Quickstart's
Branin example and Authoring a study's worked example (studies/example_study)
— through a real AgenticRun, on a free-tier OpenRouter model so this costs
nothing to run.

Marked `integration` and skipped cleanly without `OPENROUTER_API_KEY` set, so
a normal `pytest -m integration` run (no key configured) doesn't spuriously
fail. Intended trigger is the scheduled `wet_docs_smoke.yml` workflow (once
daily + manual dispatch), not the regular push/PR `Tests` workflow.

The bar this enforces is deliberately narrow: does the tutorial, followed
literally, crash? Whether the run closes GATED or UNGATED is NOT asserted —
a free/weaker model may legitimately not clear the full science gate, and
that is a model-capability signal, not a docs or runtime bug. Run this
manually with:
    OPENROUTER_API_KEY=... uv run pytest tests/test_docs_tutorials_wet.py \
        -v -s --no-cov -m integration
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration

# A genuinely free-tier OpenRouter model that supports tool-calling, confirmed
# live against https://openrouter.ai/api/v1/models at the time this was
# written. OpenRouter's free catalog shifts over time — if this model is
# retired, swap it for a current `:free` entry with "tools" in
# supported_parameters (query the API above to check).
_FREE_MODEL = "openai/gpt-oss-20b:free"

# Wall-clock and eval caps sized for a free-tier model on a trivial problem —
# tight enough that a stuck/looping run doesn't burn the whole scheduled slot.
_EVAL_BUDGET = 60
_WALLCLOCK_BUDGET_S = 20 * 60


def _require_openrouter_key() -> None:
    if not os.environ.get("OPENROUTER_API_KEY"):
        pytest.skip("OPENROUTER_API_KEY not set — wet docs-tutorial smoke test skipped")


def _run_and_check(study_dir: Path) -> None:
    """Execute study_dir through a real AgenticRun and assert it didn't crash.

    backend is config.yaml-only (AgenticRun has no backend= constructor
    kwarg — only model=), so the caller must have written/overwritten
    study_dir/config.yaml's `backend:` key to "openrouter" before calling
    this.
    """
    from a3dasm import AgenticRun

    report = AgenticRun(
        study_dir=study_dir,
        model=_FREE_MODEL,
        eval_budget=_EVAL_BUDGET,
        budget=_WALLCLOCK_BUDGET_S,
        interactive=False,
    ).execute()

    assert report, "AgenticRun.execute() returned an empty report"
    assert (study_dir / "pipeline.ipynb").exists(), (
        "tutorial did not produce pipeline.ipynb — the deliverable is missing"
    )

    runs_dir = study_dir / "runs"
    run_dirs = sorted(runs_dir.iterdir()) if runs_dir.exists() else []
    assert run_dirs, "no runs/<timestamp>/ directory was written"
    status_path = run_dirs[-1] / "run_status.json"
    assert status_path.exists(), f"run_status.json missing under {run_dirs[-1]}"

    # Informational only — NOT a pass/fail condition (see module docstring).
    import json

    status = json.loads(status_path.read_text()).get("status", "UNKNOWN")
    print(f"\n[wet docs smoke] {study_dir.name}: run closed as {status} "
          f"(GATED/UNGATED is informational here, not a failure condition)")


def test_quickstart_branin_runs_without_crashing(tmp_path):
    """Mirrors docs/notebooks/quickstart.ipynb's Branin example exactly."""
    _require_openrouter_key()

    study_dir = tmp_path / "quickstart_branin"
    study_dir.mkdir()
    (study_dir / "PROBLEM_STATEMENT.md").write_text(
        "Minimise the 2D Branin function over its standard domain.\n"
        "Report the best design found and the objective value there.\n"
    )
    (study_dir / "config.yaml").write_text("backend: openrouter\n")
    _run_and_check(study_dir)


def test_authoring_a_study_worked_example_runs_without_crashing(tmp_path):
    """Mirrors docs/authoring-a-study.md's worked example (studies/example_study),
    routed through OpenRouter instead of its own config.yaml's Claude default —
    the study's own PROBLEM_STATEMENT.md/evaluator.py are used unmodified."""
    _require_openrouter_key()

    src = Path(__file__).resolve().parent.parent / "studies" / "example_study"
    study_dir = tmp_path / "example_study"
    shutil.copytree(
        src, study_dir,
        ignore=shutil.ignore_patterns("runs", "__pycache__", "pipeline.ipynb"),
    )
    # The repo's config.yaml pins backend: claude / model: claude-haiku-...;
    # backend has no constructor override, so overwrite the COPY's
    # config.yaml (never the repo's own) to route this run through
    # OpenRouter instead. eval_budget/evaluator are unmodified.
    (study_dir / "config.yaml").write_text(
        "backend: openrouter\n"
        "eval_budget: 200\n"
        "evaluator:\n"
        '  entrypoint: "workspace/evaluator.py:evaluate"\n'
        "  output_names: [y]\n"
    )
    _run_and_check(study_dir)
