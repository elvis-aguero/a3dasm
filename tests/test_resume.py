"""Tests for durable checkpoint + resume in AgenticRun."""
from pathlib import Path

from a3dasm._src.agent_runtime import AgenticRun


def _make_study(tmp_path: Path) -> Path:
    study = tmp_path / "study"
    study.mkdir()
    (study / "PROBLEM_STATEMENT.md").write_text("# trivial\n")
    return study


def test_execute_persists_thread_id_and_checkpoint_db(tmp_path, monkeypatch):
    study = _make_study(tmp_path)
    run = AgenticRun(study_dir=study, interactive=False)

    seen = {}

    class _StubGraph:
        def invoke(self, state, config=None):
            seen["config"] = config
            seen["state"] = state
            return {"last_report": "done", "evals_used": 0}

    run._graph = _StubGraph()
    run.execute()

    # recursion_limit raised well past the LangGraph default of 25
    assert seen["config"].get("recursion_limit", 25) >= 200
    tid = seen["config"]["configurable"]["thread_id"]
    assert seen["state"] is not None  # fresh run passes initial state

    run_dir = next((study / "runs").iterdir())
    assert (run_dir / "debug" / "thread_id").read_text().strip() == tid
    assert (run_dir / "debug" / "checkpoints.sqlite").exists()


def test_resume_from_reuses_thread_id_and_passes_none(tmp_path, monkeypatch):
    study = _make_study(tmp_path)

    run1 = AgenticRun(study_dir=study, interactive=False)
    seen1 = {}

    class _Stub1:
        def invoke(self, state, config=None):
            seen1["config"] = config
            return {"last_report": "done", "evals_used": 0}

    run1._graph = _Stub1()
    run1.execute()
    run_dir = next((study / "runs").iterdir())
    first_tid = seen1["config"]["configurable"]["thread_id"]

    run2 = AgenticRun(study_dir=study, interactive=False, resume_from=run_dir)
    seen2 = {}

    class _Stub2:
        def invoke(self, state, config=None):
            seen2["state"] = state
            seen2["config"] = config
            return {"last_report": "resumed", "evals_used": 0}

    run2._graph = _Stub2()
    run2.execute()

    assert seen2["config"]["configurable"]["thread_id"] == first_tid
    assert seen2["state"] is None  # resume → None input replays checkpoint
    # reused the same run dir, did not create a second
    assert len(list((study / "runs").iterdir())) == 1


def test_resume_refreshes_budgets_into_checkpoint(tmp_path):
    """On resume the new budgets/clock are re-seeded into the checkpointed
    state via update_state, so a run that halted on a budget can progress
    once the user raises it."""
    study = _make_study(tmp_path)

    run1 = AgenticRun(study_dir=study, interactive=False)

    class _Stub1:
        def invoke(self, state, config=None):
            return {"last_report": "done", "evals_used": 0}

    run1._graph = _Stub1()
    run1.execute()
    run_dir = next((study / "runs").iterdir())

    run2 = AgenticRun(
        study_dir=study, interactive=False, resume_from=run_dir,
        budget_usd=5.0, budget=123.0,
    )
    updates = {}

    class _Stub2:
        def update_state(self, config, values):
            updates.update(values)

        def invoke(self, state, config=None):
            return {"last_report": "resumed", "evals_used": 0}

    run2._graph = _Stub2()
    run2.execute()

    assert updates["budget_usd"] == 5.0
    assert updates["budget_seconds"] == 123.0
    assert updates["start_time"] is not None


def test_invoke_crash_writes_resumable_status(tmp_path):
    """An unhandled crash during graph.invoke records a resumable run_status
    (so resume_from is always an option) and then re-raises."""
    study = _make_study(tmp_path)
    run = AgenticRun(study_dir=study, interactive=False)

    class _Boom:
        def invoke(self, state, config=None):
            raise RuntimeError("kaboom")

    run._graph = _Boom()
    try:
        run.execute()
    except RuntimeError as exc:
        assert "kaboom" in str(exc)
    else:
        raise AssertionError("expected the crash to propagate")

    import json
    run_dir = next((study / "runs").iterdir())
    status = json.loads(
        (run_dir / "debug" / "run_status.json").read_text()
    )
    assert status["status"] == "crashed"
    assert status["resumable"] is True
    assert status["thread_id"]


def test_resume_from_missing_marker_raises(tmp_path):
    study = _make_study(tmp_path)
    bogus = study / "runs" / "nonexistent"
    bogus.mkdir(parents=True)
    run = AgenticRun(study_dir=study, interactive=False, resume_from=bogus)
    try:
        run.execute()
    except Exception as exc:  # AgenticRunError
        assert "resumable" in str(exc)
    else:
        raise AssertionError("expected resume to fail on missing thread_id")
