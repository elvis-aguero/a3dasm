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


# ---------------------------------------------------------------------------
# pipeline.ipynb is study-scoped, not run-scoped — _load_or_new_notebook
# (routing.py) loads a prior run's leftover wholesale, leaking stale cells
# into a fresh run's deliverable. A fresh (non-resume) run must archive it
# aside (never delete) before any node can touch study_dir.
# ---------------------------------------------------------------------------

def _seed_notebook(study: Path, run_id: str | None) -> None:
    import nbformat
    nb = nbformat.v4.new_notebook()
    nb.cells.append(nbformat.v4.new_markdown_cell("# stale prior-run content"))
    if run_id is not None:
        nb.metadata["agentic"] = {"run": f"{study}/runs/{run_id}"}
    nbformat.write(nb, str(study / "pipeline.ipynb"))


def test_fresh_run_archives_prior_pipeline_notebook_by_its_own_run_id(tmp_path):
    study = _make_study(tmp_path)
    _seed_notebook(study, "20260101T000000")

    run = AgenticRun(study_dir=study, interactive=False)

    class _Stub:
        def invoke(self, state, config=None):
            return {"last_report": "done", "evals_used": 0}

    run._graph = _Stub()
    run.execute()

    assert not (study / "pipeline.ipynb").exists()
    archived = study / "pipeline_20260101T000000.ipynb"
    assert archived.exists()
    import nbformat
    nb = nbformat.read(str(archived), as_version=4)
    assert "stale prior-run content" in nb.cells[0]["source"]


def test_fresh_run_archives_notebook_with_no_provenance_as_unknown(tmp_path):
    study = _make_study(tmp_path)
    _seed_notebook(study, None)  # never stamped — no agentic.run metadata

    run = AgenticRun(study_dir=study, interactive=False)

    class _Stub:
        def invoke(self, state, config=None):
            return {"last_report": "done", "evals_used": 0}

    run._graph = _Stub()
    run.execute()

    assert not (study / "pipeline.ipynb").exists()
    assert (study / "pipeline_unknown.ipynb").exists()


def test_resume_does_not_archive_the_notebook(tmp_path):
    study = _make_study(tmp_path)
    _seed_notebook(study, "prior-run")

    run1 = AgenticRun(study_dir=study, interactive=False)

    class _Stub1:
        def invoke(self, state, config=None):
            return {"last_report": "done", "evals_used": 0}

    run1._graph = _Stub1()
    run1.execute()
    # run1 is a FRESH run too — it archives the pre-seeded notebook before
    # doing anything else, exactly like the tests above.
    assert not (study / "pipeline.ipynb").exists()
    run_dir = next((study / "runs").iterdir())

    # Re-seed a notebook as if THIS run (run_dir) authored it, then resume.
    _seed_notebook(study, run_dir.name)

    run2 = AgenticRun(study_dir=study, interactive=False, resume_from=run_dir)

    class _Stub2:
        def invoke(self, state, config=None):
            return {"last_report": "resumed", "evals_used": 0}

    run2._graph = _Stub2()
    run2.execute()

    # Resume must NOT archive/rename — it keeps working on the same notebook.
    assert (study / "pipeline.ipynb").exists()
    assert not (study / f"pipeline_{run_dir.name}.ipynb").exists()


# ---------------------------------------------------------------------------
# BACKLOG #35 — invoke(None, config) on an already-terminal checkpoint
# (every normal close: GATED/UNGATED/FAILED all reach Command(goto=END)) is a
# genuine LangGraph no-op: no node re-runs, no new model call, the stale
# last_report is handed back verbatim. resume_from must detect this via
# graph.get_state(config).next and force real re-execution with fresh input
# instead — except when the run already closed cleanly (GATED) with an
# unchanged PROBLEM_STATEMENT.md, where there is genuinely nothing new to do.
# ---------------------------------------------------------------------------

class _FakeSnapshot:
    def __init__(self, next_tasks):
        self.next = next_tasks


def _run_to_close(study, status_extra=None):
    """Execute a fresh run whose stub reports terminal state, closing it
    with the given run_status.json contents (status defaults to GATED via
    the real _gate_outcome computation off last_report text)."""
    run = AgenticRun(study_dir=study, interactive=False)

    class _Stub:
        def invoke(self, state, config=None):
            return {"last_report": "done", "evals_used": 0}

    run._graph = _Stub()
    run.execute()
    run_dir = next((study / "runs").iterdir())
    if status_extra:
        import json as _json
        status_path = run_dir / "debug" / "run_status.json"
        data = _json.loads(status_path.read_text())
        data.update(status_extra)
        status_path.write_text(_json.dumps(data))
    return run_dir


def test_resume_refuses_a_cleanly_closed_unchanged_run(tmp_path):
    study = _make_study(tmp_path)
    run_dir = _run_to_close(study)  # closes GATED (no UNGATED banner in text)

    run2 = AgenticRun(study_dir=study, interactive=False, resume_from=run_dir)

    class _Stub2:
        def get_state(self, config):
            return _FakeSnapshot(())  # terminal: nothing pending

        def invoke(self, state, config=None):
            raise AssertionError("must not invoke — nothing to do")

    run2._graph = _Stub2()
    try:
        run2.execute()
    except Exception as exc:  # AgenticRunError
        assert "nothing new for this run to do" in str(exc)
    else:
        raise AssertionError("expected resume to refuse")


def test_resume_forces_fresh_execution_after_external_stop(tmp_path):
    study = _make_study(tmp_path)
    run_dir = _run_to_close(
        study, status_extra={"status": "UNGATED", "stop_reason": "org_spend_limit"}
    )

    run2 = AgenticRun(study_dir=study, interactive=False, resume_from=run_dir)
    seen = {}

    class _Stub2:
        def get_state(self, config):
            return _FakeSnapshot(())  # terminal

        def invoke(self, state, config=None):
            seen["state"] = state
            return {"last_report": "resumed for real", "evals_used": 0}

    run2._graph = _Stub2()
    run2.execute()

    assert seen["state"] is not None  # NOT None — real fresh input, not a no-op
    assert seen["state"]["done"] is False
    msg = seen["state"]["messages"][0].content
    assert "org_spend_limit" in msg
    assert "[RESUME]" in msg


def test_resume_forces_fresh_execution_when_problem_statement_changed(tmp_path):
    study = _make_study(tmp_path)
    run_dir = _run_to_close(study)  # closes GATED

    (study / "PROBLEM_STATEMENT.md").write_text("# a genuinely new statement\n")

    run2 = AgenticRun(study_dir=study, interactive=False, resume_from=run_dir)
    seen = {}

    class _Stub2:
        def get_state(self, config):
            return _FakeSnapshot(())  # terminal

        def invoke(self, state, config=None):
            seen["state"] = state
            return {"last_report": "resumed for real", "evals_used": 0}

    run2._graph = _Stub2()
    run2.execute()  # must NOT raise, even though prior status was GATED

    assert seen["state"] is not None
    msg = seen["state"]["messages"][0].content
    assert "a genuinely new statement" in msg


def test_resume_replays_checkpoint_when_genuinely_mid_flight(tmp_path):
    """A real crash/kill leaves pending tasks (`.next` non-empty) — that
    path is unaffected by the terminal-detection logic and still passes
    None, letting LangGraph resume the interrupted node itself."""
    study = _make_study(tmp_path)
    run_dir = _run_to_close(study)

    run2 = AgenticRun(study_dir=study, interactive=False, resume_from=run_dir)
    seen = {}

    class _Stub2:
        def get_state(self, config):
            return _FakeSnapshot(("some_pending_node",))  # NOT terminal

        def invoke(self, state, config=None):
            seen["state"] = state
            return {"last_report": "resumed", "evals_used": 0}

    run2._graph = _Stub2()
    run2.execute()

    assert seen["state"] is None  # unchanged: plain checkpoint replay


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
