"""ReadProblemStatement — verbatim, on-demand read of PROBLEM_STATEMENT.md,
declaration-gated and uniform across every default agent (not just the
literature reviewer via the now-removed inject_problem_statement push flag).
"""
from __future__ import annotations

from a3dasm._src.backends.base import Agent, Edge, Graph
from a3dasm._src.delegation_log import DelegationLog
from a3dasm._src.nodes import StrategizerNode


class _Stub:
    def __init__(self) -> None:
        self.closure_tools: dict = {}
        self.last_usage: dict = {}
        self.model = "m"

    def invoke(self, messages):
        return ""


def _read_problem_statement(tmp_path, study_dir):
    run_dir = tmp_path / "runs" / "T1"
    (run_dir / "debug").mkdir(parents=True)

    class S(Agent):
        role = "strategizer"
        tools = frozenset({"Done"})
        description = "s"

    class Impl(Agent):
        role = "implementer"
        description = "i"
        tools = frozenset({"ReadProblemStatement"})

    spec = Graph(
        nodes={"strategizer": S(), "implementer": Impl()},
        edges=(Edge("strategizer", "implementer"),),
        entry="strategizer")
    dlog = DelegationLog(run_dir / "debug" / "delegation_log.jsonl")
    n = StrategizerNode(
        _Stub(), name="implementer", outgoing=[],
        spec=spec, worker_adapters={},
        notes_dir=None, delegation_log=dlog,
        study_dir=str(study_dir) if study_dir is not None else None,
    )
    return n._build_routing_closures()["ReadProblemStatement"]


def test_returns_file_contents_verbatim(tmp_path):
    study_dir = tmp_path / "study"
    study_dir.mkdir()
    text = "SUCCESS CRITERION: global min y=0 at (1, -2), don't stop early."
    (study_dir / "PROBLEM_STATEMENT.md").write_text(text, encoding="utf-8")
    read = _read_problem_statement(tmp_path, study_dir)
    assert read() == text


def test_errors_cleanly_when_file_missing(tmp_path):
    study_dir = tmp_path / "study"
    study_dir.mkdir()
    read = _read_problem_statement(tmp_path, study_dir)
    assert "ERROR" in read()
    assert "not found" in read()


def test_errors_cleanly_when_study_dir_unset(tmp_path):
    read = _read_problem_statement(tmp_path, None)
    assert "ERROR" in read()


def test_declared_uniformly_on_every_default_agent():
    """Every one of the 6 default agents declares ReadProblemStatement — the
    tool is uniform, not gated to whichever agent used to set
    inject_problem_statement=True."""
    from a3dasm._src.agents.critic import AdversarialCritiqueAgent
    from a3dasm._src.agents.datagenerator import DataGeneratorAgent
    from a3dasm._src.agents.debugger import DebuggerAgent
    from a3dasm._src.agents.implementer import F3dasmImplementerAgent
    from a3dasm._src.agents.literature import LiteratureReviewAgent
    from a3dasm._src.agents.strategizer import StrategizerAgent

    for cls in (
        StrategizerAgent, LiteratureReviewAgent, DataGeneratorAgent,
        F3dasmImplementerAgent, AdversarialCritiqueAgent, DebuggerAgent,
    ):
        assert "ReadProblemStatement" in cls.tools, (
            f"{cls.__name__} is missing ReadProblemStatement"
        )
