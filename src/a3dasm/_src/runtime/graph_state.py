"""State types for the LangGraph-based agentic runtime."""

from __future__ import annotations

from dataclasses import dataclass

from langgraph.graph import MessagesState


class AgenticState(MessagesState):
    """LangGraph state for one agentic run.

    Inherits ``messages: Annotated[list[AnyMessage], add_messages]``
    from :class:`~langgraph.graph.MessagesState`.
    """

    study_dir: str
    done: bool
    last_report: str | None
    total_delegations: int
    budget_seconds: float | None
    budget_usd: float | None     # hard USD cost ceiling; None = no USD ceiling
    run_dir: str | None          # absolute path to runs/<timestamp>/
    eval_budget: int | None      # max function evaluations (from config)
    evals_used: int              # running count across delegations
    # time.time() at run start, for budget enforcement
    start_time: float | None
    return_to: str | None
    # paths relative to study_dir; checked before Done accepted
    required_deliverables: list | None
    # Token usage accumulated across all agents; set by StrategizerNode on Done
    token_totals: dict | None
    # Per-node tool-call error counts (ERROR: returns + exceptions)
    error_counts: dict | None
    # Canonical run-level ExperimentData store project_dir.
    # Set by AgenticRun.execute() so nodes and workers can locate the
    # shared store without re-deriving it from run_dir.
    experiment_data_dir: str | None


# ---------------------------------------------------------------------------
# Data-transfer objects (moved from agent_runtime)
# ---------------------------------------------------------------------------


@dataclass
class Task:
    """A unit of work delegated from one agent to another."""

    intent: str
    expected_report: str
    target: str


@dataclass
class Report:
    """The result returned by an implementer agent."""

    content: str
    target: str | None = None


@dataclass
class Delegation:
    """A delegation request parsed from an agent's response."""

    target: str
    task: str
    expected_report: str = ""
    # Optional design namespace this delegation operates in (Axis 3a). None →
    # the canonical single-study oracle + ledger (today's behavior). A non-None
    # value scopes the worker to that namespace's oracle via F3DASM_NAMESPACE.
    namespace: str | None = None


@dataclass
class StudyConfig:
    """Lightweight config loaded from the study directory."""

    study_dir: str
    model: str = "claude-opus-4-5"
    budget_seconds: float | None = None
