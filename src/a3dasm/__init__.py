"""Public API for the agentic-f3dasm layer.

Architecture: LangGraph StateGraph with a 5-node default topology:
- **StrategizerAgent** — entry; forms hypotheses, plans, synthesises.
- **LiteratureReviewAgent** — methodology from primary literature.
- **DataGeneratorAgent** — BUILDS the physics DataGenerator Block.
- **F3dasmImplementerAgent** — RUNS the f3dasm pipeline end-to-end:
  DoE-execution (sampling), data-generation runs, ML, Optimization.
  The ONLY agent that evaluates designs.
- **AdversarialCritiqueAgent** — final adversarial quality gate.

The user's only required input is ``<study-dir>/PROBLEM_STATEMENT.md``.
"""

from __future__ import annotations

from ._src.agent_runtime import (
    DEFAULT_MODEL,
    AgenticRun,
    AgenticRunError,
    Delegation,
    Report,
    StudyConfig,
    Task,
)
from ._src.agents import (
    AdversarialCritiqueAgent,
    DataGeneratorAgent,
    DebuggerAgent,
    F3dasmImplementerAgent,
    ImplementerAgent,
    LiteratureReviewAgent,
    StrategizerAgent,
)
from ._src.backends.base import Agent, Edge, Graph
from ._src.backends.claude import ClaudeAdapter
from ._src.backends.ollama import OllamaAdapter
from ._src.graph_builder import build_graph
from ._src.graph_state import AgenticState
# Only get_evaluator() is agent-facing — the ONE door to the registered
# oracle. InstrumentedDataGenerator stays internal (constructed solely inside
# get_evaluator); it is deliberately not re-exported so agents cannot build a
# store-redirected evaluator. See KB 0001.
from ._src.instrumented import get_evaluator, load_experiments
from ._src.lookup import LookupDataGenerator
from ._src.nodes import (
    AgentNode,
    ImplementerNode,
    StrategizerNode,
    WorkerNode,
)
from ._src.optimizer import AgenticOptimizerAdapter

__author__ = "Elvis Aguero (elvis_alexander_aguero_vera@brown.edu)"
__credits__ = ["Elvis Aguero"]
__status__ = "Experimental"

__all__ = [
    "Agent",
    "AgentNode",
    "get_evaluator",
    "load_experiments",
    "AgenticOptimizerAdapter",
    "DataGeneratorAgent",
    "F3dasmImplementerAgent",
    "AgenticRun",
    "AgenticRunError",
    "AgenticState",
    "ClaudeAdapter",
    "DEFAULT_MODEL",
    "Delegation",
    "Edge",
    "Graph",
    "AdversarialCritiqueAgent",
    "DebuggerAgent",
    "ImplementerAgent",
    "LiteratureReviewAgent",
    "ImplementerNode",
    "WorkerNode",
    "LookupDataGenerator",
    "OllamaAdapter",
    "Report",
    "StrategizerAgent",
    "StrategizerNode",
    "StudyConfig",
    "Task",
    "build_graph",
]
