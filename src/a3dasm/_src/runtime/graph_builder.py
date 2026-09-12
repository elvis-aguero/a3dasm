"""LangGraph StateGraph builder for f3dasm agentic runs."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph

from ..backends.base import Agent, Graph
from ..infra.delegation_log import DelegationLog
from ..nodes import Node
from .graph_state import AgenticState

__all__ = ["build_graph"]


def build_graph(
    spec: Graph,
    make_adapter: Callable[[str, Agent], Any],
    checkpointer: Any = None,
    study_dir: Any = None,
    interactive: bool = False,
    max_ask: int = 1,
    notes_dir: Any = None,
    workspace_dir: Any = None,
    delegation_log: DelegationLog | None = None,
) -> Any:
    """Build and compile a LangGraph StateGraph from a Graph spec.

    Parameters
    ----------
    spec : Graph
        Agent graph specification (nodes, edges, entry).
    make_adapter : callable
        ``(name: str, agent: Agent) -> adapter`` — factory that produces a
        ``ClaudeAdapter`` or ``OllamaAdapter`` for the given node.
    checkpointer : any, optional
        LangGraph checkpointer.  Defaults to an in-memory :class:`MemorySaver`.
    delegation_log : DelegationLog, optional
        Graph-wide delegation log for episodic memory (RecallHistory tool).

    Returns
    -------
    CompiledGraph
        A compiled LangGraph graph ready to invoke.
    """
    builder = StateGraph(AgenticState)

    # ONE adapter per named node — shared across all orchestrating nodes.
    node_adapters = {n: make_adapter(n, spec.nodes[n]) for n in spec.nodes}

    for name, agent in spec.nodes.items():
        adapter = node_adapters[name]  # shared instance, NOT make_adapter() again
        outgoing = spec.outgoing(name)

        # ONE node class. Whether it orchestrates or answers follows from
        # `outgoing`, and what it may DO follows from what its Agent declares
        # in `tools` — never from its type. notes_dir is passed only to the
        # entry node: it owns the run's hypothesis + milestone ledgers.
        node = Node(
            adapter,
            name=name,
            outgoing=outgoing,
            spec=spec,
            study_dir=study_dir,
            interactive=interactive,
            max_ask=max_ask,
            worker_adapters={n: node_adapters[n] for n in outgoing},
            notes_dir=notes_dir if name == spec.entry else None,
            workspace_dir=workspace_dir,
            delegation_log=delegation_log,
            report_sections=getattr(agent, "report_sections", None),
            agent_tools=getattr(agent, "tools", None),
        )

        builder.add_node(name, node)

    builder.set_entry_point(spec.entry)

    return builder.compile(checkpointer=checkpointer or MemorySaver())
