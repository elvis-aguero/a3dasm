"""``Node`` — the one kind of thing the agent graph is made of.

There is no hierarchy of node types here, and deliberately so. A node has
outgoing edges or it does not, and that single fact decides everything that
differs between two nodes:

    def __call__(self, state):
        return self._orchestrate(state) if self._outgoing else self._respond(state)

Everything else a node can do comes from what its ``Agent`` DECLARES in
``tools`` (see ``nodes/tools/routing``), never from what it is. The
strategizer, the implementer and the critic are all this class; they differ by
the name they are registered under, the prompt they carry, and the tools they
declare — which is exactly how a reader should think about them.

The two behaviours live in ``orchestration`` and ``leaf``; the remaining
mixins are per-concern, not per-type: ``recording`` (notes, retrospectives),
``critic_gate`` (the adversarial review), ``lifecycle`` (budget/backstop
halts), ``reproduction_gate`` (the notebook's lazy-reproduction
check). The ledger tools are a tool family like any other and live in
``nodes/tools/routing/ledger.py``.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..epistemics.hypothesis_ledger import HypothesisLedger
    from ..runtime.graph_state import AgenticState

from ..infra.delegation_log import DelegationLog
from .critic_gate import CriticGateMixin
from .leaf import LeafMixin
from .lifecycle import LifecycleMixin
from .orchestration import OrchestrationMixin
from .recording import RecordingMixin
from .reproduction_gate import ReproductionGateMixin


class Node(
    RecordingMixin,
    CriticGateMixin,
    LifecycleMixin,
    ReproductionGateMixin,
    OrchestrationMixin,
    LeafMixin,
):
    """One node in the agent graph.

    Parameters
    ----------
    adapter
        The LLM adapter this node speaks through.
    name
        The node's name in the graph. This is what distinguishes one agent
        from another — there is no per-agent node class.
    outgoing
        Names of the nodes this one may delegate to. Non-empty makes this an
        orchestrating node; empty makes it a leaf. That is the only structural
        choice in the node layer.
    spec
        The whole :class:`~..backends.base.Graph`, so a node can read its
        peers' roles and descriptions.
    report_sections, agent_tools
        This agent's own declared report contract and toolset, used to
        validate its reports and to gate which capability closures it is
        granted.

    ``inspect.getsource(Node.__call__)`` reads the routing topology.
    """

    def __init__(
        self,
        adapter: Any,
        *,
        name: str = "node",
        outgoing: list[str] | tuple[str, ...] = (),
        spec: Any = None,
        study_dir: Any = None,
        interactive: bool = False,
        max_ask: int = 1,
        worker_adapters: dict | None = None,
        notes_dir: Any = None,
        workspace_dir: Any = None,
        delegation_log: DelegationLog | None = None,
        report_sections: tuple[str, ...] | None = None,
        agent_tools: frozenset[str] | None = None,
    ) -> None:
        self.adapter = adapter
        self._name = name
        self._outgoing = list(outgoing)
        self._spec = spec
        self._init_recording()
        # Only one of the two initialisers runs: a leaf never allocates the
        # delegation registry, its locks or its ledgers, and an orchestrating
        # node never installs the leaf's single-turn report contract.
        if self._outgoing:
            self._init_orchestration(
                name=name, outgoing=self._outgoing, spec=spec,
                study_dir=study_dir, interactive=interactive, max_ask=max_ask,
                worker_adapters=worker_adapters, notes_dir=notes_dir,
                workspace_dir=workspace_dir, delegation_log=delegation_log,
            )
        else:
            self._init_leaf(
                study_dir=study_dir, workspace_dir=workspace_dir,
                delegation_log=delegation_log, name=name,
                report_sections=report_sections, agent_tools=agent_tools,
            )

    def __call__(self, state: AgenticState) -> Any:
        """Take one turn. Which turn depends only on whether this node can delegate."""
        return (
            self._orchestrate(state) if self._outgoing
            else self._respond(state)
        )

    def _init_recording(self) -> None:
        """Establish the state ``RecordingMixin`` writes to, on EVERY node.

        Recording is a property of a node — any node's tools can return an
        ERROR, and any node's adapter reports token usage — so its substrate
        belongs here rather than in one behaviour's setup. It used to be
        created only by the orchestration setup, which was invisible while
        leaves were a separate class that did not carry RecordingMixin at all:
        the moment a leaf reached any recording call it raised AttributeError
        on ``_registry_lock``. The orchestration setup still assigns these
        itself, to the same values, so an orchestrating node is unchanged.
        """
        self._registry_lock = threading.Lock()
        self._notifications: list[str] = []
        self._notifications_lock = threading.Lock()
        self._error_counts: dict[str, int] = {}
        self._token_totals: dict = {
            "input_tokens": 0,
            "output_tokens": 0,
            "cache_read_input_tokens": 0,
            "cache_creation_input_tokens": 0,
            "total_cost_usd": 0.0,
        }
        self._cost_observed: bool = False
        self._current_notes_dir: Path | None = None
        self._telemetry: Any = None

    # ── Run-context resolution (shared by every node) ────────────────────────
    # The read tools (RecallStore/QueryStore/HypothesisList/Get) may be granted
    # to any node — the entry node, a mid-tier delegating node, or a leaf such
    # as the critic. They need the run's store/ledger paths, which are resolved
    # here so the tools work identically wherever they are granted.
    def _resolve_run_dir(self) -> Path | None:
        """Best-effort run_dir, valid on any node.

        An orchestrating node sets _current_notes_dir (=run_dir/debug/
        strategizer_notes) inside its turn; a leaf never takes that path, so it
        stays unset. Every node DOES hold the shared delegation log at
        run_dir/debug/delegation_log.jsonl, so derive run_dir from that when
        the notes dir is unavailable.
        """
        notes = getattr(self, "_current_notes_dir", None)
        if notes is not None:
            return Path(notes).parent.parent
        dlog = getattr(self, "_delegation_log", None)
        p = getattr(dlog, "_path", None)
        return Path(p).parent.parent if p is not None else None

    def _read_ledger(self) -> HypothesisLedger | None:
        """The hypothesis ledger for READ access, resolved on any node.

        Returns the node's own bound ledger when it has one (the entry node);
        otherwise resolves a read-only view from the run's strategizer_notes
        when hypotheses.json exists. HypothesisLedger.__init__ performs no I/O,
        and only READ callers use this, so there is no write race with the
        entry node that owns the file.
        """
        own = getattr(self, "_ledger", None)
        if own is not None:
            return own
        rd = self._resolve_run_dir()
        if rd is None:
            return None
        notes = rd / "debug" / "strategizer_notes"
        if (notes / "hypotheses.json").exists():
            from ..epistemics.hypothesis_ledger import HypothesisLedger
            return HypothesisLedger(notes)
        return None
