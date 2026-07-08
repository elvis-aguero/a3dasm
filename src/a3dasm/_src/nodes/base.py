"""Base node class for ADAS-inspectable LangGraph nodes."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..graph_state import AgenticState
    from ..hypothesis_ledger import HypothesisLedger


class AgentNode:
    """Base class for ADAS-inspectable LangGraph nodes.

    Subclasses override __call__ to define routing topology.
    inspect.getsource(MyNode.__call__) reads the full routing logic.
    """

    def __init__(self, adapter: Any) -> None:
        self.adapter = adapter

    def __call__(self, state: AgenticState) -> Any:
        raise NotImplementedError

    # ── Run-context resolution (shared by every node type) ───────────────────
    # The read tools (RecallStore/QueryStore/HypothesisList/Get) may be granted
    # to any node — the entry strategizer, a delegating worker, or a leaf worker
    # such as the critic. They need the run's store/ledger paths, which are
    # resolved here so the tools work identically regardless of node type.
    def _resolve_run_dir(self) -> Path | None:
        """Best-effort run_dir, valid on every node type.

        The entry node sets _current_notes_dir (=run_dir/debug/
        strategizer_notes) inside __call__; workers never run __call__, so it
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
            from ..hypothesis_ledger import HypothesisLedger
            return HypothesisLedger(notes)
        return None
