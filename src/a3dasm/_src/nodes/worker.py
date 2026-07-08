"""WorkerNode: generic worker node for non-orchestrator agents."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..graph_state import AgenticState

from ..delegation_log import DelegationLog
from .base import AgentNode
from .parsing import _classify_response, _to_adapter_messages


class WorkerNode(AgentNode):
    """Generic worker node: executes tasks, writes Reports, returns to caller.

    Used for any non-orchestrator agent (Implementer, Debugger,
    LiteratureReviewer, etc.).  The node name in the graph is what
    distinguishes agents — not the node class.
    """

    def __init__(
        self,
        adapter: Any,
        study_dir: Any = None,
        workspace_dir: Any = None,
        delegation_log: DelegationLog | None = None,
        name: str = "worker",
        report_sections: tuple[str, ...] | None = None,
        agent_tools: frozenset[str] | None = None,
    ) -> None:
        super().__init__(adapter)
        self._name = name
        # The agent's declared tools — the single source of truth for which
        # capability closures this leaf worker is granted (read-only ledger/
        # store tools). Kept as a frozenset for membership checks.
        self._agent_tools: frozenset[str] = frozenset(agent_tools or ())
        # This agent's declared report sections (e.g. the critic's
        # Findings/Verdict, not the implementer's Conclusions/Files touched).
        # Used to validate the worker's report against ITS OWN contract instead
        # of the implementer-shaped default — otherwise a correct critic or
        # literature report is wrongly flagged malformed (audit BF-10/O40).
        self._report_sections = report_sections
        self._delegation_log = delegation_log
        self._evals_reported: dict = {}
        self._workspace_dir = Path(workspace_dir) if workspace_dir else None
        self._setup_sandboxed_write()
        self.adapter.closure_tools.update(self._build_eval_closures())
        if delegation_log is not None:
            self.adapter.closure_tools["RecallHistory"] = self._make_recall_history()
        # Declaration-gated read-only ledger/store tools — the SAME builder the
        # orchestrating nodes use, so a leaf worker (e.g. the critic) gets an
        # identical, working RecallStore/QueryStore/HypothesisList/Get surface
        # whenever it declares them. Resolves the run via the shared
        # AgentNode._resolve_run_dir (delegation-log path).
        from .tools.routing import build_declared_shared_closures
        self.adapter.closure_tools.update(
            build_declared_shared_closures(self, self._agent_tools))

    def _make_recall_history(self) -> Any:
        """Build the RecallHistory closure for this worker node."""
        node = self

        def RecallHistory(n: int = 5) -> str:
            """Return the last n delegations received by this node as (task, deliverable) pairs.
            Call at the start of a delegation to recall prior work. Returns oldest-first."""
            if node._delegation_log is None:
                return "No delegation log available."
            records = node._delegation_log.query_received(node._name, n)
            if not records:
                return "No prior delegations found."
            parts = []
            for i, r in enumerate(records, 1):
                parts.append(
                    f"== Prior delegation {i} ==\n"
                    f"Task: {r['task']}\n\n"
                    f"Deliverable:\n{r['deliverable']}"
                )
            return "\n\n---\n\n".join(parts)

        return RecallHistory

    def _setup_sandboxed_write(self) -> None:
        """Replace native Write with a workspace-sandboxed closure.

        Removes 'Write' from native_tools so the SDK doesn't expose it,
        then injects a closure that hard-rejects any path that resolves
        outside the workspace.  Path.resolve() collapses '..' and symlinks,
        so traversal attacks are blocked at the tool level, not just the prompt.

        Bash is kept native but cwd is already set to study_dir by the adapter;
        the prompt further constrains it to the workspace.
        """
        if self._workspace_dir is None:
            return  # no sandboxing if study_dir unknown (e.g. tests)

        workspace = self._workspace_dir.resolve()

        # Remove native Write so the SDK doesn't expose an unrestricted version
        if hasattr(self.adapter, "native_tools") and "Write" in self.adapter.native_tools:
            self.adapter.native_tools = [
                t for t in self.adapter.native_tools if t != "Write"
            ]

        def Write(path: str, body: str) -> str:
            """Write a file.  Restricted to workspace — no exceptions."""
            try:
                # Resolve against workspace so relative paths land there
                candidate = (workspace / path).resolve()
            except Exception as exc:  # noqa: BLE001
                return f"ERROR: invalid path {path!r}: {exc}"

            # Reject anything that escapes the delegations tree
            try:
                candidate.relative_to(workspace)
            except ValueError:
                return (
                    f"ERROR: write rejected — path resolves to {candidate}, "
                    f"which is outside the workspace ({workspace}). "
                    "Only paths inside the workspace are permitted."
                )

            candidate.parent.mkdir(parents=True, exist_ok=True)
            candidate.write_text(body, encoding="utf-8")
            return f"Written: {candidate}"

        self.adapter.closure_tools["Write"] = Write

    def _build_eval_closures(self) -> dict:
        evals = self._evals_reported

        def ReportEvals(count: int) -> str:
            """Report the number of function evaluations used in this task."""
            evals["count"] = int(count)
            return f"Recorded {count} evaluations."

        return {"ReportEvals": ReportEvals}

    def __call__(self, state: AgenticState) -> Any:
        from langchain_core.messages import AIMessage
        from langgraph.types import Command

        from ..agent_prompts import build_report_retry_prompt

        self._evals_reported.clear()
        messages = _to_adapter_messages(state["messages"])
        text = self.adapter.invoke(messages)

        _req_sections = (
            list(self._report_sections) if self._report_sections else None
        )
        diagnosis = _classify_response(text, _req_sections)
        if diagnosis is not None:
            # One retry — the correction prompt is built from THIS agent's own
            # report_sections so it can't command a structure that omits a
            # section the parser requires (e.g. the implementer's Retrospective).
            retry_messages = messages + [
                {"role": "ai", "content": text},
                {
                    "role": "user",
                    "content": (
                        f"{build_report_retry_prompt(_req_sections)}"
                        f"\n\nDiagnosis: {diagnosis}"
                    ),
                },
            ]
            text = self.adapter.invoke(retry_messages)

        ai_msg = AIMessage(content=text)
        evals_delta = self._evals_reported.get("count", 0)
        return_to = state.get("return_to")
        return Command(
            goto=return_to,
            update={
                "messages": [ai_msg],
                "last_report": text,
                "evals_used": state.get("evals_used", 0) + evals_delta,
            },
        )


# Backward-compatible alias — ImplementerNode is the WorkerNode used in the
# canonical 2-node topology.  New code should use WorkerNode directly.
ImplementerNode = WorkerNode
