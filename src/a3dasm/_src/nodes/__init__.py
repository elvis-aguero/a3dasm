"""Agentic graph nodes. Public surface for the nodes package.

One class: :class:`~.node.Node`. A node has outgoing edges or it does not, and
nothing else about the node layer varies by agent — see ``node.py``.
"""

# Intentionally re-exports private helpers (_*) needed by tests and internal callers.
from .node import Node
from .parsing import (  # noqa: F401
    _classify_response,
    _consult_handbook,
    _extract_report_section,
    _parse_verdict,
    _reconcile_delegation_evals,
    _resolve_delegation_evals,
    _stamped_eval_count,
    _to_adapter_messages,
)
from .tools.routing import (
    _EXIT_INTERVIEW,  # noqa: F401 – canonical def in routing/feedback.py
)

__all__ = [
    "Node",
    "_EXIT_INTERVIEW",
    "_classify_response",
    "_consult_handbook",
    "_extract_report_section",
    "_parse_verdict",
    "_reconcile_delegation_evals",
    "_resolve_delegation_evals",
    "_stamped_eval_count",
    "_to_adapter_messages",
]
