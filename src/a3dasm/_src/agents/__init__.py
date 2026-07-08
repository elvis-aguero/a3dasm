"""Agent definitions for f3dasm agentic runs.

Each agent class lives in its own module with its system prompt inlined,
making agents ADAS-searchable via inspect.getsource().
"""

from ._graphs import _default_graph
from .critic import AdversarialCritiqueAgent
from .datagenerator import DataGeneratorAgent
from .debugger import DebuggerAgent
from .implementer import (
    F3dasmImplementer,
    F3dasmImplementerAgent,
    ImplementerAgent,
)
from .literature import LiteratureReviewAgent
from .strategizer import StrategizerAgent

__all__ = [
    "AdversarialCritiqueAgent",
    "DataGeneratorAgent",
    "F3dasmImplementerAgent",
    "StrategizerAgent",
    "F3dasmImplementer",
    "ImplementerAgent",  # backward-compatible alias
    "DebuggerAgent",
    "LiteratureReviewAgent",
    "_default_graph",
]
