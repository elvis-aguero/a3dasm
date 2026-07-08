"""Default graph factory for f3dasm agentic runs.

5-node ratified topology:
  strategizer (entry)  →  literature_reviewer
                       →  datagenerator
                       →  implementer
                       →  critic
  datagenerator        →  literature_reviewer
  implementer          →  literature_reviewer
"""

from __future__ import annotations

from ..backends.base import Edge, Graph
from .critic import AdversarialCritiqueAgent
from .datagenerator import DataGeneratorAgent
from .implementer import F3dasmImplementerAgent
from .literature import LiteratureReviewAgent
from .strategizer import StrategizerAgent


def _default_graph() -> Graph:
    return Graph(
        nodes={
            "strategizer": StrategizerAgent(),
            "literature_reviewer": LiteratureReviewAgent(),
            "datagenerator": DataGeneratorAgent(),
            "implementer": F3dasmImplementerAgent(),
            "critic": AdversarialCritiqueAgent(),
        },
        edges=(
            Edge("strategizer", "literature_reviewer"),
            Edge("strategizer", "datagenerator"),
            Edge("strategizer", "implementer"),
            Edge("strategizer", "critic"),
            Edge("datagenerator", "literature_reviewer"),
            Edge("implementer", "literature_reviewer"),
        ),
        entry="strategizer",
    )
