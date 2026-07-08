"""The f3dasm data-driven process phases — promoted from prose to a real enum.

These are the canonical stages a data-driven study moves through. A delegation
may carry one as `phase` to declare its intent in the larger process; the tag is
the shared vocabulary that timing/telemetry, milestone gates (C2), and the
critic key off. The tag is always OPTIONAL — `None` (untagged) is valid, and an
unrecognised string resolves to `None` (a soft note, never a refusal), mirroring
the forgiving `resolve_target` contract.
"""

from __future__ import annotations

import re
from enum import Enum

__all__ = ["Phase", "resolve_phase"]


class Phase(str, Enum):
    LITERATURE = "literature"          # survey / DoE methodology grounding
    DOE = "doe"                        # design of experiments (the domain + sampling plan)
    DATA_GENERATION = "data_generation"  # evaluating designs through the oracle
    ML = "ml"                          # fitting a surrogate
    OPTIMIZATION = "optimization"      # surrogate-guided search for better designs
    SETUP = "setup"                    # infra: author/register the oracle, scaffolding


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


# Minimal, normalized synonyms for the handful of genuinely common spellings.
# Not a brittle target-routing hack: phases are a fixed, closed vocabulary, and
# these only absorb spelling/abbreviation variants of the SAME six members.
_SYNONYMS = {
    "designofexperiments": Phase.DOE,
    "domain": Phase.DOE,
    "datagen": Phase.DATA_GENERATION,
    "datageneration": Phase.DATA_GENERATION,
    "machinelearning": Phase.ML,
    "surrogate": Phase.ML,
    "optimisation": Phase.OPTIMIZATION,
    "opt": Phase.OPTIMIZATION,
    "litreview": Phase.LITERATURE,
    "literaturereview": Phase.LITERATURE,
}


def resolve_phase(value) -> Phase | None:
    """Resolve a loose phase string/Phase to a `Phase`, or `None` if unknown.

    Accepts a `Phase`, an exact value, a normalized value, or a curated spelling
    variant. Never raises — an unknown value returns `None` (soft)."""
    if value is None or isinstance(value, Phase):
        return value
    n = _norm(str(value))
    if not n:
        return None
    for p in Phase:
        if _norm(p.value) == n or _norm(p.name) == n:
            return p
    return _SYNONYMS.get(n)
