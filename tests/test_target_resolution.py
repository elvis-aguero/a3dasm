"""Forward-compatible delegation-target resolution.

The strategizer names a delegation target; resolve_target maps it to a live
graph node by exact name, normalized name (case/separator-insensitive), or
normalized ROLE — surviving node renames as long as the role is stable. There
is NO hardcoded capability-synonym table: the root-cause fix is that the
strategizer prompt names targets by their hint/role, so it never invents a
'pipeline'/'oracle' target. An unresolvable target returns None (the caller
errors with the valid-target list, and the agent self-corrects).
"""
from __future__ import annotations

from a3dasm._src.nodes.tools.routing import resolve_target

# Canonical 5-node topology; here role == node name.
_OUT = ["implementer", "datagenerator", "literature_reviewer", "critic"]
_ROLES = {n: n for n in _OUT}


def test_exact_and_case_separator_normalization():
    assert resolve_target("implementer", _OUT, _ROLES) == "implementer"
    assert resolve_target("Implementer", _OUT, _ROLES) == "implementer"
    assert resolve_target("literature-reviewer", _OUT, _ROLES) == "literature_reviewer"
    assert resolve_target("LITERATURE_REVIEWER", _OUT, _ROLES) == "literature_reviewer"


def test_resolves_by_role_when_node_renamed():
    """Forward-compatible: a renamed implementer node (different name, same
    role) still resolves when the target is named by its role."""
    out = ["doer", "critic"]
    roles = {"doer": "implementer", "critic": "critic"}
    assert resolve_target("implementer", out, roles) == "doer"  # by role
    assert resolve_target("doer", out, roles) == "doer"  # by exact name


def test_capability_synonyms_no_longer_resolve():
    """The hardcoded synonym table is gone. Capability words that are neither a
    node name nor a role do NOT resolve — the caller errors and the agent
    self-corrects to a real hint name."""
    for syn in ("pipeline", "pipeline_executor", "datagen", "oracle"):
        assert resolve_target(syn, _OUT, _ROLES) is None, syn


def test_unresolvable_returns_none():
    assert resolve_target("nonsense_xyz", _OUT, _ROLES) is None
    assert resolve_target("", _OUT, _ROLES) is None


def test_exact_name_match_takes_precedence():
    # if a real node is literally named 'pipeline', exact match resolves it
    out = ["pipeline", "implementer", "critic"]
    roles = {"pipeline": "pipeline", "implementer": "implementer", "critic": "critic"}
    assert resolve_target("pipeline", out, roles) == "pipeline"
