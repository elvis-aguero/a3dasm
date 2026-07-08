"""Every specialist agent must declare its own role.

Regression for runs 20260708T021335 + 20260629T145434: LiteratureReviewAgent
and DebuggerAgent declared no `role`, so they inherited the base default
`Agent.role = "implementer"`. Every implementer-keyed branch then mis-fired on
them — most visibly the milestone-backlog nudge on delegations to the
literature_reviewer (diagnostics `MILESTONE_BLOCK node=literature_reviewer` in
both runs), and the Abaqus eval-parallelism resource nudge, and role telemetry.
A missing role must never impersonate the implementer.
"""
from __future__ import annotations

from a3dasm._src.agents import (
    AdversarialCritiqueAgent,
    DataGeneratorAgent,
    DebuggerAgent,
    F3dasmImplementerAgent,
    LiteratureReviewAgent,
    StrategizerAgent,
)
from a3dasm._src.backends.base import Agent


def test_every_specialist_declares_its_own_role():
    assert StrategizerAgent.role == "strategizer"
    assert F3dasmImplementerAgent.role == "implementer"
    assert DataGeneratorAgent.role == "datagenerator"
    assert AdversarialCritiqueAgent.role == "critic"
    assert LiteratureReviewAgent.role == "literature_reviewer"
    assert DebuggerAgent.role == "debugger"


def test_base_default_role_does_not_impersonate_a_privileged_role():
    # A bare Agent with no declared role must NOT be treated as the implementer
    # (which carries the milestone gate + eval-parallelism nudge).
    assert Agent.role != "implementer"
