"""The critic can search WITHIN a file (Grep), not just Read/Glob.

Regression: run 20260718T132852's critic-1/critic-3 had no way to search
delegation_log.jsonl once it exceeded Read's token cap, so they paged through
it blind, one line at a time, with offset=/limit= guesses (confirmed against
the raw transcript: one failed full-file Read, then 7 consecutive
single-line reads). Grep is already a native, read-only tool other roles
declare (datagenerator/implementer/debugger); the critic's own read-only
design principle (Read/Glob/RecallStore/QueryStore/OracleStatus/Hypothesis*,
never Bash/Edit/Write) is not violated by adding it — Grep searches, it
never mutates.
"""
from __future__ import annotations

from a3dasm._src.agents import AdversarialCritiqueAgent


def test_critic_declares_grep():
    assert "Grep" in AdversarialCritiqueAgent.tools


def test_critic_still_has_no_mutating_or_execution_tools():
    # Grep must not have been added alongside anything that breaks the
    # critic's read-only contract.
    forbidden = {"Bash", "Edit", "Write", "BashOutput", "KillShell"}
    assert not (AdversarialCritiqueAgent.tools & forbidden)
