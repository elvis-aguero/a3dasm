"""Fault classification (system vs agent) for the diagnostics KPI.

Regression for run 20260630T164908: the classifier substring-matched the whole
message for words like "retry"/"connection", so a tool's own advice text
("...ShowNotebook('analysis') and retry.") tagged an AGENT error as "system",
corrupting the agent-vs-system error KPI the run-analysis protocol relies on.
"""
from __future__ import annotations

from a3dasm._src.nodes.recording import _classify_fault


def test_agent_error_with_advice_word_is_not_system():
    # tool advice containing "retry" must NOT be tagged system
    assert _classify_fault("", "ERROR: `old` not found. ShowNotebook() and retry.") == "agent"
    # bad-args / wrong-usage messages stay agent even if they mention connections/timeouts
    assert _classify_fault("ValueError", "expected_rev required; the connection to the cell is stale") == "agent"


def test_genuine_system_errors_are_system():
    assert _classify_fault("ConnectionError", "boom") == "system"
    assert _classify_fault("ReadTimeout", "boom") == "system"
    assert _classify_fault("", "HTTP 429 rate limit exceeded") == "system"
    assert _classify_fault("", "Anthropic API overloaded, please retry") == "system"
    assert _classify_fault("", "503 Service Unavailable") == "system"
