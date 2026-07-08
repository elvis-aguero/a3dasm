"""Critic GATE verdict parsing.

Regression: the gate regex `### Verdict\\s*\\n\\s*(\\w+)` could not skip the
markdown bold the critic actually emits (`### Verdict\\n\\n**PASS**`), so a
genuinely-earned PASS parsed as UNKNOWN and the run looped forever. The parser
must tolerate emphasis/punctuation and fall back to a `verdict: X` line.
"""
from __future__ import annotations

from a3dasm._src.nodes import _parse_verdict


def test_bold_pass_is_parsed():  # the exact bug from the wet run
    text = (
        "### Verdict\n\n**PASS** — no CRITICAL or MAJOR findings. The "
        "headline (-1.5220) is reproducible.\n\n### Numbers\nverdict: PASS\n"
    )
    assert _parse_verdict(text) == "PASS"


def test_bold_revise_is_parsed():
    text = "### Verdict\n\n**REVISE**\n\nH2 was not tested."
    assert _parse_verdict(text) == "REVISE"


def test_plain_verdict_still_parsed():  # backward compatible
    assert _parse_verdict("### Verdict\nPASS\n\n") == "PASS"
    assert _parse_verdict("### Verdict\n  REJECT") == "REJECT"


def test_numbers_line_fallback():
    # No parseable heading token, but a verdict: line exists.
    text = "### Verdict\n\n(see below)\n\n### Numbers\nverdict: PASS\n"
    assert _parse_verdict(text) == "PASS"


def test_bold_in_numbers_fallback():
    text = "blah\nverdict: **REVISE**\n"
    assert _parse_verdict(text) == "REVISE"


def test_unknown_when_no_verdict():
    assert _parse_verdict("## Report\nNo verdict here.\n") == "UNKNOWN"
    assert _parse_verdict("") == "UNKNOWN"


def test_garbage_token_is_not_a_false_verdict():
    # A word that isn't a valid verdict must not be returned as one.
    assert _parse_verdict("### Verdict\n\n**Maybe**\n") == "UNKNOWN"
