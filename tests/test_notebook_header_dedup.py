"""Regression: the notebook cell tools prepend the canonical heading themselves,
so when the author ALSO opened their content with a header the cell rendered a
duplicated heading (run 20260623T212346: '## Hypotheses\n\n## Hypotheses',
'### analysis\n\n### analysis', '# Problem & objective\n\n## Problem & Objective').
_strip_leading_md_header removes a leading author header so exactly one (the
tool's canonical) heading remains.
"""
from __future__ import annotations

from a3dasm._src.nodes.tools.routing import _strip_leading_md_header


def test_strips_a_leading_header():
    assert _strip_leading_md_header("## Hypotheses\n\nTwo claims.") == "Two claims."
    assert _strip_leading_md_header("### analysis\n\nDerived from ledger.") == "Derived from ledger."
    assert _strip_leading_md_header("# Problem & objective\n\n**Task:** min f.") == "**Task:** min f."


def test_passthrough_when_no_leading_header():
    assert _strip_leading_md_header("Two competing claims.") == "Two competing claims."
    assert _strip_leading_md_header("**Task:** minimise f over the box.") == "**Task:** minimise f over the box."


def test_preserves_internal_subheaders():
    # only the LEADING header is dropped; structure inside the body survives.
    out = _strip_leading_md_header("## Hypotheses\n\n### H1\nclaim one\n### H2\nclaim two")
    assert out == "### H1\nclaim one\n### H2\nclaim two"
    assert out.count("###") == 2


def test_canonical_heading_not_doubled_after_prepend():
    # Simulate what the tool does: canonical heading + stripped author content.
    canonical = "## Hypotheses"
    author = "## Hypotheses\n\nThe registered claims."
    rendered = canonical + "\n\n" + _strip_leading_md_header(author)
    assert rendered.count("## Hypotheses") == 1
    assert rendered == "## Hypotheses\n\nThe registered claims."


def test_handles_empty_and_header_only():
    assert _strip_leading_md_header("") == ""
    assert _strip_leading_md_header("## Only a header") == ""
