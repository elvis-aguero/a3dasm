"""The report-retry prompt is built from the agent's own report_sections.

Regression for run 20260708T021335 (D005/D007/D011): the static
IMPLEMENTER_REPORT_RETRY_PROMPT hardcoded 4 subsections and commanded "EXACTLY
this structure / do not skip any subsection", which would make a compliant
model DROP the implementer's 5th section, ### Retrospective — and was wrong for
the critic/lit-reviewer whose report_sections differ. The prompt must track
whatever sections the parser actually validates.
"""
from __future__ import annotations

from a3dasm._src.agent_prompts import build_report_retry_prompt
from a3dasm._src.agents import (
    AdversarialCritiqueAgent,
    F3dasmImplementerAgent,
)


def test_prompt_includes_every_implementer_section_incl_retrospective():
    secs = F3dasmImplementerAgent.report_sections
    p = build_report_retry_prompt(secs)
    for s in secs:
        assert s in p, f"retry prompt dropped {s!r}"
    assert "### Retrospective" in p  # the section the old static prompt omitted


def test_prompt_tracks_critic_sections_not_implementer():
    secs = AdversarialCritiqueAgent.report_sections
    if not secs:
        return
    p = build_report_retry_prompt(secs)
    for s in secs:
        assert s in p
    # must not force implementer-only sections onto the critic
    if "### Files touched" not in secs:
        assert "### Files touched" not in p


def test_default_prompt_still_has_report_and_four_headings():
    p = build_report_retry_prompt()
    assert "## Report" in p
    for s in ("### Actions taken", "### Files touched", "### Conclusions",
              "### Numbers"):
        assert s in p
