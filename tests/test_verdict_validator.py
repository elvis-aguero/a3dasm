"""Pure-logic tests for the #9 live verdict validator (no model, no node)."""
from __future__ import annotations

from a3dasm._src.verdict_validator import (
    CLOSING_STATUSES,
    build_judge_prompt,
    parse_judge_reply,
)


# ── parse_judge_reply ────────────────────────────────────────────────────────

def test_parse_ok_is_not_flagged_and_has_no_critique():
    assert parse_judge_reply("SUBSTANCE: OK") == (False, "")


def test_parse_ok_case_insensitive_and_with_trailing_noise():
    flagged, critique = parse_judge_reply("substance:ok\n(all three checks pass)")
    assert flagged is False and critique == ""


def test_parse_flag_extracts_critique():
    reply = "SUBSTANCE: FLAG\nCRITIQUE: §4 goalpost-moved — verdict cites a post-hoc metric."
    flagged, critique = parse_judge_reply(reply)
    assert flagged is True
    assert "§4" in critique and "post-hoc" in critique


def test_parse_flag_without_critique_still_flags():
    flagged, critique = parse_judge_reply("SUBSTANCE: FLAG")
    assert flagged is True and critique  # non-empty fallback reason


def test_parse_unparseable_does_not_fabricate_a_flag():
    # An advisory check must never invent a flag from noise; keep the raw text.
    flagged, critique = parse_judge_reply("the model rambled without the tag")
    assert flagged is False
    assert critique == "the model rambled without the tag"


def test_parse_empty_reply():
    assert parse_judge_reply("") == (False, "")
    assert parse_judge_reply("   ") == (False, "")


# ── build_judge_prompt ───────────────────────────────────────────────────────

def test_prompt_injects_charter_and_all_inputs():
    prompt = build_judge_prompt(
        statement="local search beats BO on deceptive landscapes",
        prediction="best_f(local) < best_f(BO) on the 3D oracle",
        criterion="best_f(local) >= best_f(BO)",
        status="FALSIFIED",
        comment="BO won",
        evidence={"delegation": "D004", "numbers": {"best_f": -0.70}},
        delegation_report="BO reached -0.70; local search reached -0.55.",
    )
    # charter is injected verbatim (same standard as the gate critic)
    assert "SCIENTIFIC-METHOD CHARTER" in prompt
    assert "§2" in prompt and "§3" in prompt and "§4" in prompt
    # the verdict under review and its evidence are present
    assert "FALSIFIED" in prompt
    assert "D004" in prompt
    assert "best_f(local) < best_f(BO)" in prompt  # the registered prediction
    assert "BO reached -0.70" in prompt  # the cited delegation result
    # the required reply contract is stated
    assert "SUBSTANCE: OK" in prompt and "SUBSTANCE: FLAG" in prompt


def test_prompt_handles_missing_delegation_report():
    prompt = build_judge_prompt(
        statement="s", prediction="p", criterion="c", status="SUPPORTED",
        comment="x", evidence=None, delegation_report=None,
    )
    assert "(no delegation report available)" in prompt


def test_closing_statuses_excludes_open():
    assert "OPEN" not in CLOSING_STATUSES
    assert CLOSING_STATUSES == {"SUPPORTED", "FALSIFIED", "INCONCLUSIVE"}
