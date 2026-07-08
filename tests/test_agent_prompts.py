"""Tests for ``a3dasm._src.agent_prompts``.

Validates that the four prompt constants shipped with the agentic-f3dasm
v2 runtime satisfy structural, content, and integration requirements.
The runtime behaviour itself is covered by ``test_agent_runtime.py``.

Notes
-----
These are *sanity tests* — they enforce the contract between the prompt
module and the runtime, not the scientific quality of the prose.
"""
#                                                                       Modules
# =============================================================================
from __future__ import annotations

import re

import pytest

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Elvis Aguero (elvis_alexander_aguero_vera@brown.edu)"
__credits__ = ["Elvis Aguero"]
__status__ = "Experimental"
# =============================================================================


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tag_count(text: str, tag: str) -> tuple[int, int]:
    """Return (opening_count, closing_count) for an XML tag pair.

    Parameters
    ----------
    text : str
        The string to search.
    tag : str
        The bare tag name, e.g. ``"role"``.

    Returns
    -------
    tuple[int, int]
        ``(n_opening, n_closing)`` occurrence counts.
    """
    opening = len(re.findall(rf"<{tag}>", text))
    closing = len(re.findall(rf"</{tag}>", text))
    return opening, closing


def _assert_tag_once(text: str, tag: str) -> None:
    """Assert an XML tag pair appears exactly once in *text*.

    An "exact pair" means exactly one ``</tag>`` closing tag exists.
    Opening tags may appear more than once when the tag name is
    referenced in prose (e.g. ``<output_format>.``), so only the
    closing count is used as the uniqueness signal.

    Parameters
    ----------
    text : str
        The string to search.
    tag : str
        The bare tag name.
    """
    _, closing = _tag_count(text, tag)
    assert closing == 1, (
        f"Expected exactly 1 closing </{tag}>; found {closing}"
    )
    # At least one opening must exist to form a valid pair.
    opening, _ = _tag_count(text, tag)
    assert opening >= 1, (
        f"Expected at least 1 opening <{tag}>; found {opening}"
    )


# ---------------------------------------------------------------------------
# Test 1 — constants exist and are non-empty strings
# ---------------------------------------------------------------------------

def test_constants_exist_and_are_non_empty_strings():
    """All four constants are non-empty strings of minimum required length.

    Notes
    -----
    ``IMPLEMENTER_RESET_PROMPT_TEMPLATE`` has a lower threshold (300 chars)
    because it is a short wrapper template, not a full system prompt.
    """
    from a3dasm._src.agent_prompts import (
        CHECKPOINT_STRATEGIZER_PROMPT,
        IMPLEMENTER_RESET_PROMPT_TEMPLATE,
        IMPLEMENTER_SYSTEM_PROMPT,
        STRATEGIZER_SYSTEM_PROMPT,
    )

    for name, constant in (
        ("STRATEGIZER_SYSTEM_PROMPT", STRATEGIZER_SYSTEM_PROMPT),
        ("IMPLEMENTER_SYSTEM_PROMPT", IMPLEMENTER_SYSTEM_PROMPT),
        ("CHECKPOINT_STRATEGIZER_PROMPT", CHECKPOINT_STRATEGIZER_PROMPT),
    ):
        assert isinstance(constant, str), f"{name} is not a str"
        assert len(constant) >= 500, (
            f"{name} is too short ({len(constant)} chars); expected >= 500"
        )

    assert isinstance(IMPLEMENTER_RESET_PROMPT_TEMPLATE, str), (
        "IMPLEMENTER_RESET_PROMPT_TEMPLATE is not a str"
    )
    assert len(IMPLEMENTER_RESET_PROMPT_TEMPLATE) >= 300, (
        "IMPLEMENTER_RESET_PROMPT_TEMPLATE too short; expected >= 300"
    )


# ---------------------------------------------------------------------------
# Test 2 — STRATEGIZER_SYSTEM_PROMPT XML structure
# ---------------------------------------------------------------------------

def test_strategizer_xml_sections_appear_exactly_once():
    """Required XML-tagged sections each appear exactly once.

    Notes
    -----
    The runtime relies on these sections to give structure to the prompt;
    duplicate tags would confuse any downstream XML parser.
    """
    from a3dasm._src.agent_prompts import (
        STRATEGIZER_SYSTEM_PROMPT,
    )

    # tool_usage / output_format / examples were removed: per-tool usage rules
    # and worked examples now live ON the tool definitions (docstrings +
    # @tool_examples) and are injected via the generated <tools> catalog, so
    # they're DRY and shared, not hand-maintained in the prompt.
    required_tags = [
        "role",
        "deliverables",
        "operating_principles",
        "failure_modes_to_avoid",
    ]
    for tag in required_tags:
        _assert_tag_once(STRATEGIZER_SYSTEM_PROMPT, tag)


# ---------------------------------------------------------------------------
# Test 3 — STRATEGIZER named failure modes
# ---------------------------------------------------------------------------

def test_strategizer_names_all_failure_modes():
    """All six cognitive-bias failure modes are named in the prompt.

    Notes
    -----
    The names are matched case-insensitively to allow stylistic variation
    (e.g. ``ANCHORING BIAS`` vs ``anchoring``).
    """
    from a3dasm._src.agent_prompts import (
        STRATEGIZER_SYSTEM_PROMPT,
    )

    lower = STRATEGIZER_SYSTEM_PROMPT.lower()
    required_terms = [
        "anchoring",
        "confirmation",
        "availability",
        "role drift",
        "sycophancy",
        "premature convergence",
    ]
    for term in required_terms:
        assert term in lower, (
            f"STRATEGIZER_SYSTEM_PROMPT does not mention '{term}'"
        )


# ---------------------------------------------------------------------------
# Test 4 — STRATEGIZER mentions all five tools
# ---------------------------------------------------------------------------

def test_strategizer_mentions_all_five_tools():
    """The Strategizer prompt names every tool the agent may call.

    Notes
    -----
    The tool names are checked as literal substrings; the prompt must
    spell them exactly so the model sees the correct function names.
    """
    from a3dasm._src.agent_prompts import (
        STRATEGIZER_SYSTEM_PROMPT,
    )

    required_tools = ["Read", "WriteNote", "FollowUp", "Delegate", "Done"]
    for tool in required_tools:
        assert tool in STRATEGIZER_SYSTEM_PROMPT, (
            f"STRATEGIZER_SYSTEM_PROMPT does not mention tool '{tool}'"
        )


# ---------------------------------------------------------------------------
# Test 5 — STRATEGIZER briefing-clarification ritual
# ---------------------------------------------------------------------------

def test_strategizer_briefing_clarification_ritual():
    """Prompt encodes the FollowUp-before-hypothesis briefing ritual."""
    from a3dasm._src.agent_prompts import (
        STRATEGIZER_SYSTEM_PROMPT,
    )

    lower = STRATEGIZER_SYSTEM_PROMPT.lower()

    # FollowUp tool must appear (replaced Ask).
    assert "followup" in lower, (
        "STRATEGIZER_SYSTEM_PROMPT does not mention 'FollowUp' tool"
    )

    # BRIEFING-CLARIFICATION RITUAL section must exist.
    assert "briefing" in lower, (
        "STRATEGIZER_SYSTEM_PROMPT does not contain 'briefing' ritual"
    )

    # 'before' and 'briefing' must appear within 200 chars of each other.
    idx = lower.find("briefing")
    window = lower[max(0, idx - 200): idx + 200]
    assert "before" in window, (
        "STRATEGIZER_SYSTEM_PROMPT does not link 'briefing' and 'before' "
        "within a 200-char window — briefing ritual may be missing"
    )


# ---------------------------------------------------------------------------
# Test 6 — IMPLEMENTER_SYSTEM_PROMPT XML structure
# ---------------------------------------------------------------------------

def test_implementer_xml_sections_appear_exactly_once():
    """All required XML-tagged sections appear exactly once.

    Notes
    -----
    Includes the f3dasm-specific ``<f3dasm_api>`` tag that teaches the
    Implementer how to use the framework.
    """
    from a3dasm._src.agent_prompts import (
        IMPLEMENTER_SYSTEM_PROMPT,
    )

    required_tags = [
        "role",
        "deliverables",
        "f3dasm_api",
        "operating_principles",
        "failure_modes_to_avoid",
        "tool_usage",
        "output_format",
        "examples",
    ]
    for tag in required_tags:
        _assert_tag_once(IMPLEMENTER_SYSTEM_PROMPT, tag)


# ---------------------------------------------------------------------------
# Test 7 — IMPLEMENTER f3dasm primer references
# ---------------------------------------------------------------------------

def test_implementer_f3dasm_primer_references_key_classes():
    """The f3dasm primer section names all critical public API.

    Notes
    -----
    Checked as literal substrings (case-sensitive) because the model must
    use the exact public names when generating code. Samplers are reached
    through the public ``create_sampler`` factory with string keys
    (e.g. ``"latin_sampler"``), not the private sampler classes.
    """
    from a3dasm._src.agent_prompts import (
        IMPLEMENTER_SYSTEM_PROMPT,
    )

    required_terms = [
        "Domain",
        "ExperimentData",
        "create_sampler",
        "latin_sampler",
        "DataGenerator",
        "LookupDataGenerator",
    ]
    for term in required_terms:
        assert term in IMPLEMENTER_SYSTEM_PROMPT, (
            f"IMPLEMENTER_SYSTEM_PROMPT primer missing '{term}'"
        )


# ---------------------------------------------------------------------------
# Test 8 — IMPLEMENTER report format headings
# ---------------------------------------------------------------------------

def test_implementer_report_format_headings():
    """The output-format section contains the exact required headings.

    Notes
    -----
    The runtime greps for ``## Report`` to extract the Implementer's
    response; the other headings are checked here as a structural contract.
    """
    from a3dasm._src.agent_prompts import (
        IMPLEMENTER_SYSTEM_PROMPT,
    )

    required_headings = [
        "## Report",
        "### Actions taken",
        "### Files touched",
        "### Conclusions",
        "### Numbers",
    ]
    for heading in required_headings:
        assert heading in IMPLEMENTER_SYSTEM_PROMPT, (
            f"IMPLEMENTER_SYSTEM_PROMPT missing heading '{heading}'"
        )


# ---------------------------------------------------------------------------
# Test 9 — IMPLEMENTER refuses hypothesis-verification tasks
# ---------------------------------------------------------------------------

def test_implementer_scope_boundary_is_execute_not_adjudicate():
    """O14: the implementer's scope boundary is stated as a division of labour,
    NOT as blind keyword refusal.

    The old rule told the implementer to REFUSE any task whose wording said
    "verify"/"confirm a hypothesis" — which stranded the Strategizer with no
    evidence on legitimately-phrased measurement tasks. The boundary now says:
    the implementer EXECUTES and MEASURES; the Strategizer draws the verdict.
    (The previous keyword test passed only incidentally — "hypothesis" survived
    elsewhere in the prompt — so it never actually guarded the behaviour.)
    """
    from a3dasm._src.agent_prompts import (
        IMPLEMENTER_SYSTEM_PROMPT,
    )

    p = IMPLEMENTER_SYSTEM_PROMPT
    assert "SCOPE BOUNDARY" in p
    assert "do not adjudicate" in p
    # the old keyword-refusal anti-pattern must not return
    assert "outside Implementer scope" not in p


# ---------------------------------------------------------------------------
# Test 10 — No supercompressible leakage
# ---------------------------------------------------------------------------

def test_no_supercompressible_leakage():
    """None of the four prompts contain problem-specific vocabulary.

    Notes
    -----
    The prompts must be problem-agnostic.  Any leakage of terms from the
    supercompressible metamaterial study would bias the agent toward that
    domain.
    """
    from a3dasm._src.agent_prompts import (
        CHECKPOINT_STRATEGIZER_PROMPT,
        IMPLEMENTER_RESET_PROMPT_TEMPLATE,
        IMPLEMENTER_SYSTEM_PROMPT,
        STRATEGIZER_SYSTEM_PROMPT,
    )

    forbidden_terms = [
        "coilable",
        "sigma_crit",
        "ratio_d",
        "ratio_pitch",
        "ratio_top_diameter",
        "bessa",
        "supercompressible",
    ]
    prompts = {
        "STRATEGIZER_SYSTEM_PROMPT": STRATEGIZER_SYSTEM_PROMPT,
        "IMPLEMENTER_SYSTEM_PROMPT": IMPLEMENTER_SYSTEM_PROMPT,
        "CHECKPOINT_STRATEGIZER_PROMPT": CHECKPOINT_STRATEGIZER_PROMPT,
        "IMPLEMENTER_RESET_PROMPT_TEMPLATE": (
            IMPLEMENTER_RESET_PROMPT_TEMPLATE
        ),
    }
    for const_name, text in prompts.items():
        lower = text.lower()
        for term in forbidden_terms:
            assert term not in lower, (
                f"{const_name} leaks domain-specific term '{term}'"
            )


# ---------------------------------------------------------------------------
# Test 11 — CHECKPOINT_STRATEGIZER_PROMPT contract
# ---------------------------------------------------------------------------

def test_checkpoint_prompt_section_headings():
    """Checkpoint prompt contains the exact headings the runtime expects.

    Notes
    -----
    These headings are case-sensitive because the Strategizer model is
    instructed to reproduce them verbatim in its checkpoint report.
    """
    from a3dasm._src.agent_prompts import (
        CHECKPOINT_STRATEGIZER_PROMPT,
    )

    required_elements = [
        "## Checkpoint",
        "What we have learned",
        "What we have ruled out",
        "Open questions",
        "Recommended next direction",
    ]
    for element in required_elements:
        assert element in CHECKPOINT_STRATEGIZER_PROMPT, (
            f"CHECKPOINT_STRATEGIZER_PROMPT missing element '{element}'"
        )


# ---------------------------------------------------------------------------
# Test 12 — CHECKPOINT forbids new hypotheses
# ---------------------------------------------------------------------------

def test_checkpoint_prompt_forbids_new_hypotheses():
    """Checkpoint prompt explicitly forbids generating new hypotheses.

    Notes
    -----
    Either of two acceptable phrasings is checked case-insensitively.
    """
    from a3dasm._src.agent_prompts import (
        CHECKPOINT_STRATEGIZER_PROMPT,
    )

    lower = CHECKPOINT_STRATEGIZER_PROMPT.lower()
    acceptable = [
        "do not generate new hypotheses",
        "no new hypotheses",
    ]
    assert any(phrase in lower for phrase in acceptable), (
        "CHECKPOINT_STRATEGIZER_PROMPT does not forbid new hypotheses; "
        f"expected one of: {acceptable}"
    )


# ---------------------------------------------------------------------------
# Test 13 — IMPLEMENTER_RESET_PROMPT_TEMPLATE placeholder
# ---------------------------------------------------------------------------

def test_reset_template_has_exactly_one_placeholder():
    """Template contains ``{checkpoint_summary}`` exactly once and no other
    format placeholders.

    Notes
    -----
    A ``KeyError`` from ``.format()`` would indicate a stray placeholder
    that the runtime does not supply.
    """
    from a3dasm._src.agent_prompts import (
        IMPLEMENTER_RESET_PROMPT_TEMPLATE,
    )

    assert IMPLEMENTER_RESET_PROMPT_TEMPLATE.count(
        "{checkpoint_summary}"
    ) == 1, (
        "IMPLEMENTER_RESET_PROMPT_TEMPLATE must contain "
        "'{checkpoint_summary}' exactly once"
    )

    # Must not raise KeyError from any unknown placeholder.
    try:
        IMPLEMENTER_RESET_PROMPT_TEMPLATE.format(
            checkpoint_summary="STUB"
        )
    except KeyError as exc:
        raise AssertionError(
            f"IMPLEMENTER_RESET_PROMPT_TEMPLATE has unknown placeholder: "
            f"{exc}"
        ) from exc


# ---------------------------------------------------------------------------
# Test 14 — Reset template structure after formatting
# ---------------------------------------------------------------------------

def test_reset_template_structure_after_formatting():
    """Formatted template contains the stub text plus required anchors.

    Notes
    -----
    ``Strategizer`` and ``Task`` are checked as loose anchors that orient
    the fresh Implementer session to its role and workflow.
    """
    from a3dasm._src.agent_prompts import (
        IMPLEMENTER_RESET_PROMPT_TEMPLATE,
    )

    result = IMPLEMENTER_RESET_PROMPT_TEMPLATE.format(
        checkpoint_summary="STUB_SUMMARY_TEXT"
    )
    assert "STUB_SUMMARY_TEXT" in result, (
        "Formatted reset template does not contain the substituted summary"
    )
    assert "Strategizer" in result, (
        "Formatted reset template does not reference 'Strategizer'"
    )
    assert "Task" in result, (
        "Formatted reset template does not reference 'Task'"
    )


# ---------------------------------------------------------------------------
# NEW Test 16 — Piece A: hypothesis_log tag appears exactly once
# ---------------------------------------------------------------------------

def test_strategizer_hypothesis_log_tag_appears_once():
    """STRATEGIZER_SYSTEM_PROMPT has exactly one <hypothesis_ledger> pair."""
    from a3dasm._src.agent_prompts import (
        STRATEGIZER_SYSTEM_PROMPT,
    )

    _assert_tag_once(STRATEGIZER_SYSTEM_PROMPT, "hypothesis_ledger")


# ---------------------------------------------------------------------------
# NEW Test 17 — Piece A: hypothesis_log section content
# ---------------------------------------------------------------------------

def test_strategizer_hypothesis_log_content():
    """hypothesis_ledger section mentions all required concepts."""
    from a3dasm._src.agent_prompts import (
        STRATEGIZER_SYSTEM_PROMPT,
    )

    lower = STRATEGIZER_SYSTEM_PROMPT.lower()
    required_terms = [
        "hypotheses.json",
        "hypothesispropose",
        "hypothesisupdate",
        "hypothesislist",
        "hypothesis_ids",
        "falsified",
    ]
    for term in required_terms:
        assert term in lower, (
            f"STRATEGIZER_SYSTEM_PROMPT hypothesis_ledger missing '{term}'"
        )


# ---------------------------------------------------------------------------
# NEW Test 18 — Piece B: on_error tag appears exactly once
# ---------------------------------------------------------------------------

def test_strategizer_on_error_tag_appears_once():
    """STRATEGIZER_SYSTEM_PROMPT has exactly one <on_error> pair."""
    from a3dasm._src.agent_prompts import (
        STRATEGIZER_SYSTEM_PROMPT,
    )

    _assert_tag_once(STRATEGIZER_SYSTEM_PROMPT, "on_error")


# ---------------------------------------------------------------------------
# NEW Test 19 — Piece B: on_error section content
# ---------------------------------------------------------------------------

def test_strategizer_on_error_content():
    """on_error covers the async Errored: path and forbids verbatim re-delegation."""
    from a3dasm._src.agent_prompts import (
        STRATEGIZER_SYSTEM_PROMPT,
    )

    lower = STRATEGIZER_SYSTEM_PROMPT.lower()
    # Must describe the Errored: path from GetStatus() (async, not REFLECT:)
    assert "errored:" in lower, (
        "STRATEGIZER_SYSTEM_PROMPT on_error does not mention 'Errored:' path"
    )
    # Must warn against verbatim re-delegation without addressing root cause
    assert "verbatim" in lower, (
        "STRATEGIZER_SYSTEM_PROMPT on_error missing 'verbatim' re-delegation guard"
    )


# ---------------------------------------------------------------------------
# NEW Test 20 — Piece A: CHECKPOINT prompt contains Comment log section
# ---------------------------------------------------------------------------

def test_checkpoint_prompt_contains_ledger_digest():
    """CHECKPOINT_STRATEGIZER_PROMPT contains the ### Hypothesis ledger digest section."""
    from a3dasm._src.agent_prompts import (
        CHECKPOINT_STRATEGIZER_PROMPT,
    )

    assert "### Hypothesis ledger digest" in CHECKPOINT_STRATEGIZER_PROMPT, (
        "CHECKPOINT_STRATEGIZER_PROMPT missing '### Hypothesis ledger digest' section"
    )


# ---------------------------------------------------------------------------
# NEW Test 21 — Piece C: reasoning_protocol tag appears exactly once
# ---------------------------------------------------------------------------

def test_implementer_reasoning_protocol_tag_appears_once():
    """IMPLEMENTER_SYSTEM_PROMPT has exactly one <reasoning_protocol> pair."""
    from a3dasm._src.agent_prompts import (
        IMPLEMENTER_SYSTEM_PROMPT,
    )

    _assert_tag_once(IMPLEMENTER_SYSTEM_PROMPT, "reasoning_protocol")


# ---------------------------------------------------------------------------
# NEW Test 22 — Piece C: reasoning_protocol section content
# ---------------------------------------------------------------------------

def test_implementer_reasoning_protocol_content():
    """reasoning_protocol mentions Stages 1-3 and their headings."""
    from a3dasm._src.agent_prompts import (
        IMPLEMENTER_SYSTEM_PROMPT,
    )

    required_terms = [
        "Stage 1",
        "Stage 2",
        "Stage 3",
        "Task restatement",
        "Workspace inventory",
        "Execution plan",
    ]
    for term in required_terms:
        assert term in IMPLEMENTER_SYSTEM_PROMPT, (
            f"IMPLEMENTER_SYSTEM_PROMPT reasoning_protocol missing "
            f"'{term}'"
        )


# ---------------------------------------------------------------------------
# NEW Test 23 — Piece A/B/C: cleanliness invariant (extended)
# ---------------------------------------------------------------------------

def test_no_domain_specific_leakage_extended():
    """No prompt contains 'coilable', 'sigma_crit', or 'Bessa'."""
    from a3dasm._src.agent_prompts import (
        CHECKPOINT_STRATEGIZER_PROMPT,
        IMPLEMENTER_RESET_PROMPT_TEMPLATE,
        IMPLEMENTER_SYSTEM_PROMPT,
        STRATEGIZER_SYSTEM_PROMPT,
    )

    forbidden_terms = ["coilable", "sigma_crit", "bessa"]
    prompts = {
        "STRATEGIZER_SYSTEM_PROMPT": STRATEGIZER_SYSTEM_PROMPT,
        "IMPLEMENTER_SYSTEM_PROMPT": IMPLEMENTER_SYSTEM_PROMPT,
        "CHECKPOINT_STRATEGIZER_PROMPT": CHECKPOINT_STRATEGIZER_PROMPT,
        "IMPLEMENTER_RESET_PROMPT_TEMPLATE": (
            IMPLEMENTER_RESET_PROMPT_TEMPLATE
        ),
    }
    for const_name, text in prompts.items():
        lower = text.lower()
        for term in forbidden_terms:
            assert term not in lower, (
                f"{const_name} leaks term '{term}'"
            )


# ---------------------------------------------------------------------------
# NEW Test 24 — RUN_PATHS_PREAMBLE_TEMPLATE existence and type
# ---------------------------------------------------------------------------

def test_run_paths_preamble_template_exists_and_is_string():
    """RUN_PATHS_PREAMBLE_TEMPLATE is a non-empty string."""
    from a3dasm._src.agent_prompts import (
        RUN_PATHS_PREAMBLE_TEMPLATE,
    )

    assert isinstance(RUN_PATHS_PREAMBLE_TEMPLATE, str), (
        "RUN_PATHS_PREAMBLE_TEMPLATE is not a str"
    )
    assert len(RUN_PATHS_PREAMBLE_TEMPLATE) > 0, (
        "RUN_PATHS_PREAMBLE_TEMPLATE is empty"
    )


# ---------------------------------------------------------------------------
# NEW Test 25 — RUN_PATHS_PREAMBLE_TEMPLATE placeholders and content
# ---------------------------------------------------------------------------

def test_run_paths_preamble_template_placeholders():
    """Template substitutes study_dir, run_dir, debug_dir, notes_dir; missing kwarg raises.

    Notes
    -----
    The template must mention ``study_dir`` and
    ``strategizer_notes_dir`` as literal labels in its body so that
    the Strategizer understands the semantics of each path.
    """
    from a3dasm._src.agent_prompts import (
        RUN_PATHS_PREAMBLE_TEMPLATE,
    )

    result = RUN_PATHS_PREAMBLE_TEMPLATE.format(
        study_dir="/a/study",
        run_dir="/a/study/runs/ts",
        debug_dir="/a/study/runs/ts/debug",
        notes_dir="/a/notes",
        experiment_data_dir="/a/study/runs/ts/experiment_data",
        resources="",
        knowledge="",
    )
    assert "/a/study" in result, (
        "study_dir substitution not found in result"
    )
    assert "/a/notes" in result, (
        "notes_dir substitution not found in result"
    )
    assert "study_dir" in RUN_PATHS_PREAMBLE_TEMPLATE, (
        "RUN_PATHS_PREAMBLE_TEMPLATE does not contain literal "
        "'study_dir'"
    )
    assert "strategizer_notes_dir" in RUN_PATHS_PREAMBLE_TEMPLATE, (
        "RUN_PATHS_PREAMBLE_TEMPLATE does not contain literal "
        "'strategizer_notes_dir'"
    )

    import pytest
    with pytest.raises(KeyError):
        RUN_PATHS_PREAMBLE_TEMPLATE.format(study_dir="/a/study")


# ---------------------------------------------------------------------------
# NEW Test 26 — WORKSPACE_PREAMBLE_TEMPLATE existence and type
# ---------------------------------------------------------------------------

def test_workspace_preamble_template_exists_and_is_string():
    """WORKSPACE_PREAMBLE_TEMPLATE is a non-empty string."""
    from a3dasm._src.agent_prompts import (
        WORKSPACE_PREAMBLE_TEMPLATE,
    )

    assert isinstance(WORKSPACE_PREAMBLE_TEMPLATE, str), (
        "WORKSPACE_PREAMBLE_TEMPLATE is not a str"
    )
    assert len(WORKSPACE_PREAMBLE_TEMPLATE) > 0, (
        "WORKSPACE_PREAMBLE_TEMPLATE is empty"
    )


# ---------------------------------------------------------------------------
# NEW Test 27 — WORKSPACE_PREAMBLE_TEMPLATE placeholder and /tmp warning
# ---------------------------------------------------------------------------

def test_workspace_preamble_template_placeholder_and_no_tmp():
    """Template substitutes workspace_dir and study_dir; warns against /tmp.

    Notes
    -----
    ``/tmp`` (case-insensitive) must appear as a forbidden example so
    the Implementer understands that scratch files outside the workspace
    will be lost.
    """
    from a3dasm._src.agent_prompts import (
        WORKSPACE_PREAMBLE_TEMPLATE,
    )

    result = WORKSPACE_PREAMBLE_TEMPLATE.format(
        workspace_dir="/a/workspace",
        study_dir="/a/study",
        resources="",
        knowledge="",
    )
    assert "/a/workspace" in result, (
        "workspace_dir substitution not found in result"
    )
    assert "workspace_dir" in WORKSPACE_PREAMBLE_TEMPLATE, (
        "WORKSPACE_PREAMBLE_TEMPLATE does not contain literal "
        "'workspace_dir'"
    )
    assert "/tmp" in WORKSPACE_PREAMBLE_TEMPLATE.lower(), (
        "WORKSPACE_PREAMBLE_TEMPLATE does not warn against /tmp"
    )

    import pytest
    with pytest.raises(KeyError):
        WORKSPACE_PREAMBLE_TEMPLATE.format()


# ---------------------------------------------------------------------------
# NEW Test 28 — IMPLEMENTER_REPORT_RETRY_PROMPT existence and content
# ---------------------------------------------------------------------------

def test_implementer_report_retry_prompt_exists_and_is_string():
    """IMPLEMENTER_REPORT_RETRY_PROMPT is a non-empty string."""
    from a3dasm._src.agent_prompts import (
        IMPLEMENTER_REPORT_RETRY_PROMPT,
    )

    assert isinstance(IMPLEMENTER_REPORT_RETRY_PROMPT, str), (
        "IMPLEMENTER_REPORT_RETRY_PROMPT is not a str"
    )
    assert len(IMPLEMENTER_REPORT_RETRY_PROMPT) > 0, (
        "IMPLEMENTER_REPORT_RETRY_PROMPT is empty"
    )


# ---------------------------------------------------------------------------
# NEW Test 29 — IMPLEMENTER_REPORT_RETRY_PROMPT required subsections
# ---------------------------------------------------------------------------

def test_implementer_report_retry_prompt_required_headings():
    """Retry prompt contains '## Report' and all four subsection headings.

    Notes
    -----
    The runtime greps for ``## Report``; the four ``###`` subsections
    are structural requirements checked by ``_classify_failed_*``.
    """
    from a3dasm._src.agent_prompts import (
        IMPLEMENTER_REPORT_RETRY_PROMPT,
    )

    assert "## Report" in IMPLEMENTER_REPORT_RETRY_PROMPT, (
        "IMPLEMENTER_REPORT_RETRY_PROMPT missing '## Report'"
    )
    for subsection in (
        "### Actions taken",
        "### Files touched",
        "### Conclusions",
        "### Numbers",
    ):
        assert subsection in IMPLEMENTER_REPORT_RETRY_PROMPT, (
            f"IMPLEMENTER_REPORT_RETRY_PROMPT missing '{subsection}'"
        )


# ---------------------------------------------------------------------------
# NEW Test 30 — REFLECT_DIAGNOSIS_SHORT existence and keyword
# ---------------------------------------------------------------------------

def test_reflect_diagnosis_short_exists_and_keyword():
    """REFLECT_DIAGNOSIS_SHORT is a non-empty string mentioning 'unusually
    short'."""
    from a3dasm._src.agent_prompts import (
        REFLECT_DIAGNOSIS_SHORT,
    )

    assert isinstance(REFLECT_DIAGNOSIS_SHORT, str), (
        "REFLECT_DIAGNOSIS_SHORT is not a str"
    )
    assert len(REFLECT_DIAGNOSIS_SHORT) > 0, (
        "REFLECT_DIAGNOSIS_SHORT is empty"
    )
    assert "unusually short" in REFLECT_DIAGNOSIS_SHORT.lower(), (
        "REFLECT_DIAGNOSIS_SHORT does not mention 'unusually short'"
    )


# ---------------------------------------------------------------------------
# NEW Test 31 — REFLECT_DIAGNOSIS_CAPABILITY_LIMIT existence and keyword
# ---------------------------------------------------------------------------

def test_reflect_diagnosis_capability_limit_exists_and_keyword():
    """REFLECT_DIAGNOSIS_CAPABILITY_LIMIT is a non-empty string mentioning
    'capability'."""
    from a3dasm._src.agent_prompts import (
        REFLECT_DIAGNOSIS_CAPABILITY_LIMIT,
    )

    assert isinstance(REFLECT_DIAGNOSIS_CAPABILITY_LIMIT, str), (
        "REFLECT_DIAGNOSIS_CAPABILITY_LIMIT is not a str"
    )
    assert len(REFLECT_DIAGNOSIS_CAPABILITY_LIMIT) > 0, (
        "REFLECT_DIAGNOSIS_CAPABILITY_LIMIT is empty"
    )
    assert "capability" in REFLECT_DIAGNOSIS_CAPABILITY_LIMIT.lower(), (
        "REFLECT_DIAGNOSIS_CAPABILITY_LIMIT does not mention 'capability'"
    )


# ---------------------------------------------------------------------------
# NEW Test 32 — REFLECT_DIAGNOSIS_MISSING_SUBSECTIONS_TEMPLATE
# ---------------------------------------------------------------------------

def test_reflect_diagnosis_missing_subsections_template():
    """Template contains the literal placeholder before formatting and
    substitutes correctly when formatted.

    Notes
    -----
    The literal substring ``{missing_subsections}`` must exist in the
    raw template text so the contract is visible to code readers.
    """
    from a3dasm._src.agent_prompts import (
        REFLECT_DIAGNOSIS_MISSING_SUBSECTIONS_TEMPLATE,
    )

    assert isinstance(
        REFLECT_DIAGNOSIS_MISSING_SUBSECTIONS_TEMPLATE, str
    ), "REFLECT_DIAGNOSIS_MISSING_SUBSECTIONS_TEMPLATE is not a str"
    assert len(REFLECT_DIAGNOSIS_MISSING_SUBSECTIONS_TEMPLATE) > 0, (
        "REFLECT_DIAGNOSIS_MISSING_SUBSECTIONS_TEMPLATE is empty"
    )
    assert (
        "{missing_subsections}"
        in REFLECT_DIAGNOSIS_MISSING_SUBSECTIONS_TEMPLATE
    ), (
        "REFLECT_DIAGNOSIS_MISSING_SUBSECTIONS_TEMPLATE does not contain "
        "literal '{missing_subsections}' placeholder"
    )

    result = REFLECT_DIAGNOSIS_MISSING_SUBSECTIONS_TEMPLATE.format(
        missing_subsections="'Numbers'"
    )
    assert "'Numbers'" in result, (
        "Formatted result does not contain substituted value"
    )

    import pytest
    with pytest.raises(KeyError):
        REFLECT_DIAGNOSIS_MISSING_SUBSECTIONS_TEMPLATE.format()


# ---------------------------------------------------------------------------
# NEW Test 36 — IMPLEMENTER_SYSTEM_PROMPT_OLLAMA existence and content
# ---------------------------------------------------------------------------

def test_ollama_implementer_prompt_exists():
    from a3dasm._src.agent_prompts import (
        IMPLEMENTER_SYSTEM_PROMPT_OLLAMA,
    )
    assert "bash" in IMPLEMENTER_SYSTEM_PROMPT_OLLAMA.lower()
    assert "## Report" in IMPLEMENTER_SYSTEM_PROMPT_OLLAMA


# ---------------------------------------------------------------------------
# NEW Test 33 — REFLECT_DIAGNOSIS_NO_REPORT_HEADING existence and keyword
# ---------------------------------------------------------------------------

def test_reflect_diagnosis_no_report_heading_exists_and_keyword():
    """REFLECT_DIAGNOSIS_NO_REPORT_HEADING is a non-empty string."""
    from a3dasm._src.agent_prompts import (
        REFLECT_DIAGNOSIS_NO_REPORT_HEADING,
    )

    assert isinstance(REFLECT_DIAGNOSIS_NO_REPORT_HEADING, str), (
        "REFLECT_DIAGNOSIS_NO_REPORT_HEADING is not a str"
    )
    assert len(REFLECT_DIAGNOSIS_NO_REPORT_HEADING) > 0, (
        "REFLECT_DIAGNOSIS_NO_REPORT_HEADING is empty"
    )
    assert "## report" in REFLECT_DIAGNOSIS_NO_REPORT_HEADING.lower(), (
        "REFLECT_DIAGNOSIS_NO_REPORT_HEADING does not mention '## Report'"
    )


# ---------------------------------------------------------------------------
# NEW Test 34 — REFLECT_DIAGNOSIS_DEFAULT existence
# ---------------------------------------------------------------------------

def test_reflect_diagnosis_default_exists_and_keyword():
    """REFLECT_DIAGNOSIS_DEFAULT is a non-empty string."""
    from a3dasm._src.agent_prompts import (
        REFLECT_DIAGNOSIS_DEFAULT,
    )

    assert isinstance(REFLECT_DIAGNOSIS_DEFAULT, str), (
        "REFLECT_DIAGNOSIS_DEFAULT is not a str"
    )
    assert len(REFLECT_DIAGNOSIS_DEFAULT) > 0, (
        "REFLECT_DIAGNOSIS_DEFAULT is empty"
    )
    assert "malformed" in REFLECT_DIAGNOSIS_DEFAULT.lower(), (
        "REFLECT_DIAGNOSIS_DEFAULT does not mention 'malformed'"
    )


# ---------------------------------------------------------------------------
# Single canonical evaluator handle (both backends) — no raw-import bypass
# ---------------------------------------------------------------------------

def _impl_prompts():
    from a3dasm._src.agent_prompts import (
        IMPLEMENTER_SYSTEM_PROMPT,
        IMPLEMENTER_SYSTEM_PROMPT_OLLAMA,
    )
    return {
        "IMPLEMENTER_SYSTEM_PROMPT": IMPLEMENTER_SYSTEM_PROMPT,
        "IMPLEMENTER_SYSTEM_PROMPT_OLLAMA": IMPLEMENTER_SYSTEM_PROMPT_OLLAMA,
    }


def test_implementer_prompts_single_get_evaluator_path():
    for name, prompt in _impl_prompts().items():
        assert "get_evaluator()" in prompt, (
            f"{name} missing the single get_evaluator() handle"
        )


def test_implementer_prompts_have_no_raw_import_bypass():
    """The bypass the wet run exhibited was a prompted pattern — these must
    never reappear: raw evaluator import, sys.path hacks, or inner= wrapping
    that teaches reaching the oracle off the canonical handle."""
    for name, prompt in _impl_prompts().items():
        assert "sys.path.insert" not in prompt, f"{name} teaches sys.path hack"
        assert "get_evaluator(inner=" not in prompt, f"{name} uses inner="
        assert "from D001" not in prompt and "from D###" not in prompt, (
            f"{name} teaches a raw delegation-folder import"
        )
        assert "from evaluator import" not in prompt, (
            f"{name} teaches a raw evaluator import"
        )


def test_implementer_prompts_state_metering_scope():
    for name, prompt in _impl_prompts().items():
        low = prompt.lower()
        assert "metered" in low, f"{name} missing metering-scope statement"
        # surrogates/optimizers are free, not metered
        assert "surrogate" in low, f"{name} missing free-surrogate guidance"


def test_implementer_oracle_contract_consolidated():
    """BF-11: the oracle invariant is stated ONCE in the implementer prompt
    (one 'ORACLE DOOR' block) rather than re-sermonised across three drifting
    sections. Distinct facts survive the merge and the canonical home (KB 0001)
    is named so the rationale lives in one place."""
    from a3dasm._src.agents.implementer import IMPLEMENTER_SYSTEM_PROMPT
    p = IMPLEMENTER_SYSTEM_PROMPT
    assert p.count("THE ORACLE DOOR") == 1, "oracle contract not consolidated"
    # the three old drifting headers are gone
    assert "THE CANONICAL ORACLE" not in p
    assert "SINGLE source of truth" not in p
    assert "METERING SCOPE —" not in p
    # distinct facts preserved through the merge
    assert "RuntimeError" in p, "lost the don't-clobber-canonical-store fact"
    assert "D000" in p, "lost the reading-the-pool-is-free fact"
    # canonical home named (the ConsultHandbook chapter id)
    assert "evaluate-through-get-evaluator" in p, "KB 0001 not referenced"


def test_no_agent_bare_advertises_its_closures():
    """Contract (BF-13): a tool the agent is shown must be one it can CALL.

    Closures are exposed to the model ONLY by their MCP-qualified name
    (mcp__f3dasm_agent_tools__X); the generated <tools> catalog advertises them
    that way. An agent that ALSO hand-lists its closures by a bare call-signature
    (e.g. 'search_semantic_scholar(query, ...)') gives the model a name it cannot
    call — the model copies the bare form and the SDK rejects it ("No such tool
    available"). This is the general invariant the literature reviewer violated
    (confirmed live): advertise via the catalog, never a bare signature.

    Deterministic regression guard — no wet run needed. Loops every agent, so a
    future closure-injecting agent is covered too.
    """
    import re as _re
    import tempfile as _tmp

    from a3dasm._src.agents.critic import AdversarialCritiqueAgent
    from a3dasm._src.agents.datagenerator import DataGeneratorAgent
    from a3dasm._src.agents.implementer import F3dasmImplementerAgent
    from a3dasm._src.agents.literature import LiteratureReviewAgent
    from a3dasm._src.agents.strategizer import StrategizerAgent

    native = {"Bash", "Edit", "Read", "Write", "Glob", "Grep"}
    for Ag in (StrategizerAgent, F3dasmImplementerAgent, LiteratureReviewAgent,
               DataGeneratorAgent, AdversarialCritiqueAgent):
        ag = Ag()
        with _tmp.TemporaryDirectory() as d:
            closures = set(ag.build_closure_tools(study_dir=d) or {})
        bare = sorted(
            n for n in closures
            if n not in native
            and _re.search(rf"(?<![\w.]){_re.escape(n)}\s*\(", ag.system_prompt)
        )
        assert not bare, (
            f"{Ag.__name__} advertises closures by bare call-signature {bare} — "
            "the model can only call the qualified catalog name, so it will hit "
            "'No such tool available'. Present capabilities and defer to the "
            "<tools> catalog for exact names (as the strategizer does)."
        )


def test_resource_envelope_stanza_primes_ram_and_parallelism(tmp_path):
    """The static envelope injected at delegation start names cores, the HARD RAM
    cap, and free disk, and primes safe parallelism — and both preambles carry the
    {resources} placeholder so the stanza actually lands."""
    from types import SimpleNamespace
    from a3dasm._src.agent_runtime import AgenticRun
    from a3dasm._src.agent_prompts import (
        RUN_PATHS_PREAMBLE_TEMPLATE,
        WORKSPACE_PREAMBLE_TEMPLATE,
    )
    ns = SimpleNamespace(_mem_cap_bytes=4 * 1024 ** 3, study_dir=tmp_path)
    worker = AgenticRun._resource_stanza(ns, tmp_path, for_worker=True)
    strat = AgenticRun._resource_stanza(ns, tmp_path, for_worker=False)
    # Shared facts in BOTH roles.
    for s in (worker, strat):
        assert "CPU cores" in s
        assert "RAM cap 4.0 GB" in s and "KILLED" in s and "disk free" in s
    # Role-aware parallelism: the worker is primed to parallelize its
    # EVALUATIONS (compute, epistemically neutral); the strategizer is NOT
    # resource-nudged to fan out experiments (that's its design call, and the
    # nudge drove the monolithic run 20260628T224159).
    assert "EVALUATIONS within your campaign" in worker
    assert "parallel" not in strat.lower()
    assert "{resources}" in RUN_PATHS_PREAMBLE_TEMPLATE
    assert "{resources}" in WORKSPACE_PREAMBLE_TEMPLATE
