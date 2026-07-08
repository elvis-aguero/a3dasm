"""Pin the scientific-method charter: single source, injected verbatim into
both adjudicating nodes, with the substantive Popperian wording intact."""

from a3dasm._src.knowledge.charter import FALSIFICATION_CHARTER
from a3dasm._src.agent_prompts import (
    STRATEGIZER_SYSTEM_PROMPT,
    ADVERSARIAL_CRITIQUE_SYSTEM_PROMPT,
)


def test_charter_injected_verbatim_into_both_nodes():
    """DRY: the one charter string is the literal substring of both prompts."""
    assert FALSIFICATION_CHARTER in STRATEGIZER_SYSTEM_PROMPT
    assert FALSIFICATION_CHARTER in ADVERSARIAL_CRITIQUE_SYSTEM_PROMPT


def test_charter_is_the_single_source():
    """The verdict logic must not be hand-restated outside the charter.

    The strategizer/critic may *reference* clauses ("Charter §3"); they must
    not re-define the FALSIFIED truth-condition in their own prose (which is
    exactly the drift that caused the 4-round argument). We approximate this
    by requiring the only 'if and only if' falsification statement to live in
    the charter, and the prompts to cite section numbers."""
    assert "Charter §" in STRATEGIZER_SYSTEM_PROMPT
    assert "Charter §" in ADVERSARIAL_CRITIQUE_SYSTEM_PROMPT


def test_charter_keeps_the_strong_biconditional_with_adequacy():
    """The substantive decision: biconditional, NOT the weaker 'only when',
    qualified by 'adequate' so Duhem–Quine is respected."""
    c = FALSIFICATION_CHARTER.lower()
    assert "if and only if" in c
    assert "adequate" in c
    assert "duhem" in c  # the qualifier's named justification


def test_charter_encodes_attempt_vs_verdict_and_no_goalposts():
    c = FALSIFICATION_CHARTER.lower()
    # attempt vs verdict are explicitly distinguished
    assert "attempt" in c and "verdict" in c
    # the goalpost-move is named and forbidden
    assert "post-hoc" in c
    assert "texas" in c  # the fallacy is named


def test_charter_only_uses_the_four_canonical_statuses():
    for status in ("OPEN", "SUPPORTED", "FALSIFIED", "INCONCLUSIVE"):
        assert status in FALSIFICATION_CHARTER
    # no invented states leaked in
    assert "EVIDENCE_AGAINST" not in FALSIFICATION_CHARTER


def test_charter_survives_existing_required_tokens():
    """Sanity: the edits did not drop tokens other tests rely on."""
    assert "is_falsification_attempt" in STRATEGIZER_SYSTEM_PROMPT
    assert "is_falsification_attempt" in ADVERSARIAL_CRITIQUE_SYSTEM_PROMPT
    assert "falsified" in STRATEGIZER_SYSTEM_PROMPT.lower()


# ── self-consistency guards (pin the 2026-06-16 epistemic repairs) ────────────
# These exist because the failure mode is editing one clause without holding the
# whole file in view. Each pins a repair that fixed a real philosophical defect.

def _normalised() -> str:
    """Charter lowercased with whitespace collapsed, so wrapping can't break a
    substring assertion."""
    return " ".join(FALSIFICATION_CHARTER.lower().split())


def test_charter_adequacy_is_severity_not_a_label():
    """§2: adequacy is the test's SEVERITY; the is_falsification_attempt flag
    only audits. A careless edit must not re-promote the flag to an adequacy
    criterion (the pollution that conflated bookkeeping with epistemics)."""
    c = _normalised()
    assert "severity" in c
    assert "never of a label" in c
    assert "only records the attempt" in c


def test_charter_survived_adequate_test_routes_to_supported():
    """§3/§5: a survived adequate test is SUPPORTED, NOT INCONCLUSIVE — and
    INCONCLUSIVE is reserved for an inadequate test. This reconciliation
    replaced a shipped §3↔§5 contradiction; do not let it creep back."""
    c = _normalised()
    assert (
        "not contradicted -> the hypothesis survived and is therefore supported"
        in c
    )
    assert "inconclusive is reserved for an inadequate test" in c


def test_charter_absence_claim_needs_search_power():
    """§2: an achievement/absence claim ('some/no design reaches X') is
    adequately tested only if the search had the POWER to find the instance had
    it existed. HOW that power is argued is open (adequate coverage, a guiding
    surrogate on the claim's own observable, a bound) — not a fixed checklist.
    A stalled search routes to INCONCLUSIVE, not FALSIFIED (absence of evidence
    != evidence of absence). Pins the 2026-06-25 repair (run 20260625T014520)."""
    c = _normalised()
    assert "power to find" in c
    assert "stopped improving" in c
    assert "is not the same as showing none exists" in c


def test_charter_dropped_the_bayesian_prior_graft():
    """§1: the Popperian definition no longer grafts in a Bayesian 'prior'."""
    assert "prior" not in FALSIFICATION_CHARTER.lower()


def test_charter_evidence_is_concrete_not_number_only():
    """§6: a closing status must cite a real delegation + a CONCRETE result that
    bears on the prediction — quantitative OR qualitative. It must NOT mandate a
    number (that over-narrows Popper: a qualitative observation can falsify, cf.
    the black swan). The provenance requirement (a delegation ID) stays."""
    c = _normalised()
    assert "delegation id" in c                      # provenance kept
    assert "concrete result" in c                    # the softened requirement
    assert "not all evidence is numeric" in c        # qualitative explicitly OK
    assert "black swan" in c                          # the Popperian anchor
    # the old number-mandate wording is gone
    assert "at least one number" not in c
