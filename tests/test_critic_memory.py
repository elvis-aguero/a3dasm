"""Fix A — the critic sees its own earlier reviews for this run.

The critic is invoked one-shot per gate with no live session and no
RecallHistory; without injecting its prior reviews it can silently contradict
a verdict it reached an earlier round (the H1 whipsaw that drove the
REVISE-spin). These tests pin the bounded digest and that ``_invoke_critic``
prepends it.
"""

from __future__ import annotations

from a3dasm._src.nodes.critic_gate import (
    CriticGateMixin,
    _extract_md_section,
)


def _write_review(notes_dir, n: int, verdict: str, findings: str) -> None:
    d = notes_dir.parent / "critic_reviews"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"call_{n:03d}.md").write_text(
        "## Report\n\n"
        "### Actions taken\n- read files\n\n"
        f"### Findings\n{findings}\n\n"
        f"### Verdict\n{verdict}\n\n"
        "### Numbers\nverdict: X\n\n"
        "### Retrospective\n- CONSISTENCY: ok\n",
        encoding="utf-8",
    )


class _Stub(CriticGateMixin):
    """Minimal carrier exposing only what the digest helper needs."""

    def __init__(self, notes_dir):
        self._current_notes_dir = notes_dir


def test_extract_md_section_pulls_only_that_section():
    text = (
        "### Findings\n[CRITICAL] foo\n[MINOR] bar\n\n"
        "### Verdict\nREJECT — because foo\n\n"
        "### Numbers\nverdict: REJECT\n"
    )
    assert _extract_md_section(text, "### Verdict") == "REJECT — because foo"
    assert "[CRITICAL] foo" in _extract_md_section(text, "### Findings")
    assert "verdict: REJECT" not in _extract_md_section(text, "### Findings")
    assert _extract_md_section(text, "### Missing") == ""


def test_no_prior_reviews_yields_empty_digest(tmp_path):
    notes = tmp_path / "notes"
    notes.mkdir()
    assert _Stub(notes)._prior_reviews_digest() == ""


def test_digest_contains_prior_verdicts_and_findings_in_order(tmp_path):
    notes = tmp_path / "notes"
    notes.mkdir()
    _write_review(notes, 1, "REVISE — H2 wrong", "[MAJOR] H2 status off")
    _write_review(notes, 2, "REJECT — H1 wrong", "[CRITICAL] H1 verdict")
    digest = _Stub(notes)._prior_reviews_digest()
    assert "<prior_reviews_this_run>" in digest
    assert "call_001" in digest and "call_002" in digest
    assert "REVISE — H2 wrong" in digest
    assert "[CRITICAL] H1 verdict" in digest
    # ordering preserved
    assert digest.index("call_001") < digest.index("call_002")
    # consistency instruction present
    assert "CONSISTENT" in digest
    assert "reverse" in digest.lower()


def test_digest_excludes_actions_numbers_retrospective(tmp_path):
    notes = tmp_path / "notes"
    notes.mkdir()
    _write_review(notes, 1, "PASS", "[MINOR] tiny")
    digest = _Stub(notes)._prior_reviews_digest()
    assert "read files" not in digest          # Actions taken
    assert "CONSISTENCY: ok" not in digest      # Retrospective
    assert "verdict: X" not in digest           # Numbers


def test_digest_is_bounded_to_recent_reviews(tmp_path):
    notes = tmp_path / "notes"
    notes.mkdir()
    for i in range(1, 9):  # 8 reviews; only newest 6 kept
        _write_review(notes, i, f"REVISE r{i}", f"[MAJOR] finding {i}")
    digest = _Stub(notes)._prior_reviews_digest()
    assert "call_001" not in digest
    assert "call_002" not in digest
    assert "call_008" in digest
    assert "elided" in digest


def test_invoke_critic_prepends_digest_to_task_msg(tmp_path):
    """Integration: _invoke_critic prepends the digest before the task."""
    notes = tmp_path / "notes"
    notes.mkdir()
    _write_review(notes, 1, "REVISE — earlier", "[MAJOR] earlier objection")

    captured = {}

    class _CaptureAdapter:
        last_usage: dict = {}
        model = "stub"

        def copy(self):
            return self

        def invoke(self, messages):
            captured["msg"] = messages[0]["content"]
            return "## Report\n### Verdict\nPASS\n"

    class _Node(CriticGateMixin):
        def __init__(self):
            self._current_notes_dir = notes
            self._worker_adapters = {"critic": _CaptureAdapter()}
            self._outgoing = ["critic"]

            class _Spec:
                nodes = {"critic": type("A", (), {"role": "critic"})()}

            self._spec = _Spec()

        def _record_usage(self, *a, **k):
            pass

        def _role_of(self, name):
            return "critic"

        def _persist_critic_review(self, *a, **k):
            pass

        def _record_retrospective(self, *a, **k):
            pass

    out = _Node()._invoke_critic("<mode>GATE</mode>\n\nTHE ACTUAL TASK")
    assert out.strip().endswith("PASS")
    assert "<prior_reviews_this_run>" in captured["msg"]
    assert "earlier objection" in captured["msg"]
    # digest comes BEFORE the task body
    assert captured["msg"].index("prior_reviews_this_run") < captured[
        "msg"
    ].index("THE ACTUAL TASK")
