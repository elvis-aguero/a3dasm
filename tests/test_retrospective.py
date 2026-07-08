"""Worker retrospective: self-consistency-first exit interview.

Each worker ends its report with a ### Retrospective section auditing the
SYSTEM (instructions/contracts/tools), led by a CONSISTENCY: ok|flagged line.
The runtime parses it, persists to retrospectives.jsonl, and — the instant a
worker flags contradictory instructions — emits a diagnostic + notification.
"""
from __future__ import annotations

from a3dasm._src.nodes import _extract_report_section


class TestSectionExtractor:
    def test_extracts_retrospective_body(self):
        text = (
            "## Report\n\n### Conclusions\nFound x.\n\n"
            "### Numbers\nbest: 1.2\n\n"
            "### Retrospective\n- CONSISTENCY: ok\n- DECISION: used seed 0\n"
            "- FRICTION: none\n"
        )
        body = _extract_report_section(text, "Retrospective")
        assert "CONSISTENCY: ok" in body
        assert "DECISION: used seed 0" in body
        # Must not bleed earlier sections in
        assert "best: 1.2" not in body

    def test_absent_section_returns_empty(self):
        assert _extract_report_section("## Report\n### Numbers\na: 1", "Retrospective") == ""

    def test_stops_at_trailing_rule(self):
        text = "### Retrospective\n- CONSISTENCY: ok\n---\nRequired keys: ...\n"
        body = _extract_report_section(text, "Retrospective")
        assert "CONSISTENCY: ok" in body
        assert "Required keys" not in body

    def test_flag_detection_regex(self):
        import re
        flagged = "- CONSISTENCY: flagged — 'do X' vs 'never do X'"
        ok = "- CONSISTENCY: ok"
        assert re.search(r"CONSISTENCY:\s*flagged", flagged, re.I)
        assert not re.search(r"CONSISTENCY:\s*flagged", ok, re.I)


class TestReportValidationUsesAgentSections:
    """Audit BF-10: the worker report validator must check each agent against
    ITS OWN declared sections, not the implementer-shaped default. worker.py
    used to call _classify_response(text) with no sections, so a correct critic
    report (Findings/Verdict, no Conclusions/Files touched) was flagged
    malformed and handed the implementer retry prompt (the routing.py Delegate
    path already passed report_sections; worker.py did not — the O40 asymmetry).
    """

    _CRITIC_REPORT = (
        "## Report\n\n### Actions taken\n- read\n\n### Findings\n- none\n\n"
        "### Verdict\nPASS\n\n### Numbers\nverdict: PASS\n\n"
        "### Retrospective\n- CONSISTENCY: ok\n- DECISION: x\n- FRICTION: none\n"
        "- BLOCKED: none\n"
    )

    def test_correct_critic_report_passes_against_its_own_sections(self):
        from a3dasm._src.agents.critic import AdversarialCritiqueAgent
        from a3dasm._src.nodes.parsing import _classify_response
        sections = list(AdversarialCritiqueAgent.report_sections)
        assert _classify_response(self._CRITIC_REPORT, sections) is None

    def test_same_report_wrongly_flagged_by_implementer_default(self):
        from a3dasm._src.nodes.parsing import _classify_response
        # The default (implementer-shaped) demands Conclusions/Files touched,
        # which a critic never emits — the misclassification BF-10 prevents.
        assert _classify_response(self._CRITIC_REPORT) is not None

    def test_worker_node_stores_report_sections(self):
        from a3dasm._src.nodes.worker import WorkerNode

        class _Adapter:
            def __init__(self):
                self.closure_tools = {}
                self.native_tools = []

        node = WorkerNode(_Adapter(), name="critic",
                          report_sections=("### Findings", "### Verdict"))
        assert node._report_sections == ("### Findings", "### Verdict")


class TestPromptsCarryRetrospective:
    def test_implementer_claude_prompt(self):
        from a3dasm._src.agents.implementer import (
            IMPLEMENTER_SYSTEM_PROMPT,
        )
        assert "### Retrospective" in IMPLEMENTER_SYSTEM_PROMPT
        assert "CONSISTENCY: ok | flagged" in IMPLEMENTER_SYSTEM_PROMPT

    def test_implementer_ollama_prompt(self):
        from a3dasm._src.agent_prompts import (
            IMPLEMENTER_SYSTEM_PROMPT_OLLAMA,
        )
        assert "### Retrospective" in IMPLEMENTER_SYSTEM_PROMPT_OLLAMA
        assert "CONSISTENCY: ok | flagged" in IMPLEMENTER_SYSTEM_PROMPT_OLLAMA

    def test_datagenerator_prompt(self):
        from a3dasm._src.agents.datagenerator import (
            DATA_GENERATOR_SYSTEM_PROMPT,
        )
        assert "### Retrospective" in DATA_GENERATOR_SYSTEM_PROMPT

    def test_critic_prompt(self):
        from a3dasm._src.agents.critic import (
            ADVERSARIAL_CRITIQUE_SYSTEM_PROMPT,
        )
        assert "### Retrospective" in ADVERSARIAL_CRITIQUE_SYSTEM_PROMPT

    def test_literature_prompt(self):
        from a3dasm._src.agents.literature import (
            LITERATURE_REVIEW_SYSTEM_PROMPT,
        )
        assert "### Retrospective" in LITERATURE_REVIEW_SYSTEM_PROMPT

    def test_worker_retrospectives_carry_blocked_field(self):
        """Audit BF-9: worker retrospectives now carry the BLOCKED field
        (capability gaps). Previously only the orchestrator's runtime exit
        interview probed BLOCKED, so short-lived workers (implementer,
        datagenerator, critic, literature) had no channel to report 'a tool I
        needed and didn't have'. CLAUDE.md specifies four retrospective fields.
        """
        from a3dasm._src.agent_prompts import (
            IMPLEMENTER_SYSTEM_PROMPT_OLLAMA,
        )
        from a3dasm._src.agents.critic import (
            ADVERSARIAL_CRITIQUE_SYSTEM_PROMPT,
        )
        from a3dasm._src.agents.datagenerator import (
            DATA_GENERATOR_SYSTEM_PROMPT,
        )
        from a3dasm._src.agents.implementer import (
            IMPLEMENTER_SYSTEM_PROMPT,
        )
        from a3dasm._src.agents.literature import (
            LITERATURE_REVIEW_SYSTEM_PROMPT,
        )
        for prompt in (
            IMPLEMENTER_SYSTEM_PROMPT, IMPLEMENTER_SYSTEM_PROMPT_OLLAMA,
            DATA_GENERATOR_SYSTEM_PROMPT, ADVERSARIAL_CRITIQUE_SYSTEM_PROMPT,
            LITERATURE_REVIEW_SYSTEM_PROMPT,
        ):
            assert "BLOCKED:" in prompt

    def test_exit_interview_already_probes_blocked(self):
        """The orchestrator's post-Done exit interview already probed BLOCKED —
        this is why Stage-2 forensics found BLOCKED entries from the
        strategizer. Pinned so the worker change stays consistent with it."""
        from a3dasm._src.nodes import _EXIT_INTERVIEW
        assert "BLOCKED:" in _EXIT_INTERVIEW

    def test_strategizer_prompt_is_NOT_polluted(self):
        """The strategizer must NOT carry the interview in its working
        context — that would pollute every orchestration turn. The exit
        interview is asked by the runtime AFTER the critic accepts."""
        from a3dasm._src.agents.strategizer import (
            STRATEGIZER_SYSTEM_PROMPT,
        )
        assert "### Retrospective" not in STRATEGIZER_SYSTEM_PROMPT
        assert "<retrospective>" not in STRATEGIZER_SYSTEM_PROMPT

    def test_exit_interview_is_a_runtime_post_done_turn(self):
        from a3dasm._src.nodes import _EXIT_INTERVIEW
        # Asked only after acceptance; carries the same 3 questions.
        assert "accepted by the critic" in _EXIT_INTERVIEW
        assert "CONSISTENCY" in _EXIT_INTERVIEW
        assert "DECISION" in _EXIT_INTERVIEW
        assert "FRICTION" in _EXIT_INTERVIEW
        assert "Done() ONE more time" in _EXIT_INTERVIEW


class TestNoCanonicalSourceNudge:
    """Strategizer flowchart: when a datagenerator exists but no canonical
    source is registered, recommend delegating to it (soft, ≤3×)."""

    def _spec_with_datagenerator(self):
        from a3dasm._src.backends.base import Agent, Edge, Graph

        class S(Agent):
            role = "strategizer"
            description = "strategizer"

        class DG(Agent):
            role = "datagenerator"
            description = "datagenerator"

        return Graph(
            nodes={"strategizer": S(), "datagenerator": DG()},
            edges=(Edge("strategizer", "datagenerator"),),
            entry="strategizer",
        )

    def _node(self):
        from a3dasm._src.nodes import StrategizerNode

        class _Stub:
            def __init__(self):
                self.role = "strategizer"
                self.closure_tools = {}
                self.route_watcher = None
                self.last_usage = {}

            def invoke(self, messages):
                return "ok"

        return StrategizerNode(
            _Stub(), name="strategizer", outgoing=["datagenerator"],
            spec=self._spec_with_datagenerator(),
            worker_adapters={"datagenerator": _Stub()},
        )

    def test_finds_datagenerator(self):
        assert self._node()._find_datagenerator_name() == "datagenerator"

    def test_source_unregistered_when_entrypoint_absent(self, tmp_path):
        import json
        node = self._node()
        notes = tmp_path / "debug" / "strategizer_notes"
        notes.mkdir(parents=True)
        (notes.parent / "run_config.json").write_text(json.dumps(
            {"evaluator_entrypoint": None, "evaluator_lookup": None}))
        node._current_notes_dir = notes
        assert node._canonical_source_registered() is False

    def test_source_registered_when_entrypoint_present(self, tmp_path):
        import json
        node = self._node()
        notes = tmp_path / "debug" / "strategizer_notes"
        notes.mkdir(parents=True)
        (notes.parent / "run_config.json").write_text(json.dumps(
            {"evaluator_entrypoint": "workspace/e.py:f"}))
        node._current_notes_dir = notes
        assert node._canonical_source_registered() is True

    def test_lookup_counts_as_registered(self, tmp_path):
        import json
        node = self._node()
        notes = tmp_path / "debug" / "strategizer_notes"
        notes.mkdir(parents=True)
        (notes.parent / "run_config.json").write_text(json.dumps(
            {"evaluator_lookup": "experiment_data"}))
        node._current_notes_dir = notes
        assert node._canonical_source_registered() is True

    def test_nudge_counter_caps_at_three(self):
        # The cap field exists and starts at 0; __call__ increments up to 3.
        node = self._node()
        assert node._no_source_nudges == 0

    def test_report_sections_include_retrospective(self):
        from a3dasm._src.agents.critic import (
            AdversarialCritiqueAgent,
        )
        from a3dasm._src.agents.datagenerator import DataGeneratorAgent
        from a3dasm._src.agents.implementer import (
            F3dasmImplementerAgent,
        )
        from a3dasm._src.agents.literature import (
            LiteratureReviewAgent,
        )
        for agent in (F3dasmImplementerAgent, DataGeneratorAgent,
                      AdversarialCritiqueAgent, LiteratureReviewAgent):
            assert "### Retrospective" in agent.report_sections, agent.__name__


class TestStrategizerGranularity:
    def test_anti_monolith_rule_present(self):
        from a3dasm._src.agents.strategizer import (
            STRATEGIZER_SYSTEM_PROMPT,
        )
        assert "MONOLITHIC DELEGATION" in STRATEGIZER_SYSTEM_PROMPT
        assert "ONE bounded experiment" in STRATEGIZER_SYSTEM_PROMPT

    def test_hypothesis_framing_guidance_present(self):
        from a3dasm._src.agents.strategizer import (
            STRATEGIZER_SYSTEM_PROMPT,
        )
        # domain-neutral: no leaked example framings, but the principle is there
        assert "not a bet on which method" in STRATEGIZER_SYSTEM_PROMPT.lower() \
            or "NOT a bet on which method" in STRATEGIZER_SYSTEM_PROMPT


class TestRetrospectiveTextCap:
    def test_long_retrospective_not_truncated_at_2000(self, tmp_path):
        """B9 (run 20260624T021359): the strategizer's end-of-run retrospective
        is the highest-signal first-person record; the old 2000-char cap cut it
        mid-sentence (losing the BLOCKED field). The cap is now uniform at 8000."""
        import json

        from a3dasm._src.backends.base import Agent, Edge, Graph
        from a3dasm._src.nodes import StrategizerNode

        class A(Agent):
            role = "strategizer"
            tools = frozenset({"Done"})
            description = "s"

        class B(Agent):
            description = "i"

        spec = Graph(nodes={"strategizer": A(), "implementer": B()},
                     edges=(Edge("strategizer", "implementer"),), entry="strategizer")

        class _Stub:
            def __init__(self):
                self.closure_tools = {}

            def invoke(self, messages):
                return "ok"

        notes = tmp_path / "debug" / "strategizer_notes"
        notes.mkdir(parents=True)
        node = StrategizerNode(_Stub(), name="strategizer", outgoing=["implementer"],
                               spec=spec, study_dir=tmp_path)
        node._current_notes_dir = notes

        body = "- CONSISTENCY: ok\n- BLOCKED: " + ("x" * 3000)
        node._record_retrospective(
            "strategizer", "DONE", f"## Report\n\n### Retrospective\n{body}\n")

        line = (tmp_path / "debug" / "retrospectives.jsonl").read_text().splitlines()[0]
        rec = json.loads(line)
        assert len(rec["text"]) > 2000  # pre-fix this was capped at exactly 2000
        assert rec["text"].rstrip().endswith("x")  # the tail survived, not cut off
