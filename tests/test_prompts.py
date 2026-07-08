"""Prompts must match the runtime — no stale references."""

from a3dasm._src.agent_prompts import (
    CHECKPOINT_STRATEGIZER_PROMPT,
    IMPLEMENTER_SYSTEM_PROMPT_OLLAMA,
    RUN_PATHS_PREAMBLE_TEMPLATE,
)
from a3dasm._src.agents.critic import (
    ADVERSARIAL_CRITIQUE_SYSTEM_PROMPT,
)
from a3dasm._src.agents.strategizer import (
    STRATEGIZER_SYSTEM_PROMPT,
)


def test_no_stale_hypotheses_md_references():
    from a3dasm._src.agent_prompts import (
        RUN_PATHS_PREAMBLE_TEMPLATE,
    )
    assert "hypotheses.md" not in STRATEGIZER_SYSTEM_PROMPT
    assert "hypotheses.md" not in CHECKPOINT_STRATEGIZER_PROMPT
    assert "hypotheses.md" not in RUN_PATHS_PREAMBLE_TEMPLATE
    assert "hypotheses.md" not in ADVERSARIAL_CRITIQUE_SYSTEM_PROMPT


def test_no_false_runtime_enforcement_claim():
    assert "enforced at runtime" not in STRATEGIZER_SYSTEM_PROMPT


def test_strategizer_documents_new_signatures():
    for token in ("falsification_criterion", "prediction", "prior",
                  "posterior", "is_falsification_attempt"):
        assert token in STRATEGIZER_SYSTEM_PROMPT, token


def test_implementer_doe_example_has_no_store_after_evaluator():
    """B7 (run 20260624T021359): the implementer's DoE example must not show a
    data.store() AFTER get_evaluator()/flush() — that re-writes FINISHED rows as
    IN_PROGRESS and corrupts the ledger (contradicts the prompt's own rule)."""
    from a3dasm._src.agents.implementer import IMPLEMENTER_SYSTEM_PROMPT
    assert 'data.store("{delegation_id}/results")' not in IMPLEMENTER_SYSTEM_PROMPT
    assert "Do NOT call data.store() afterwards" in IMPLEMENTER_SYSTEM_PROMPT


def test_deliverable_spec_requires_explicit_objective_column():
    """B3 (run 20260624T021359): the notebook spec must NOT recommend the
    sort-order auto-detect idiom (a constraint flag sorts before the objective
    and is silently picked); it must require naming the objective explicitly."""
    from a3dasm._src.notebook_exec import notebook_deliverable_spec
    spec = notebook_deliverable_spec()
    assert "not c.startswith('_')" not in spec  # the fragile idiom is gone
    assert "EXPLICITLY" in spec
    assert "evaluator_output_names" in spec


def test_strategizer_requires_primary_criteria_met_before_done():
    """Closure discipline (run 20260624T021359): Done() must require the PRIMARY
    success criteria to be MET, not merely tested, and frame budget as runway."""
    p = STRATEGIZER_SYSTEM_PROMPT
    assert "PRIMARY success criterion" in p or "primary success criterion" in p.lower()
    assert "INCONCLUSIVE" in p and "RUNWAY" in p.upper()


def test_checkpoint_asks_for_ledger_digest():
    assert "posterior" in CHECKPOINT_STRATEGIZER_PROMPT


def test_critic_uses_falsification_flags():
    assert "is_falsification_attempt" in \
        ADVERSARIAL_CRITIQUE_SYSTEM_PROMPT


def test_prompts_are_case_generic():
    # no references to specific past studies in any prompt
    for prompt in (STRATEGIZER_SYSTEM_PROMPT,
                   CHECKPOINT_STRATEGIZER_PROMPT,
                   ADVERSARIAL_CRITIQUE_SYSTEM_PROMPT):
        for stale in ("supercompressible", "black_box_8d",
                      "sigma_crit"):
            assert stale not in prompt, stale


def test_strategizer_mentions_science_monitor():
    assert "[SCIENCE MONITOR" in STRATEGIZER_SYSTEM_PROMPT


def test_run_paths_preamble_has_experiment_data_dir():
    from a3dasm._src.agent_prompts import (
        RUN_PATHS_PREAMBLE_TEMPLATE,
    )
    # must format cleanly with the new required field
    out = RUN_PATHS_PREAMBLE_TEMPLATE.format(
        study_dir="/s", run_dir="/s/runs/T", debug_dir="/s/runs/T/debug",
        notes_dir="/s/runs/T/debug/strategizer_notes",
        experiment_data_dir="/s/runs/T/experiment_data",
        resources="",
        knowledge="",
    )
    # exposed as the canonical-store project_dir with an explicit from_file call
    assert "canonical_store" in out
    assert "/s/runs/T/experiment_data" in out
    assert "from_file(project_dir=" in out


def test_pipeline_deliverable_is_lazy_and_self_asserting():
    # BF-12: the lazy-reproduction contract lives in ONE place — the injected
    # <deliverable_format> spec, not a second copy in the strategizer base
    # prompt. Assert against the assembled prompt the strategizer actually
    # receives (base + injected spec).
    from a3dasm._src.notebook_exec import notebook_deliverable_spec
    p = STRATEGIZER_SYSTEM_PROMPT + notebook_deliverable_spec("strategizer")
    # the deliverable is the notebook; its code cells load the ledger and reach
    # the oracle via get_evaluator()
    assert "pipeline.ipynb" in p
    assert "ExperimentData.from_file" in p
    assert "get_evaluator()" in p
    # lazy: re-running must add zero new oracle evals
    assert "ZERO new oracle evals" in p
    # self-asserting headline, never hardcoded
    assert "REPRODUCED" in p
    assert "hardcode" in p.lower()


def test_strategizer_has_pipeline_authoring_primer():
    """The strategizer AUTHORS pipeline.ipynb, so the assembled prompt must
    carry a concrete f3dasm primer — load the ledger, read its frames, cache
    heavy blocks, and derive (not hardcode) the headline. The reproduction
    rules live in the injected spec (BF-12), so assert the assembled prompt."""
    from a3dasm._src.notebook_exec import notebook_deliverable_spec
    base = STRATEGIZER_SYSTEM_PROMPT
    p = base + notebook_deliverable_spec("strategizer")
    # concrete ledger-read API, not just from_file (lives in the base primer)
    assert "to_pandas()" in base
    # heavy non-oracle blocks must cache-or-load; store path via env
    assert "CACHE-OR-LOAD" in p
    assert "F3DASM_CANONICAL_STORE" in p
    # explicit anti-hardcoding guidance the critic can lean on
    low = p.lower()
    assert "hardcod" in low and "derive" in low
    # the base prompt points at the single-source deliverable_format spec
    assert "deliverable_format" in base.lower()


# ---------------------------------------------------------------------------
# Prompt-audit tests (audit fixes A-M)
# ---------------------------------------------------------------------------

def test_no_two_agent_framing_in_any_prompt():
    """No prompt should use the stale 'two-agent' framing."""
    from a3dasm._src.agents.implementer import (
        IMPLEMENTER_SYSTEM_PROMPT,
    )
    from a3dasm._src.agents.literature import (
        LITERATURE_REVIEW_SYSTEM_PROMPT,
    )
    for name, prompt in (
        ("STRATEGIZER", STRATEGIZER_SYSTEM_PROMPT),
        ("IMPLEMENTER", IMPLEMENTER_SYSTEM_PROMPT),
        ("OLLAMA", IMPLEMENTER_SYSTEM_PROMPT_OLLAMA),
        ("LITERATURE", LITERATURE_REVIEW_SYSTEM_PROMPT),
        ("CRITIC", ADVERSARIAL_CRITIQUE_SYSTEM_PROMPT),
    ):
        assert "two-agent" not in prompt.lower(), (
            f"{name} prompt contains stale 'two-agent' framing"
        )


def test_strategizer_ledger_read_tools_documented(tmp_path):
    """RecallStore, QueryStore, RecallHistory are documented to the strategizer.

    Post-Spec-B these live in the GENERATED <tools> catalog (the single source
    of truth, derived from the live closures), not the static base prompt — so
    assert the catalog of a real strategizer node, which is what the model
    actually receives appended to its system prompt."""
    from a3dasm._src.backends.base import Agent, Edge, Graph
    from a3dasm._src.nodes import StrategizerNode
    from a3dasm._src.tool_catalog import render_tool_catalog

    class _Stub:
        def __init__(self):
            self.closure_tools: dict = {}
            self.last_usage: dict = {}
            self.model = "m"

        def invoke(self, messages):
            return ""

    class A(Agent):
        role = "strategizer"
        tools = frozenset({"Done", "HypothesisPropose", "HypothesisUpdate", "HypothesisList", "HypothesisGet", "LinkFalsificationAttempt", "MilestoneList", "MilestonePropose", "MilestoneComplete", "MilestoneSkip", "RecallStore", "QueryStore"})
        description = "strategizer"

    class B(Agent):
        role = "implementer"
        description = "implementer"

    spec = Graph(
        nodes={"strategizer": A(), "implementer": B()},
        edges=(Edge("strategizer", "implementer"),), entry="strategizer")
    from a3dasm._src.delegation_log import DelegationLog
    n = StrategizerNode(
        _Stub(), name="strategizer", outgoing=["implementer"], spec=spec,
        worker_adapters={"implementer": _Stub()},
        delegation_log=DelegationLog(tmp_path / "dlog.jsonl"))  # → RecallHistory
    cat = render_tool_catalog(n.adapter.closure_tools)
    for tool in ("RecallStore", "QueryStore", "RecallHistory"):
        assert tool in cat, f"generated tool catalog missing '{tool}'"


def test_strategizer_ledger_is_ground_truth():
    """Strategizer prompt states the canonical ledger is ground truth."""
    lower = STRATEGIZER_SYSTEM_PROMPT.lower()
    assert "ground truth" in lower, (
        "STRATEGIZER_SYSTEM_PROMPT missing 'ground truth' ledger-"
        "evidence statement"
    )


def test_strategizer_no_builtin_gp_claim():
    """Strategizer architecture section must not list CMA-ES/GP as
    f3dasm-native; tpesampler should be present near optimization."""
    lower = STRATEGIZER_SYSTEM_PROMPT.lower()
    assert "tpesampler" in lower, (
        "STRATEGIZER_SYSTEM_PROMPT does not mention 'tpesampler' in "
        "optimization guidance"
    )


def test_ollama_implementer_contains_get_evaluator():
    """Ollama implementer prompt must document get_evaluator."""
    assert "get_evaluator" in IMPLEMENTER_SYSTEM_PROMPT_OLLAMA, (
        "IMPLEMENTER_SYSTEM_PROMPT_OLLAMA missing 'get_evaluator'"
    )


def test_run_paths_renders_delegations_not_workspace():
    """RUN_PATHS_PREAMBLE_TEMPLATE must render 'delegations', not
    '/workspace', as workspace_dir."""
    out = RUN_PATHS_PREAMBLE_TEMPLATE.format(
        study_dir="/s",
        run_dir="/s/runs/T",
        debug_dir="/s/runs/T/debug",
        notes_dir="/s/runs/T/debug/strategizer_notes",
        experiment_data_dir="/s/runs/T/experiment_data",
        resources="",
        knowledge="",
    )
    assert "delegations" in out, (
        "RUN_PATHS_PREAMBLE_TEMPLATE did not render 'delegations'"
    )
    assert "/workspace" not in out, (
        "RUN_PATHS_PREAMBLE_TEMPLATE still renders '/workspace'"
    )
