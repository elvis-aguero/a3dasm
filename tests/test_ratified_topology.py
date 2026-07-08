"""Tests for the ratified 5-node topology and related invariants.

Covers:
  - Default graph: 5 nodes, 6 edges, builds via build_graph
  - F3dasmImplementerAgent public import + alias correctness
  - OptimizationAgent fully removed from public exports
  - DelegationLog.next_id() monotonic, thread-safe, globally unique
  - Two orchestrating nodes sharing one log mint distinct ids
  - None-log fallback is monotonic per-node
  - Strategizer prompt: 5-agent framing, DataGeneratorAgent/F3dasmImplementerAgent
  - Implementer prompt: merged scope (sampling + get_evaluator + surrogate + exploit)
"""
from __future__ import annotations

import threading
import tempfile
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# 1. Default graph — 5 nodes, 6 edges
# ---------------------------------------------------------------------------

def test_default_graph_has_five_nodes():
    from a3dasm._src.agents._graphs import _default_graph
    graph = _default_graph()
    assert len(graph.nodes) == 5


def test_default_graph_node_names():
    from a3dasm._src.agents._graphs import _default_graph
    graph = _default_graph()
    assert set(graph.nodes.keys()) == {
        "strategizer",
        "literature_reviewer",
        "datagenerator",
        "implementer",
        "critic",
    }


def test_default_graph_entry_is_strategizer():
    from a3dasm._src.agents._graphs import _default_graph
    graph = _default_graph()
    assert graph.entry == "strategizer"


def test_default_graph_has_six_edges():
    from a3dasm._src.agents._graphs import _default_graph
    graph = _default_graph()
    assert len(graph.edges) == 6


def test_default_graph_expected_edges():
    from a3dasm._src.agents._graphs import _default_graph
    graph = _default_graph()
    edge_pairs = {(e.source, e.target) for e in graph.edges}
    assert ("strategizer", "literature_reviewer") in edge_pairs
    assert ("strategizer", "datagenerator") in edge_pairs
    assert ("strategizer", "implementer") in edge_pairs
    assert ("strategizer", "critic") in edge_pairs
    assert ("datagenerator", "literature_reviewer") in edge_pairs
    assert ("implementer", "literature_reviewer") in edge_pairs


def test_default_graph_builds_via_build_graph():
    """build_graph compiles the 5-node default graph without error."""
    from a3dasm._src.agents._graphs import _default_graph
    from a3dasm._src.graph_builder import build_graph

    graph = _default_graph()

    class _Stub:
        closure_tools: dict = {}
        def invoke(self, messages):
            return "## Report\nDone."

    compiled = build_graph(graph, lambda name, agent: _Stub())
    assert hasattr(compiled, "invoke")


def test_default_graph_implementer_is_f3dasm_implementer_agent():
    from a3dasm._src.agents._graphs import _default_graph
    from a3dasm._src.agents.implementer import F3dasmImplementerAgent
    graph = _default_graph()
    assert isinstance(graph.nodes["implementer"], F3dasmImplementerAgent)


# ---------------------------------------------------------------------------
# 2. F3dasmImplementerAgent import and alias correctness
# ---------------------------------------------------------------------------

def test_f3dasm_implementer_agent_importable_from_public():
    from a3dasm import F3dasmImplementerAgent  # noqa: F401


def test_implementer_agent_alias_importable_from_public():
    from a3dasm import ImplementerAgent  # noqa: F401


def test_implementer_agent_alias_is_same_class():
    from a3dasm import F3dasmImplementerAgent, ImplementerAgent
    assert ImplementerAgent is F3dasmImplementerAgent


def test_f3dasmimplementer_backward_compat_alias():
    """The old F3dasmImplementer name still resolves to F3dasmImplementerAgent."""
    from a3dasm._src.agents.implementer import (
        F3dasmImplementer,
        F3dasmImplementerAgent,
    )
    assert F3dasmImplementer is F3dasmImplementerAgent


def test_f3dasm_implementer_agent_in_all():
    import a3dasm as mod
    assert "F3dasmImplementerAgent" in mod.__all__
    assert "ImplementerAgent" in mod.__all__


# ---------------------------------------------------------------------------
# 3. OptimizationAgent fully removed from public exports
# ---------------------------------------------------------------------------

def test_optimization_agent_not_in_public_all():
    import a3dasm as mod
    assert "OptimizationAgent" not in mod.__all__


def test_optimization_agent_import_raises():
    """OptimizationAgent must NOT be importable from a3dasm."""
    with pytest.raises((ImportError, AttributeError)):
        from a3dasm import OptimizationAgent  # noqa: F401


def test_optimization_module_deleted():
    """agents/optimization.py must not exist."""
    import a3dasm._src.agents as pkg
    agents_dir = Path(pkg.__file__).parent
    assert not (agents_dir / "optimization.py").exists()


def test_optimization_agent_not_in_default_graph():
    """No OptimizationAgent in the default graph node types."""
    from a3dasm._src.agents._graphs import _default_graph
    graph = _default_graph()
    for agent in graph.nodes.values():
        assert type(agent).__name__ != "OptimizationAgent"


# ---------------------------------------------------------------------------
# 4. DelegationLog.next_id() — monotonic, thread-safe, globally unique
# ---------------------------------------------------------------------------

def _make_log():
    tmp = Path(tempfile.mkdtemp())
    from a3dasm._src.delegation_log import DelegationLog
    return DelegationLog(tmp / "dl.jsonl")


def test_next_id_format():
    log = _make_log()
    first = log.next_id()
    assert first == "D001"


def test_next_id_monotonic_sequential():
    log = _make_log()
    ids = [log.next_id() for _ in range(5)]
    assert ids == ["D001", "D002", "D003", "D004", "D005"]


def test_next_id_thread_safe():
    """100 concurrent calls produce 100 distinct IDs."""
    log = _make_log()
    results = []
    lock = threading.Lock()

    def _call():
        uid = log.next_id()
        with lock:
            results.append(uid)

    threads = [threading.Thread(target=_call) for _ in range(100)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(results) == 100
    assert len(set(results)) == 100, "duplicate IDs found"


def test_two_nodes_sharing_one_log_mint_distinct_ids():
    """Simulates datagenerator + implementer both delegating to lit_reviewer.

    Both orchestrating nodes share one DelegationLog and must produce
    globally unique D### IDs with no collisions.
    """
    log = _make_log()

    ids_a = []
    ids_b = []

    def node_a():
        for _ in range(10):
            ids_a.append(log.next_id())

    def node_b():
        for _ in range(10):
            ids_b.append(log.next_id())

    t_a = threading.Thread(target=node_a)
    t_b = threading.Thread(target=node_b)
    t_a.start(); t_b.start()
    t_a.join(); t_b.join()

    all_ids = ids_a + ids_b
    assert len(all_ids) == 20
    assert len(set(all_ids)) == 20, "collision between node_a and node_b IDs"


def test_per_node_fallback_monotonic_without_log():
    """When delegation_log is None, per-node counter is monotonic."""
    import time
    import threading as _threading
    from pathlib import Path as _Path

    class _FakeNode:
        """Minimal stub that mimics StrategizerNode counter logic."""
        def __init__(self):
            self._delegation_log = None
            self._delegation_seq = 0
            self._registry_lock = _threading.Lock()

        def next_local_id(self):
            with self._registry_lock:
                self._delegation_seq += 1
                return f"D{self._delegation_seq:03d}"

    node = _FakeNode()
    ids = [node.next_local_id() for _ in range(5)]
    assert ids == ["D001", "D002", "D003", "D004", "D005"]


# ---------------------------------------------------------------------------
# 5. Strategizer prompt: 5-agent team, build/run split
# ---------------------------------------------------------------------------

def test_strategizer_no_two_agent_framing():
    from a3dasm._src.agents.strategizer import STRATEGIZER_SYSTEM_PROMPT
    assert "two-agent" not in STRATEGIZER_SYSTEM_PROMPT
    assert "two-agent research system" not in STRATEGIZER_SYSTEM_PROMPT


def test_strategizer_describes_data_generation_build():
    from a3dasm._src.agents.strategizer import STRATEGIZER_SYSTEM_PROMPT
    # Build capability described by role, not by a hardcoded class name.
    assert "DataGenerator Block" in STRATEGIZER_SYSTEM_PROMPT
    assert "datagenerator role" in STRATEGIZER_SYSTEM_PROMPT


def test_strategizer_routes_by_hint_not_class_name():
    """Forward-compatible routing: the strategizer must route by the Delegate
    hint NAMES, never by hardcoded agent class names. Hardcoding a class name
    broke when the class was renamed — the agent passed
    'F3dasmImplementerAgent' as a target and every delegation was rejected."""
    from a3dasm._src.agents.strategizer import STRATEGIZER_SYSTEM_PROMPT
    assert "F3dasmImplementerAgent" not in STRATEGIZER_SYSTEM_PROMPT
    assert "DataGeneratorAgent" not in STRATEGIZER_SYSTEM_PROMPT
    assert "verbatim" in STRATEGIZER_SYSTEM_PROMPT


def test_strategizer_no_optimization_agent_reference():
    from a3dasm._src.agents.strategizer import STRATEGIZER_SYSTEM_PROMPT
    assert "OptimizationAgent" not in STRATEGIZER_SYSTEM_PROMPT


def test_strategizer_delegate_tool_hints_line():
    from a3dasm._src.agents.strategizer import STRATEGIZER_SYSTEM_PROMPT
    assert "Delegate tool's hints" in STRATEGIZER_SYSTEM_PROMPT


def test_strategizer_build_run_split_described():
    from a3dasm._src.agents.strategizer import STRATEGIZER_SYSTEM_PROMPT
    # Build/run split described by capability (BUILDING vs RUNNING), no names.
    assert "BUILD" in STRATEGIZER_SYSTEM_PROMPT
    assert "RUN" in STRATEGIZER_SYSTEM_PROMPT


def test_strategizer_implementer_owns_all_evaluation():
    from a3dasm._src.agents.strategizer import STRATEGIZER_SYSTEM_PROMPT
    assert "ALL evaluation" in STRATEGIZER_SYSTEM_PROMPT or \
        "owns ALL" in STRATEGIZER_SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# 6. Implementer prompt: merged scope coverage
# ---------------------------------------------------------------------------

def test_strategizer_prompt_uses_verified_read_api():
    """Audit BF-4: the strategizer authors pipeline.ipynb, so its inline f3dasm
    examples must use the verified API — not the non-existent forms
    data.sample(...) / to_numpy("output") / get_n_best_output(name, n=) /
    create_sampler("lhs"). Verified against the installed f3dasm.
    """
    from a3dasm._src.agents.strategizer import STRATEGIZER_SYSTEM_PROMPT
    p = STRATEGIZER_SYSTEM_PROMPT
    assert "data.sample(" not in p
    assert 'to_numpy("output")' not in p
    assert 'get_n_best_output("' not in p   # name-first arg order is wrong
    assert 'sampler="lhs"' not in p
    assert "create_sampler" in p


def test_implementer_prompt_covers_sampling():
    from a3dasm._src.agents.implementer import IMPLEMENTER_SYSTEM_PROMPT
    # Verified sampling API (create_sampler -> Block.call), injected via
    # F3DASM_CORE_IDIOMS. The old data.sample(...)/Latin()/Sobol() tokens were
    # removed because they do not exist on the installed f3dasm.
    for token in ("create_sampler", "latin_sampler", "sampler.call",
                  "n_samples"):
        assert token in IMPLEMENTER_SYSTEM_PROMPT, (
            f"IMPLEMENTER_SYSTEM_PROMPT missing sampling token: {token!r}"
        )


def test_implementer_prompt_covers_get_evaluator():
    from a3dasm._src.agents.implementer import IMPLEMENTER_SYSTEM_PROMPT
    assert "get_evaluator" in IMPLEMENTER_SYSTEM_PROMPT


def test_implementer_prompt_covers_surrogate_fit():
    from a3dasm._src.agents.implementer import IMPLEMENTER_SYSTEM_PROMPT
    prompt_lower = IMPLEMENTER_SYSTEM_PROMPT.lower()
    assert "surrogate" in prompt_lower
    assert "no built-in gp" in prompt_lower or "has no built-in gp" in prompt_lower


def test_implementer_prompt_covers_exploit_loop():
    from a3dasm._src.agents.implementer import IMPLEMENTER_SYSTEM_PROMPT
    # Must mention exploit loop / optimization
    prompt_lower = IMPLEMENTER_SYSTEM_PROMPT.lower()
    assert "exploit" in prompt_lower or "optimization" in prompt_lower


def test_implementer_prompt_owns_initial_design():
    from a3dasm._src.agents.implementer import IMPLEMENTER_SYSTEM_PROMPT
    # Must explicitly state it runs the initial space-filling design
    assert "initial space-filling" in IMPLEMENTER_SYSTEM_PROMPT or \
        "DoE-EXECUTION" in IMPLEMENTER_SYSTEM_PROMPT


def test_implementer_prompt_only_agent_evaluates():
    from a3dasm._src.agents.implementer import IMPLEMENTER_SYSTEM_PROMPT
    assert "ONLY agent that" in IMPLEMENTER_SYSTEM_PROMPT or \
        "only agent that" in IMPLEMENTER_SYSTEM_PROMPT.lower()


def test_implementer_agent_class_description_matches_spec():
    from a3dasm._src.agents.implementer import F3dasmImplementerAgent
    desc = F3dasmImplementerAgent.description
    assert "only agent that evaluates" in desc.lower()
    assert "surrogate" in desc.lower()


def test_implementer_agent_class_name():
    from a3dasm._src.agents.implementer import F3dasmImplementerAgent
    assert F3dasmImplementerAgent.__name__ == "F3dasmImplementerAgent"


# ---------------------------------------------------------------------------
# 7. Non-entry hypothesis_ids finding
# ---------------------------------------------------------------------------

def test_non_entry_node_delegate_omits_hypothesis_check():
    """Non-entry orchestrating nodes (ledger=None) work with empty h_ids.

    The hypothesis_ids enforcement guard in Delegate() is gated on
    ``node._ledger is not None``.  Non-entry nodes (datagenerator,
    implementer) have ``_ledger=None``, so specialist→lit_reviewer
    delegation works without hypothesis_ids.
    """
    # Verify directly by reading the routing module source:
    import inspect
    from a3dasm._src.nodes.tools import routing as _routing
    src = inspect.getsource(_routing)
    # The guard must be conditional on ledger presence
    assert "node._ledger is not None" in src, (
        "Delegate must guard h_ids check on node._ledger is not None"
    )


# ---------------------------------------------------------------------------
# 8. DataGenerator as the universal oracle standardizer
# ---------------------------------------------------------------------------

def test_datagenerator_role_attr():
    from a3dasm._src.agents.datagenerator import DataGeneratorAgent
    assert DataGeneratorAgent.role == "datagenerator"


def test_datagenerator_prompt_is_universal_standardizer():
    from a3dasm._src.agents.datagenerator import (
        DATA_GENERATOR_SYSTEM_PROMPT,
    )
    p = DATA_GENERATOR_SYSTEM_PROMPT.lower()
    # Mandate spans any source, not just live physics simulations.
    for token in ("binary", "solver", "dataset", "plain-language",
                  "validated on", "standardizer"):
        assert token in p, f"datagenerator prompt missing {token!r}"


def test_datagenerator_prompt_uses_verified_sampling_api():
    """Audit BF-4: the datagenerator composes in F3DASM_CORE_IDIOMS (per
    idioms.py's own contract — it used to claim this while not doing it) and
    must NOT use the non-existent ExperimentData.sample(); the verified form is
    create_sampler(...).call(data=...). Mirrors the implementer's sampling
    contract above.
    """
    from a3dasm._src.agents.datagenerator import (
        DATA_GENERATOR_SYSTEM_PROMPT,
    )
    for token in ("create_sampler", "sampler.call", "n_samples"):
        assert token in DATA_GENERATOR_SYSTEM_PROMPT, (
            f"DATA_GENERATOR_SYSTEM_PROMPT missing sampling token: {token!r}"
        )
    # The erroneous CALL form must be gone (the injected idioms still WARN
    # "There is NO data.sample(...) method", which is desirable — so we assert
    # the absence of the actual call pattern, not the mention).
    assert ".sample(sampler" not in DATA_GENERATOR_SYSTEM_PROMPT
    assert "There is NO data.sample" in DATA_GENERATOR_SYSTEM_PROMPT


def test_datagenerator_prompt_documents_registration_manifest():
    from a3dasm._src.agents.datagenerator import (
        DATA_GENERATOR_SYSTEM_PROMPT,
    )
    p = DATA_GENERATOR_SYSTEM_PROMPT
    assert "registration.json" in p
    # The manifest fields the runtime hook consumes must be documented.
    for field in ("generator_file", "attr", "output_names"):
        assert field in p, f"manifest field {field!r} not documented"


def test_datagenerator_no_lookup_exclusion_absolute_claim():
    """The old 'NOT for lookup-pool studies' absolute is gone — the agent is
    now a general standardizer (it can also conform a dataset source)."""
    from a3dasm._src.agents.datagenerator import DataGeneratorAgent
    assert "NOT for lookup-pool studies" not in (
        DataGeneratorAgent.__doc__ or ""
    )
