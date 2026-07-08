"""ConsultHandbook is injected UNIVERSALLY at adapter construction.

Every node's adapter must get the read-only handbook lookup equally, from the
single construction point (agent_runtime._make_adapter) — not duplicated per
worker/strategizer path, and not special-cased per agent. This pins that the
critic (invoked off the Delegate path) gets it just like a worker, and that the
tool is a runtime closure (its description owned by _consult_handbook), never
hardcoded into a prompt.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from a3dasm._src.agent_runtime import AgenticRun
from a3dasm._src.agents.critic import AdversarialCritiqueAgent
from a3dasm._src.agents.implementer import ImplementerAgent


class _NoOutgoing:
    def outgoing(self, name):
        return []


@pytest.mark.parametrize(
    "name,agent",
    [("critic", AdversarialCritiqueAgent()), ("implementer", ImplementerAgent())],
)
def test_make_adapter_injects_consult_handbook(tmp_path, name, agent):
    run = AgenticRun(study_dir=tmp_path)        # no config.yaml → defaults
    run._run_dir = None
    run._graph_spec = _NoOutgoing()

    adapter = run._make_adapter(name, agent)

    # Universal, runtime-injected closure (NOT declared in the agent prompt).
    assert "ConsultHandbook" in adapter.closure_tools
    # It is the shared handbook function (one source), not a per-agent copy.
    from a3dasm._src.nodes.parsing import _consult_handbook
    assert adapter.closure_tools["ConsultHandbook"] is _consult_handbook


def test_notebook_spec_always_injected(tmp_path):
    """The pipeline.ipynb deliverable contract is ALWAYS injected into the
    strategizer prompt — the system is committed to the notebook deliverable."""
    from a3dasm._src import settings
    from a3dasm._src.agents.strategizer import StrategizerAgent

    run = AgenticRun(study_dir=tmp_path)
    run._run_dir = None
    run._graph_spec = _NoOutgoing()
    try:
        settings.configure({})  # default — no flag needed
        b = run._make_adapter("strategizer", StrategizerAgent())
        assert "DELIVERABLE = pipeline.ipynb" in b.system_prompt
        assert "name='doe'" in b.system_prompt  # four-pillar template present
        # Only the strategizer is told to AUTHOR with the structured tools.
        assert "AUTHOR IT WITH THE\nSTRUCTURED TOOLS" in b.system_prompt
        assert "AddPipelineCell(phase, why, code)" in b.system_prompt
    finally:
        settings.configure({})


def test_notebook_spec_is_role_aware(tmp_path):
    """The implementer/critic get the STRUCTURE but NOT the 'author it with the
    structured tools' imperative — those tools are granted only to the
    strategizer, so telling the others to call them would be a dead instruction.
    """
    from a3dasm._src.agents.critic import AdversarialCritiqueAgent
    from a3dasm._src.agents.implementer import ImplementerAgent

    run = AgenticRun(study_dir=tmp_path)
    run._run_dir = None
    run._graph_spec = _NoOutgoing()

    impl = run._make_adapter("implementer", ImplementerAgent())
    crit = run._make_adapter("critic", AdversarialCritiqueAgent())
    for p in (impl.system_prompt, crit.system_prompt):
        assert "pipeline.ipynb" in p              # they know the deliverable
        assert "name='doe'" in p                  # and its structure
        assert "AUTHOR IT WITH" not in p          # but are NOT told to author it
        assert "AddPipelineCell(phase, why, code)" not in p
