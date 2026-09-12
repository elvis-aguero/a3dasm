"""The prompt/gate provenance map must keep telling the truth.

``internal/tools/promptmap.py`` renders a map that cites, for every piece of
text an agent sees and every gate that can stop it, an exact ``file:line``.
Such a map is only worth anything if a stale one FAILS rather than quietly
citing code that moved. These tests are that failure:

* every gate symbol in the registry still resolves (the generator raises
  ``SystemExit`` when one does not);
* every section of every assembled prompt still resolves to a real file and
  a line inside it — so a prompt that moves takes its citation along;
* the Done() sequence is still parsed out of ``feedback.py``'s own tuple
  rather than hand-listed.

No network, no model, no API cost: everything is read off the live graph
objects and the AST.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_GEN = _ROOT / "internal" / "tools" / "promptmap.py"


def _load():
    spec = importlib.util.spec_from_file_location("_promptmap", _GEN)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def promptmap():
    if not _GEN.exists():  # pragma: no cover - the generator is checked in
        pytest.skip("promptmap generator not present")
    return _load()


@pytest.fixture(scope="module")
def data(promptmap):
    return promptmap.build()


def test_every_gate_symbol_still_exists(promptmap):
    """A renamed or deleted gate must break the build, not the meeting."""
    gates = promptmap.build_gates()
    assert gates, "the gate registry is empty"
    for gate in gates:
        assert gate["line"] > 0
        assert (_ROOT / gate["file"]).exists(), gate["symbol"]


def test_done_sequence_is_read_from_the_source_tuple(promptmap):
    chain = promptmap.done_chain()
    assert chain[0] == "_pending_refusal", (
        "Done()'s first gate changed; the map's ordering claim is now wrong")
    assert len(chain) >= 3


def test_every_prompt_section_resolves_to_a_real_line(data):
    """No section may be UNRESOLVED: a map with a blank citation is worse
    than no map, because the blank is the one a reviewer will ask about."""
    unresolved = []
    for role in data["roles"]:
        for layer in role["layers"]:
            for section in layer["sections"]:
                source = section.get("source")
                if not source:
                    unresolved.append(f"{role['id']}/{layer['kind']}/{section['tag']}")
                    continue
                path = _ROOT / source["file"]
                assert path.exists(), source["file"]
                n_lines = path.read_text(encoding="utf-8").count("\n") + 1
                assert 1 <= source["line"] <= n_lines, source
    assert not unresolved, f"unresolved prompt sections: {unresolved}"


def test_shared_blocks_are_cited_at_their_definition_not_their_use(data):
    """The charter is injected into two roles; both must cite knowledge/,
    which is what makes "cite a clause number" a shared contract."""
    charter = [b for b in data["shared"] if b["name"] == "FALSIFICATION_CHARTER"]
    assert charter, "FALSIFICATION_CHARTER vanished from knowledge/charter.py"
    citing = [
        section["injects"]["file"]
        for role in data["roles"]
        for layer in role["layers"]
        for section in layer["sections"]
        if section.get("injects", {}).get("name") == "FALSIFICATION_CHARTER"
    ]
    assert len(citing) >= 2, "the charter should reach at least two roles"
    assert all(path.endswith("knowledge/charter.py") for path in citing), citing


def test_entry_node_carries_the_run_paths_preamble(data):
    entry = [r for r in data["roles"] if r["entry"]]
    assert len(entry) == 1
    kinds = [layer["label"] for layer in entry[0]["layers"]]
    assert "RUN_PATHS_PREAMBLE_TEMPLATE" in kinds
    workers = [r for r in data["roles"] if not r["entry"]]
    for worker in workers:
        labels = [layer["label"] for layer in worker["layers"]]
        assert "WORKSPACE_PREAMBLE_TEMPLATE" in labels, worker["id"]
