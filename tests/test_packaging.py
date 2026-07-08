"""a3dasm dependency partition: a torch-free core vs the heavy `extra`.

a3dasm's core dependencies must run a functional literature reviewer (pymupdf
PDF->MD + BM25 retrieval + lightweight paper search) without pulling torch. The
one torch-puller (docling) lives in the `extra` optional-dependency as a quality
upgrade. These tests pin that partition and the graceful lean-image degradation.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import tomllib

_PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"


def _dep_names(reqs: list[str]) -> set[str]:
    """Normalised distribution names from a list of requirement strings."""
    names = set()
    for r in reqs:
        head = re.split(r"[;\[]", r, maxsplit=1)[0]
        name = re.split(r"[<>=!~ ]", head, maxsplit=1)[0].strip().lower()
        if name:
            names.add(name.replace("_", "-"))
    return names


def _pyproject() -> dict:
    return tomllib.loads(_PYPROJECT.read_text())


def _core_deps() -> set[str]:
    return _dep_names(_pyproject()["project"]["dependencies"])


def _extras() -> dict:
    return {
        k: _dep_names(v)
        for k, v in _pyproject()["project"]["optional-dependencies"].items()
    }


def test_core_depends_on_f3dasm():
    """a3dasm builds on f3dasm; it must declare it as a dependency."""
    assert "f3dasm" in _core_deps()


def test_core_keeps_a_working_reviewer():
    """A functional lean reviewer lives in core: pymupdf + BM25 + paper search."""
    core = _core_deps()
    assert "pymupdf" in core
    assert "rank-bm25" in core
    assert "semanticscholar" in core


def test_core_is_torch_free():
    """The torch-puller must NOT be a core dependency."""
    assert "docling" not in _core_deps(), "docling must live in the `extra` extra"


def test_extra_holds_the_torch_puller():
    extras = _extras()
    assert "extra" in extras, "the `extra` optional-dependency must exist"
    assert "docling" in extras["extra"], "`extra` must provide docling"


def test_all_aliases_extra():
    """`all` folds in `extra` (self-referential alias normalises to a3dasm)."""
    extras = _extras()
    allx = extras["all"]
    assert allx == {"a3dasm"} or allx <= extras["extra"]


def test_literature_corpus_degrades_to_bm25_without_heavy_deps(
    tmp_path, monkeypatch
):
    """Lean-image path: with docling AND fastembed unimportable, LiteratureCorpus
    still constructs and retrieval falls back to BM25 (dense embedder is None)."""
    import a3dasm._src.literature_corpus as lc

    monkeypatch.setitem(sys.modules, "fastembed", None)
    monkeypatch.setitem(sys.modules, "docling", None)
    monkeypatch.setitem(sys.modules, "docling.document_converter", None)
    monkeypatch.setattr(lc, "_subprocess_embedder_state", False)

    corpus = lc.LiteratureCorpus(tmp_path)
    assert corpus._get_embedding_model() is None
