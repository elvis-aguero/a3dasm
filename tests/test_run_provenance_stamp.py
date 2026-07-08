"""The run-provenance stamp REPLACES, not appends — the study-scoped
pipeline.ipynb must not accumulate prior runs' metadata cells.

Regression for run 20260630T164908: its deliverable carried the previous run's
(20260629T191754) run_dir/evals_used because the close-stamp appended an
unnamed markdown cell every run and never reset the study-scoped notebook.
"""
from __future__ import annotations

import nbformat

from a3dasm._src.notebook_exec import (
    RUN_PROVENANCE_CELL,
    stamp_run_provenance,
)


def _stamps(nb):
    return [c for c in nb.cells
            if (getattr(c, "metadata", None) or {}).get("name")
            == RUN_PROVENANCE_CELL]


def test_stamp_replaces_prior_and_stays_singular_and_named():
    nb = nbformat.v4.new_notebook()
    nb.cells.append(nbformat.v4.new_markdown_cell("# authored deliverable cell"))

    stamp_run_provenance(nb, "run=RUN_A evals=125")
    stamp_run_provenance(nb, "run=RUN_B evals=283")   # next run

    stamps = _stamps(nb)
    assert len(stamps) == 1, "provenance cell accumulated across runs"
    assert "RUN_B" in stamps[0].source and "RUN_A" not in stamps[0].source
    # the authored cell is untouched
    assert any("authored deliverable" in c.source for c in nb.cells)
    # named → manageable by the structured notebook tools
    assert stamps[0].metadata.get("name") == RUN_PROVENANCE_CELL


def test_stamp_cleans_a_legacy_unnamed_then_named_going_forward():
    # simulate the old bug's residue: a stale UNNAMED stamp already present
    nb = nbformat.v4.new_notebook()
    nb.cells.append(nbformat.v4.new_markdown_cell("run=OLD evals=999"))  # unnamed
    stamp_run_provenance(nb, "run=NEW evals=1")
    # exactly one named stamp; the new content wins
    assert len(_stamps(nb)) == 1
    assert "NEW" in _stamps(nb)[0].source
