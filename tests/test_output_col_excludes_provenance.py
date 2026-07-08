"""Regression test for the notebook-deliverable `output_col` guidance.

Run 20260622T211227 (GATED) — the strategizer's retro flagged that the spec's
"portable" idiom `output_col = data.domain.output_names[0]` returns a metadata
column, not the objective. `Domain.from_data` SORTS the output names, and the
provenance columns (`_delegation_id`, `_source`, `_ts`, `_wall_ms`) are all
`_`-prefixed, so they sort BEFORE the real objective (ASCII `_` < lowercase).
The fix: select the first NON-provenance (`_`-prefixed) output column.

This test pins the sort-order trap: if a future Domain change reorders
`output_names`, the first assertion trips and we revisit the guidance.
"""
from __future__ import annotations

from f3dasm._src.design.domain import Domain


def test_output_names_zero_is_provenance_not_objective():
    # Mirror a loaded ledger: input col + provenance cols + the objective 'f'.
    domain = Domain.from_data(
        input_data=[{"x0": 1.0, "x1": 2.0}],
        output_data=[{"_delegation_id": "D1", "_source": "s", "f": 0.1}],
    )
    # The trap the old guidance fell into: [0] is a provenance column.
    assert domain.output_names[0].startswith("_"), (
        "expected a provenance column to sort first; "
        f"got {domain.output_names!r}"
    )
    # The fix: first non-provenance column is the true objective.
    output_col = next(c for c in domain.output_names if not c.startswith("_"))
    assert output_col == "f"
