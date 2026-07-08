"""QueryStore 'best' selection (friction #6 fix): must honor optimization
direction and never return an infeasibility placeholder as 'best'."""
from __future__ import annotations

import pandas as pd

from a3dasm._src.nodes.tools.routing import _select_best_index


def _vals():
    # idx2 is the infeasible placeholder (-1e9 sentinel)
    return pd.Series([0.5, -0.8, -1e9, 0.2, 1.7], index=[0, 1, 2, 3, 4])


def test_minimize_skips_sentinel():
    # smallest FEASIBLE is -0.8 (idx1), NOT the -1e9 placeholder (idx2)
    assert list(_select_best_index(_vals(), 1, minimize=True)) == [1]


def test_maximize_returns_largest_feasible():
    assert list(_select_best_index(_vals(), 1, minimize=False)) == [4]  # 1.7


def test_sentinel_never_in_results_even_when_asking_for_many():
    idx = list(_select_best_index(_vals(), 10, minimize=True))
    assert 2 not in idx           # placeholder excluded
    assert set(idx) == {0, 1, 3, 4}


def test_all_infeasible_returns_empty():
    s = pd.Series([-1e9, 1e9, -2e9], index=[0, 1, 2])
    assert len(_select_best_index(s, 3, minimize=True)) == 0


def test_non_numeric_coerced_and_dropped():
    s = pd.Series([0.3, "n/a", -0.1], index=[0, 1, 2])
    assert list(_select_best_index(s, 1, minimize=True)) == [2]  # -0.1
