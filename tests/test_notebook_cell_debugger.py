"""#13 per-cell notebook debugger: diagnose_notebook returns a per-cell trace so
the agent can see WHICH cell breaks reproduction (run 20260623T002417 burned ~10
blind gate attempts because CheckDeliverable is binary pass/fail)."""
from __future__ import annotations

import pytest

nbformat = pytest.importorskip("nbformat")
pytest.importorskip("nbclient")

from a3dasm._src.notebook_exec import diagnose_notebook


def _nb(tmp_path, cells):
    nb = nbformat.v4.new_notebook()
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3",
                                 "language": "python"}
    for src, name in cells:
        c = nbformat.v4.new_code_cell(src)
        c.metadata["name"] = name
        nb.cells.append(c)
    p = tmp_path / "pipeline.ipynb"
    nbformat.write(nb, str(p))
    return p


def test_pinpoints_the_broken_cell_by_name_and_traceback(tmp_path):
    p = _nb(tmp_path, [
        ("print('doe ok')", "doe"),
        ("raise ValueError('ledger path missing')", "ml"),
        ("print('analysis ok')", "analysis"),
    ])
    t = diagnose_notebook(p, cwd=tmp_path, env=None, timeout=60)
    assert t["executed"] == 3 and not t["timed_out"]
    fe = t["first_error"]
    assert fe is not None and fe["name"] == "ml"
    assert "ValueError" in fe["error"] and "ledger path missing" in fe["error"]
    # allow_errors=True → later cells still run; only 'ml' is flagged
    by = {c["name"]: c for c in t["cells"]}
    assert by["doe"]["errored"] is False and "doe ok" in by["doe"]["stdout"]
    assert by["ml"]["errored"] is True
    assert by["analysis"]["errored"] is False


def test_clean_notebook_has_no_first_error(tmp_path):
    p = _nb(tmp_path, [("print('a')", "doe"), ("x = 1 + 1", "ml")])
    t = diagnose_notebook(p, cwd=tmp_path, env=None, timeout=60)
    assert t["first_error"] is None
    assert all(not c["errored"] for c in t["cells"])


def test_upto_name_truncates_and_does_not_reach_later_failure(tmp_path):
    p = _nb(tmp_path, [
        ("print('doe')", "doe"),
        ("print('ml')", "ml"),
        ("raise RuntimeError('boom')", "analysis"),  # must NOT be reached
    ])
    t = diagnose_notebook(p, cwd=tmp_path, env=None, timeout=60, upto_name="ml")
    assert t["truncated"] is True
    assert t["executed"] == 2                  # only doe + ml ran
    assert t["first_error"] is None            # the broken 'analysis' was not reached
    assert {c["name"] for c in t["cells"]} == {"doe", "ml"}


def test_unknown_cell_name_is_flagged_not_executed(tmp_path):
    p = _nb(tmp_path, [("print('doe')", "doe")])
    t = diagnose_notebook(p, cwd=tmp_path, env=None, timeout=60, upto_name="nope")
    assert t["missing_name"] is True and t["executed"] == 0


def test_runpipelinecell_is_wired_into_the_strategizer():
    from a3dasm._src.agents.strategizer import StrategizerAgent
    assert "RunPipelineCell" in StrategizerAgent.tools
