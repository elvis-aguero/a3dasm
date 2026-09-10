"""A worker sees the criterion its evidence will be judged against.

A hypothesis's falsification_criterion is immutable once registered and is the
standard its verdict is judged by, but the only party a3dasm showed it to was
the delegator, and only at reconciliation time (the falsification checkpoint
fires on a Done report). The worker that produces the evidence never saw it —
Delegate put context packaging on the delegator, so the criterion reached the
worker only if the delegator remembered to paste it.

Measured cost across 52 cluster runs: INCONCLUSIVE is the largest verdict
class (100 of 295 hypotheses) and the most expensive (median lifetime 3.15h vs
1.49h FALSIFIED, 0.91h SUPPORTED), and its comments name the mechanism — "the
registered H3 falsification criterion required a 50-iter constrained BO in the
high-Ixx region. This BO was never executed"; "Test is INADEQUATE relative to
the registered 30-point LHS criterion".
"""
from __future__ import annotations

from pathlib import Path

from a3dasm._src.backends.base import Agent, Edge, Graph
from a3dasm._src.delegation_log import DelegationLog
from a3dasm._src.nodes import StrategizerNode

CRITERION = (
    "A 50-iteration constrained BO campaign in the high-Ixx region returns a "
    "feasible design with sigma_peak >= 1.122 kPa."
)
PREDICTION = "Best-found sigma_peak stays below 0.65 kPa across 50 iterations."
STATEMENT = "The bending-regime ceiling caps sigma_peak near 0.65 kPa."


class _Stub:
    def __init__(self) -> None:
        self.closure_tools: dict = {}
        self.last_usage: dict = {}
        self.model = "m"
        self.seen: list[str] = []

    def copy(self):
        c = _Stub()
        c.seen = self.seen          # share, so the test can read what was sent
        return c

    def invoke(self, messages):
        for m in messages:
            content = m.get("content") if isinstance(m, dict) else None
            if isinstance(content, str):
                self.seen.append(content)
        return "## Report\n\n### Conclusions\ndone\n"


def _node(run_dir: Path) -> StrategizerNode:
    class S(Agent):
        role = "strategizer"
        tools = frozenset({"Done", "Wait", "GetStatus"})
        description = "s"

    class W(Agent):
        description = "w"

    spec = Graph(
        nodes={"strategizer": S(), "implementer": W()},
        edges=(Edge("strategizer", "implementer"),),
        entry="strategizer",
    )
    notes = run_dir / "debug" / "strategizer_notes"
    notes.mkdir(parents=True)
    return StrategizerNode(
        _Stub(), name="strategizer", outgoing=["implementer"], spec=spec,
        worker_adapters={"implementer": _Stub()}, notes_dir=notes,
        delegation_log=DelegationLog(run_dir / "debug" / "delegation_log.jsonl"),
        study_dir=run_dir,
    )


def _register(node) -> str:
    """Returns the id the ledger assigned (propose() mints it, not us)."""
    hid = node._ledger.propose(
        statement=STATEMENT, falsification_criterion=CRITERION,
        prediction=PREDICTION, prior=0.5, proposed_by="strategizer",
    )
    assert not str(hid).startswith("ERROR"), hid
    return hid


def test_worker_task_carries_the_registered_criterion_verbatim(tmp_path):
    node = _node(tmp_path / "runs" / "T1")
    hid = _register(node)
    worker = node._worker_adapters["implementer"]

    out = node._build_routing_closures()["Delegate"](
        target="implementer",
        intent="Search the high-Ixx region.",
        expected_report="best sigma_peak",
        hypothesis_ids=[hid],
        is_falsification_attempt=True,
        wait=True,
    )
    assert out, out
    sent = "\n".join(worker.seen)
    # The contract goes through untruncated — a clipped criterion would
    # reintroduce the mismatch this exists to prevent.
    assert CRITERION in sent, sent[:2000]
    assert PREDICTION in sent, sent[:2000]
    assert hid in sent


def test_falsification_framing_tells_the_worker_the_test_is_fixed(tmp_path):
    node = _node(tmp_path / "runs" / "T2")
    hid = _register(node)
    worker = node._worker_adapters["implementer"]
    node._build_routing_closures()["Delegate"](
        target="implementer", intent="x", expected_report="y",
        hypothesis_ids=[hid], is_falsification_attempt=True, wait=True,
    )
    sent = "\n".join(worker.seen)
    assert "FALSIFICATION ATTEMPT" in sent, sent[:1500]
    # It must also license reporting a mismatch rather than quietly
    # substituting a different test.
    assert "cannot" in sent.lower() and "mismatch" in sent.lower(), sent[:1500]


def test_non_falsification_delegation_gets_context_without_the_verdict_framing(
        tmp_path):
    node = _node(tmp_path / "runs" / "T3")
    hid = _register(node)
    worker = node._worker_adapters["implementer"]
    node._build_routing_closures()["Delegate"](
        target="implementer", intent="characterise the region",
        expected_report="a map", hypothesis_ids=[hid],
        is_falsification_attempt=False, wait=True,
    )
    sent = "\n".join(worker.seen)
    assert CRITERION in sent, sent[:1500]
    assert "FALSIFICATION ATTEMPT" not in sent, sent[:1500]


def test_no_hypothesis_ids_means_no_injected_block(tmp_path):
    """Exploration with nothing filed against it must not gain a stray
    empty block."""
    node = _node(tmp_path / "runs" / "T4")
    _register(node)
    worker = node._worker_adapters["implementer"]
    node._build_routing_closures()["Delegate"](
        target="implementer", intent="explore", expected_report="notes",
        hypothesis_ids=[], wait=True,
    )
    sent = "\n".join(worker.seen)
    assert "registered_hypothesis" not in sent, sent[:1200]


def test_an_unknown_hypothesis_id_is_refused_before_any_work_is_dispatched(
        tmp_path):
    """Delegate already validates hypothesis_ids, so the injected brief can
    never be built from a dangling reference — the delegation is refused
    outright instead. Pinned here because that guard is what makes the
    brief's own missing-entry branch unreachable in practice."""
    node = _node(tmp_path / "runs" / "T5")
    hid = _register(node)
    worker = node._worker_adapters["implementer"]
    out = node._build_routing_closures()["Delegate"](
        target="implementer", intent="x", expected_report="y",
        hypothesis_ids=[hid, "H404"], is_falsification_attempt=True,
        wait=True,
    )
    assert out.startswith("ERROR"), out[:300]
    assert "H404" in out
    assert worker.seen == [], "work was dispatched despite a bad id"


def test_a_long_statement_is_capped_but_the_criterion_is_not(tmp_path):
    node = _node(tmp_path / "runs" / "T6")
    long_stmt = "S" * 3000
    # propose() returns the id, possibly followed by a length NUDGE — the id
    # is the first token, not the whole string.
    hid = node._ledger.propose(
        statement=long_stmt, falsification_criterion=CRITERION,
        prediction=PREDICTION, prior=0.5, proposed_by="strategizer",
    ).split()[0]
    worker = node._worker_adapters["implementer"]
    node._build_routing_closures()["Delegate"](
        target="implementer", intent="x", expected_report="y",
        hypothesis_ids=[hid], is_falsification_attempt=True, wait=True,
    )
    sent = "\n".join(worker.seen)
    assert long_stmt not in sent, "statement was not capped"
    assert "[…]" in sent
    assert CRITERION in sent, "the criterion must never be truncated"
