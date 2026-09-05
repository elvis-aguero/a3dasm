"""A strategizer -> critic delegation row must carry the critic's real cost.

Both critic call sites (the ``Done()`` GATE acceptance check and
``AskForFeedback``) write a delegation row, and both used to hardcode
``tokens_in=0, tokens_out=0, cost_usd=None`` on it — while the call they
represent had just spent real money, recorded only in telemetry under an
internal id (``critic-1``) that the delegation log never sees.

The consequence is a ledger that reads as complete but is not: anything
summing ``cost_usd`` across ``delegation_log.jsonl`` omits the critic's
entire budget. Measured on run 20260905T162758 — $0.697 summed from
delegations against a true $1.078, a 35% undercount, with the whole
difference being the critic.

``_invoke_critic`` now publishes the usage it already records to telemetry
as ``_last_critic_usage`` for its caller to put on the row.
"""

from __future__ import annotations

from a3dasm._src.nodes.critic_gate import CriticGateMixin


class _Worker:
    """Adapter stub whose ``invoke`` reports usage the way a real one does."""

    def __init__(self, usage, text="### Verdict\nPASS\n"):
        self.last_usage = usage
        self.model = "claude-haiku-4-5-20251001"
        self._text = text

    def invoke(self, _messages, **_kw):
        return self._text


class _Failing(_Worker):
    def invoke(self, _messages, **_kw):
        raise RuntimeError("stream died")


class _Node(CriticGateMixin):
    """Minimal carrier for the attributes ``_invoke_critic`` touches."""

    def __init__(self, worker):
        self._spec = None
        self._outgoing = ("critic",)
        self._worker_adapters = {"critic": worker}
        self._current_notes_dir = None
        self._critic_calls = 0
        self._study_dir = None
        self._name = "strategizer"
        self.recorded_usage = []

    # CriticGateMixin relies on these via MRO in the real class.
    def _find_critic_name(self):
        return "critic"

    def _role_of(self, _name):
        return "critic"

    def _record_usage(self, usage, **kw):
        self.recorded_usage.append((usage, kw))

    def _persist_critic_review(self, _n, _text):
        pass

    def _record_retrospective(self, *_a, **_kw):
        pass

    def _prior_reviews_digest(self):
        return ""


def test_invoke_critic_publishes_its_usage_for_the_delegation_row():
    usage = {"input_tokens": 120, "output_tokens": 4300,
             "total_cost_usd": 0.2205916}
    node = _Node(_Worker(usage))

    node._invoke_critic("audit this")

    # Same numbers that go to telemetry are available to the caller, so the
    # delegation row and the telemetry ledger cannot disagree.
    assert node._last_critic_usage == usage
    assert node.recorded_usage[0][0] == usage
    assert node.recorded_usage[0][1]["phase"] == "critic_review"


def test_a_failed_critic_call_does_not_inherit_the_previous_call_s_cost():
    """The riskiest way to get this wrong is to bill one call for another.

    ``worker.last_usage`` persists on the adapter, so a second invoke that
    raises would otherwise leave the FIRST call's usage in place and log it
    against a delegation that produced nothing.
    """
    first = {"input_tokens": 120, "output_tokens": 4300,
             "total_cost_usd": 0.22}
    node = _Node(_Worker(first))
    node._invoke_critic("first audit")
    assert node._last_critic_usage == first

    node._worker_adapters["critic"] = _Failing(first)
    text = node._invoke_critic("second audit, this one fails")

    assert text.startswith("ERROR:")
    assert node._last_critic_usage == {}


def test_gate_and_feedback_rows_read_usage_rather_than_hardcoding_zero():
    """Guards the call sites, which are what actually write the row.

    A regression here is invisible at runtime — the row still appears, just
    with no accounting — so it is pinned against the source.
    """
    from pathlib import Path

    src = (Path(__file__).parent.parent / "src" / "a3dasm" / "_src"
           / "nodes" / "tools" / "routing.py").read_text(encoding="utf-8")

    for marker in ("_critic_usage", "_fb_usage"):
        assert f'{marker}.get("total_cost_usd")' in src, (
            f"{marker} no longer carries cost onto its delegation row")
        assert f'{marker}.get("output_tokens", 0)' in src

    # The two rows that used to be hardcoded are the ONLY delegation records
    # in this module; neither may go back to a literal cost of None.
    assert src.count("cost_usd=None") == 0
