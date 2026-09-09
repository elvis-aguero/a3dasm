"""Runtime-injected text is marked as such, and the viewer renders it apart.

Nudges, science-monitor drift, budget warnings and operator notes all reach an
agent prepended to the text of its next tool result. On disk and in the viewer
that made them indistinguishable from the tool's own output.

Marking happens at the injection sites, not in the reader, because no
reader-side rule can do it: most notices carry a ``[TAG …]`` head but the
GetStatus poll hints are bare prose, and tools emit bracketed lines of their
own (``[exited 1]``, ``[output truncated to last …]``).
"""
from __future__ import annotations

from a3dasm._src.nodes.notices import (
    NOTICE_CLOSE,
    NOTICE_OPEN,
    split_notices,
    wrap_notice,
)


# --------------------------------------------------------------- the marker
def test_wrap_marks_text_and_split_recovers_it():
    out = wrap_notice("[EVAL BUDGET 80%] 160 of 200 evals used.")
    assert NOTICE_OPEN in out and NOTICE_CLOSE in out
    notices, rest = split_notices(out + "Recorded 12 evaluations.")
    assert notices == ["[EVAL BUDGET 80%] 160 of 200 evals used."]
    assert rest == "Recorded 12 evaluations."


def test_empty_input_yields_no_marker():
    """Call sites keep their 'no notice, no prefix' behaviour: a bare marker
    would be noise in the prompt and an empty band in the viewer."""
    assert wrap_notice("") == ""
    assert wrap_notice("   \n  ") == ""


def test_split_leaves_untouched_output_alone():
    plain = "PASSED 12 tests\n[exited 0]"
    assert split_notices(plain) == ([], plain)


def test_multiple_notices_are_kept_in_order():
    text = (wrap_notice("[NUDGE] first")
            + wrap_notice("[SCIENCE MONITOR — DRIFT] second")
            + "tool output")
    notices, rest = split_notices(text)
    assert notices == ["[NUDGE] first", "[SCIENCE MONITOR — DRIFT] second"]
    assert rest == "tool output"


def test_bracketed_tool_output_is_not_mistaken_for_a_notice():
    """The reason marking is done at the source: these are real tool output
    and any bracket-matching heuristic would mislabel them."""
    for line in ("[output truncated to last 200 lines]", "[exited 1]",
                 "[killed after 600s]", "[scratch exit 2]"):
        assert split_notices(line) == ([], line)


def test_multiline_notice_body_survives_round_trip():
    body = "NOTE: you polled D001 only 12s ago.\nDo other work in the meantime."
    notices, rest = split_notices(wrap_notice(body) + "Working")
    assert notices == [body]
    assert rest == "Working"


# --------------------------------------------------------------- the viewer
def _render(result_text: str) -> str:
    from a3dasm._src.viewer.app import _tool_result_html
    return _tool_result_html(
        {"results": [{"tool_use_id": "t1", "content": result_text}]},
        ["Bash"],
    )


def test_viewer_renders_a_notice_outside_the_result_block():
    html = _render(wrap_notice("[NUDGE] stop polling") + "PASSED 3 tests")
    assert "notice-pre" in html, html
    assert "stop polling" in html
    assert "PASSED 3 tests" in html
    # The notice must not sit inside the tool-result block, or it reads as
    # the tool's own output — the whole point of the split.
    assert html.index("notice-pre") < html.index("tool-result"), html
    # And the marker itself is never shown to the reader.
    assert NOTICE_OPEN not in html and NOTICE_CLOSE not in html, html


def test_viewer_result_without_a_notice_is_unchanged_in_shape():
    html = _render("PASSED 3 tests")
    assert "notice" not in html, html
    assert "result-pre" in html and "PASSED 3 tests" in html


def test_viewer_notice_only_result_still_renders_the_notice():
    """A tool whose entire output was consumed by the prefix (e.g. an
    operator note landing on a call that returned nothing) must not lose it."""
    html = _render(wrap_notice("[OPERATOR NOTE — from the viewer] check units"))
    assert "check units" in html, html
    assert "notice-pre" in html


def test_viewer_still_flags_an_error_result_carrying_a_notice():
    """ERROR_RETURN belongs to the tool's output, so it must be detected
    after the notice is lifted out. Testing the raw text meant any result
    carrying a prefix lost its error styling."""
    html = _render(wrap_notice("[NUDGE] mind the budget") + "ERROR: boom")
    assert "is-error" in html, html
    assert "notice-pre" in html and "mind the budget" in html


# ------------------------------------------------------- at the injection site
def _node(run_dir):
    import threading

    from a3dasm._src.backends.base import Agent, Edge, Graph
    from a3dasm._src.delegation_log import DelegationLog
    from a3dasm._src.nodes import StrategizerNode

    class _Stub:
        def __init__(self):
            self.closure_tools: dict = {}
            self.last_usage: dict = {}
            self.model = "m"

        def invoke(self, messages):
            return ""

    class S(Agent):
        role = "strategizer"
        # GetStatus is agent-declared (routing.py), so it must be opted into
        # here or the closure is never built.
        tools = frozenset({"Done", "Wait", "GetStatus"})
        description = "s"

    class W(Agent):
        description = "w"

    spec = Graph(nodes={"strategizer": S(), "worker": W()},
                 edges=(Edge("strategizer", "worker"),), entry="strategizer")
    notes = run_dir / "debug" / "strategizer_notes"
    notes.mkdir(parents=True)
    n = StrategizerNode(
        _Stub(), name="strategizer", outgoing=["worker"], spec=spec,
        worker_adapters={"worker": _Stub()}, notes_dir=notes,
        delegation_log=DelegationLog(run_dir / "debug" / "delegation_log.jsonl"),
    )
    assert isinstance(n._pending_worker_msgs_lock, threading.Lock().__class__)
    return n


def test_queued_worker_message_reaches_getstatus_marked(tmp_path):
    """A budget warning queued for a running delegation is a3dasm speaking,
    so it must arrive marked rather than looking like the worker's report."""
    node = _node(tmp_path / "runs" / "T1")
    with node._registry_lock:
        node._registry["D001"] = {
            "status": "Working", "result": None,
            "start_time": __import__("time").monotonic(), "getstatus_count": 0}
    with node._pending_worker_msgs_lock:
        node._pending_worker_msgs["D001"] = ["[EVAL BUDGET 90%] slow down"]

    out = node._build_routing_closures()["GetStatus"]("D001")

    assert "[EVAL BUDGET 90%] slow down" in out, out
    notices, _ = split_notices(out)
    assert any("EVAL BUDGET" in n for n in notices), (
        f"budget warning arrived unmarked: {out!r}")


def test_drain_notifications_marks_what_it_returns(tmp_path):
    node = _node(tmp_path / "runs" / "T2")
    with node._notifications_lock:
        node._notifications.append("[Delegation D003 finished]")
    out = node._drain_notifications()
    notices, rest = split_notices(out)
    assert notices == ["[Delegation D003 finished]"], out
    assert rest == ""


def test_drain_notifications_is_empty_when_nothing_pending(tmp_path):
    """No notifications must still mean no prefix at all — not a bare
    marker prepended to every tool result in the run."""
    node = _node(tmp_path / "runs" / "T3")
    assert node._drain_notifications() == ""
