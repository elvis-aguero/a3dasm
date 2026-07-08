"""Non-Claude (openai_compatible) Bash surface: Bash + BashOutput + KillShell.

The framework Bash must match the SDK's shape: a command past its timeout is
BACKGROUNDED (not killed) and returned with a bash_id the agent polls via
BashOutput / stops via KillShell. This is the parity upgrade that replaces the
old 120s-hardcap `subprocess.run` stopgap and the WaitForProcess stand-in.
"""
from __future__ import annotations

import re
import sys

from a3dasm._src.backends.openai_compatible import (
    _BASH_INLINE_CAP,
    _native_tool_map,
)


def _tools(tmp_path, nudge=None):
    nm = _native_tool_map(tmp_path, nudge)
    return nm["Bash"].func, nm["BashOutput"].func, nm["KillShell"].func


def _bash_id(text):
    m = re.search(r"bash_id:\s*(bash_\d+)", text)
    return m.group(1) if m else None


def test_foreground_success_is_plain_output(tmp_path):
    bash, _, _ = _tools(tmp_path)
    out = bash(command="echo hello")
    assert "hello" in out
    assert "[exit" not in out          # clean success → no trailer
    assert "bash_id" not in out


def test_nonzero_exit_is_reported(tmp_path):
    bash, _, _ = _tools(tmp_path)
    out = bash(command="sh -c 'exit 3'")
    assert "[exit 3]" in out


def test_timeout_backgrounds_not_kills(tmp_path):
    bash, bash_output, kill = _tools(tmp_path)
    out = bash(command="sleep 5", timeout=1000)   # floors to ~1s, then backgrounds
    assert "interrupted" in out.lower()
    bid = _bash_id(out)
    assert bid, out
    # It is NOT killed — still running.
    poll = bash_output(bid)
    assert "still running" in poll.lower()
    kill(bid)                                       # cleanup


def test_run_in_background_returns_immediately(tmp_path):
    bash, bash_output, kill = _tools(tmp_path)
    out = bash(command="sleep 2", run_in_background=True)
    assert "background" in out.lower()
    bid = _bash_id(out)
    assert bid, out
    kill(bid)


def test_bashoutput_reports_exit_and_output(tmp_path):
    import time
    bash, bash_output, _ = _tools(tmp_path)
    out = bash(command="echo done; sleep 0.2", run_in_background=True)
    bid = _bash_id(out)
    time.sleep(1.0)
    poll = bash_output(bid)
    assert "exited 0" in poll.lower()
    assert "done" in poll


def test_bashoutput_unknown_id_errors(tmp_path):
    _, bash_output, _ = _tools(tmp_path)
    assert bash_output("bash_999").startswith("ERROR")


def test_killshell_terminates(tmp_path):
    bash, bash_output, kill = _tools(tmp_path)
    out = bash(command="sleep 30", run_in_background=True)
    bid = _bash_id(out)
    res = kill(bid)
    assert "killed" in res.lower()
    # id is now gone
    assert bash_output(bid).startswith("ERROR")


def test_large_output_spills_to_file(tmp_path):
    bash, _, _ = _tools(tmp_path)
    n = _BASH_INLINE_CAP + 5000
    out = bash(command=f"{sys.executable} -c \"print('x'*{n})\"")
    assert "truncated" in out.lower()
    assert "full output at" in out.lower()


def test_nudge_is_appended(tmp_path):
    def nudge(name, inp):
        return "NUDGE: use get_evaluator()" if name == "Bash" else ""
    bash, _, _ = _tools(tmp_path, nudge)
    out = bash(command="echo hi")
    assert "NUDGE: use get_evaluator()" in out
