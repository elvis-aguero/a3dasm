"""Provenance marking for text a3dasm injects into an agent's tool results.

Nudges, science-monitor drift, budget warnings, operator notes and Confer
messages all reach an agent the same way: prepended to the text of whatever
tool it called next. On disk and in the viewer that made them
indistinguishable from the tool's own output — a reader could not tell
"the shell returned this" from "a3dasm said this to the agent", and neither
could the agent.

Marking has to happen HERE, at the injection sites, not in the viewer:

* the bracket convention is incomplete — most notices carry a ``[TAG …]``
  head (``[SCIENCE MONITOR — …]``, ``[EVAL BUDGET …]``, ``[NUDGE]``) but the
  GetStatus poll-escalation hints are bare prose (``NOTE: you polled …``);
* brackets are not a safe signal anyway — tools emit their own
  (``[output truncated to last …]``, ``[exited 1]``, ``[killed …]``), so a
  reader-side regex mislabels real tool output as an injection. The same
  class of heuristic was measured on this transcript corpus for error
  detection and matched 3 of 78 results, all three false.

The marker is a tag rather than a control character because this text is
part of the agent's prompt: it has to stay readable, and it matches the
``<role>``/``<tools>`` idiom the prompt corpus already uses.
"""
from __future__ import annotations

import re

#: Opening/closing marker around every runtime-injected notice block.
NOTICE_OPEN = "<a3dasm-note>"
NOTICE_CLOSE = "</a3dasm-note>"

#: Matches one complete notice block, capturing its inner text. Used by the
#: viewer to lift notices out of a tool result's body.
NOTICE_RE = re.compile(
    re.escape(NOTICE_OPEN) + r"\n?(.*?)\n?" + re.escape(NOTICE_CLOSE),
    re.DOTALL,
)


def wrap_notice(text: str, *, trailing: str = "\n\n") -> str:
    """Mark *text* as runtime-injected, or return "" when there is nothing.

    Empty and whitespace-only input yields "" so call sites keep their
    existing "no notice, no prefix" behaviour — a bare marker carrying no
    content would be noise in the prompt and an empty band in the viewer.
    """
    if not text or not text.strip():
        return ""
    return f"{NOTICE_OPEN}\n{text.strip()}\n{NOTICE_CLOSE}{trailing}"


def split_notices(text: str) -> tuple[list[str], str]:
    """Split *text* into (notice bodies, the remaining tool output).

    The counterpart to :func:`wrap_notice`, for renderers. Returns the
    notices in order of appearance and the text with them removed.
    """
    notices = [m.group(1).strip() for m in NOTICE_RE.finditer(text)]
    return notices, NOTICE_RE.sub("", text).strip()
