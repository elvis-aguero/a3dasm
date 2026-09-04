"""Read-only live web viewer for a3dasm agentic runs.

Optional feature — requires the ``viewer`` extra (``pip install
a3dasm[viewer]``). Deliberately NO eager imports here: ``readers.py`` (pure
filesystem functions, no HTTP) must stay importable/testable without
Starlette/uvicorn/Jinja2 installed, and importing any submodule of a package
always executes this file first — an eager `from .app import ...` here would
force the Starlette dependency onto every ``viewer.readers`` import too.
Callers that need the app/server import ``a3dasm._src.viewer.app`` directly
(``agent_runtime.py``'s ``serve_viewer()``, ``viewer/__main__.py``).
"""
from __future__ import annotations
