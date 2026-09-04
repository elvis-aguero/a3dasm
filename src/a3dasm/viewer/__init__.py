"""Public entry point for the read-only live run viewer.

    python -m a3dasm.viewer <study-dir>

Requires the ``viewer`` optional-dependency group
(``pip install a3dasm[viewer]``). The real implementation lives in
``a3dasm._src.viewer`` — this top-level package exists only so
``python -m a3dasm.viewer`` resolves, mirroring ``a3dasm/__main__.py``'s
own thin-forwarding convention for ``python -m a3dasm``.

Deliberately NO eager import here (not even of ``create_app``/``run_viewer``):
Python always executes a package's ``__init__.py`` before any of its
submodules, so an eager Starlette import here would run BEFORE
``__main__.py``'s own try/except ever gets a chance to turn a missing
``viewer`` extra into a friendly message — it would surface as a raw
``ModuleNotFoundError`` instead. Import ``a3dasm._src.viewer.app`` directly
for programmatic access.
"""
from __future__ import annotations
