"""a3dasm agentic implementation package.

Importing any submodule installs the f3dasm compatibility shim (see
``_f3dasm_compat``), which re-homes the two ``ExperimentData`` behaviours a3dasm
needs onto whatever f3dasm is installed. Idempotent, and never allowed to break
import.
"""
from __future__ import annotations

try:
    from ._f3dasm_compat import apply_f3dasm_compat

    apply_f3dasm_compat()
except Exception:  # noqa: BLE001 — a compat hiccup must not break import
    pass
