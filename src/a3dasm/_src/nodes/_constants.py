"""Shared run-knob accessors for the nodes package — kept here (not in
strategizer.py) so strategizer.py, lifecycle.py and routing.py can import them
without an import cycle.

Read at CALL time (not import) so they reflect config.yaml, which AgenticRun
installs after these modules import. config.yaml's runtime block is the source
of truth; F3DASM_RUN_BACKSTOP_MULTIPLE overrides."""
from ..settings import get_float


def run_backstop_multiple() -> float:
    """Run-level cost backstop multiple: a run halts once it exceeds this ×
    the (soft) time budget, bounding runaway cost. Wall-clock is a poor cost
    proxy when one oracle eval can take days, so the cap is configurable and
    DISABLED when <= 0 (knob: run_backstop_multiple, default 2.0)."""
    return get_float("run_backstop_multiple", 2.0)


def backstop_enabled() -> bool:
    """True when the run-level time backstop is active (multiple > 0)."""
    return run_backstop_multiple() > 0
