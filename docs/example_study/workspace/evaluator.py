"""Canonical minimal oracle for the study-folder contract.

Called one kwarg per input (the names come from the run's Domain); returns the
single output named in config.yaml's ``evaluator.output_names`` (here ``y``).
Executed by tests/agentic/test_study_contract.py — keep it runnable.
"""


def evaluate(x1: float, x2: float) -> float:
    return (x1 - 1.0) ** 2 + (x2 + 2.0) ** 2
