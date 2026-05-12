from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


starter_code_path = Path(__file__).parent.parent / "starter_code"
if str(starter_code_path) not in sys.path:
    sys.path.insert(0, str(starter_code_path))


def test_expected_loss_formula() -> None:
    try:
        import solution
    except ImportError:
        pytest.skip("solution.py not found")

    if not hasattr(solution, "expected_loss"):
        pytest.skip("expected_loss function not implemented")

    samples_a = np.array([0.10, 0.15, 0.20, 0.30])
    samples_b = np.array([0.12, 0.10, 0.18, 0.22])
    expected = np.mean(np.maximum(samples_a - samples_b, 0.0))

    assert solution.expected_loss(samples_a, samples_b) == pytest.approx(expected)


def test_decision_threshold_moves_with_costs() -> None:
    try:
        import solution
    except ImportError:
        pytest.skip("solution.py not found")

    if not hasattr(solution, "decision_threshold"):
        pytest.skip("decision_threshold function not implemented")

    conservative = solution.decision_threshold(false_positive_cost=9, false_negative_cost=1)
    aggressive = solution.decision_threshold(false_positive_cost=1, false_negative_cost=9)

    assert conservative > aggressive
    assert conservative == pytest.approx(0.9)
    assert aggressive == pytest.approx(0.1)
