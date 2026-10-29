"""Runner unit tests (pure functions; stack tests run as integration)."""

from __future__ import annotations

from recipe_lab.runner import attribution_delta, candidate_accuracy


def test_attribution_delta_counts_fixed_errors():
    baseline = {"class_counts": {"E1_signal": 3, "E2_policy": 2, "E3_capability": 5, "decision_mismatch": 4}}
    candidate = {"class_counts": {"E1_signal": 1, "E2_policy": 2, "E3_capability": 6, "decision_mismatch": 1}}
    assert attribution_delta(baseline, candidate) == {
        "e1_fixed": 2,
        "e2_fixed": 0,
        "decision_mismatch_fixed": 3,
        "e3_remaining": 6,
    }


def test_attribution_delta_never_negative():
    baseline = {"class_counts": {"E1_signal": 1}}
    candidate = {"class_counts": {"E1_signal": 4}}
    assert attribution_delta(baseline, candidate) == {
        "e1_fixed": 0,
        "e2_fixed": 0,
        "decision_mismatch_fixed": 0,
        "e3_remaining": 0,
    }


def test_candidate_accuracy_rounds():
    assert candidate_accuracy({"accuracy": 0.8333333333}, "capability.score") == 0.833333
    assert candidate_accuracy({"accuracy": None}, "capability.score") is None
