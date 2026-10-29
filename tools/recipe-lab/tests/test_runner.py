"""Runner unit tests (pure functions; stack tests run as integration)."""

from __future__ import annotations

import pytest

from recipe_lab.contract import ContractError
from recipe_lab.runner import attribution_delta, replay_metric_value


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


def test_replay_metric_value_rounds_and_rejects_unknown():
    # A1: the contracted replay metric maps to the runner's summary key
    assert replay_metric_value({"accuracy": 0.8333333333}, "routed_accuracy") == 0.833333
    assert replay_metric_value({"accuracy": None}, "routed_accuracy") is None
    assert replay_metric_value({}, "routed_accuracy") is None
    with pytest.raises(ContractError, match="unknown replay metric"):
        replay_metric_value({"accuracy": 1.0}, "capability.score")
