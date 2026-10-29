"""Runner unit tests (pure functions; stack tests run as integration)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from recipe_lab.contract import ContractError, OraclePolicy
from recipe_lab.runner import (
    RunnerEnv,
    RunnerTimeout,
    attribution_delta,
    evaluate_candidate,
    replay_metric_value,
)


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


# ---- A5: per-trial budget ----


def test_evaluate_candidate_trial_timeout(tmp_path):
    """A stack that never becomes ready must surface as RunnerTimeout within
    the budget deadline, not as a generic not-ready failure."""
    router = Path(tmp_path) / "router.sh"
    router.write_text("#!/bin/sh\nsleep 60\n")
    router.chmod(0o755)
    mock = Path(tmp_path) / "mock.py"
    mock.write_text("import time\nmock_port = 0\ntime.sleep(60)\n")
    cfg = Path(tmp_path) / "candidate.yaml"
    cfg.write_text("version: v0.2\n")
    env = RunnerEnv(
        router_bin=str(router),
        mock_backend_script=str(mock),
        python_bin=sys.executable,
        work_dir=Path(tmp_path) / "work",
        ready_timeout_s=30,
    )
    with pytest.raises(RunnerTimeout, match="trial exceeded"):
        evaluate_candidate(
            candidate_config=cfg,
            cases=[{"case_id": "c1", "prompt": "p", "truth": "math"}],
            oracle_policy=OraclePolicy.from_dict({"math": "local/model-math"}),
            bank={},
            env=env,
            trial_timeout_s=2.0,
        )
