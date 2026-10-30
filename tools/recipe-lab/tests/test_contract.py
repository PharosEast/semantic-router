"""Contract tests: eager validation with actionable errors."""

from __future__ import annotations

import copy

import pytest
from recipe_lab.contract import ContractError, ExperimentSpec, OraclePolicy

from tests.fixtures import POLICY, VALID_SPEC


def test_valid_spec_roundtrip():
    spec = ExperimentSpec.from_dict(copy.deepcopy(VALID_SPEC))
    assert spec.objective.metric_key == "capability.score"
    assert spec.objective.direction == "maximize"
    assert spec.workload.profile == "quick"
    assert spec.oracle_policy.expected("math") == "local/model-math"
    assert len(spec.gates) == 1
    assert spec.budget.rounds == 5
    assert spec.stack is not None and spec.stack.mode == "external"
    # to_dict -> from_dict must round-trip
    spec2 = ExperimentSpec.from_dict(spec.to_dict())
    assert spec2.to_dict() == spec.to_dict()


def test_missing_api_version_rejected():
    raw = copy.deepcopy(VALID_SPEC)
    raw["api_version"] = "recipe-lab/v1beta9"
    with pytest.raises(ContractError, match="api_version"):
        ExperimentSpec.from_dict(raw)


def test_bad_direction_rejected():
    raw = copy.deepcopy(VALID_SPEC)
    raw["objective"]["direction"] = "sideways"
    with pytest.raises(ContractError, match="direction"):
        ExperimentSpec.from_dict(raw)


def test_empty_metric_key_rejected():
    raw = copy.deepcopy(VALID_SPEC)
    raw["objective"]["metric_key"] = "  "
    with pytest.raises(ContractError, match="metric_key"):
        ExperimentSpec.from_dict(raw)


def test_missing_content_hash_rejected():
    raw = copy.deepcopy(VALID_SPEC)
    del raw["workload"]["content_hash"]
    with pytest.raises(ContractError, match="content_hash"):
        ExperimentSpec.from_dict(raw)


def test_negative_min_improvement_rejected():
    raw = copy.deepcopy(VALID_SPEC)
    raw["objective"]["min_improvement_pct"] = -0.5
    with pytest.raises(ContractError, match="min_improvement_pct"):
        ExperimentSpec.from_dict(raw)


def test_empty_oracle_policy_rejected():
    raw = copy.deepcopy(VALID_SPEC)
    raw["attribution"]["oracle_policy"] = {}
    with pytest.raises(ContractError, match="oracle_policy"):
        ExperimentSpec.from_dict(raw)


def test_bad_gate_floor_rejected():
    raw = copy.deepcopy(VALID_SPEC)
    raw["gates"][0]["floor"] = "kinda-baseline"
    with pytest.raises(ContractError, match="floor"):
        ExperimentSpec.from_dict(raw)


def test_zero_rounds_rejected():
    raw = copy.deepcopy(VALID_SPEC)
    raw["budget"]["rounds"] = 0
    with pytest.raises(ContractError, match="rounds"):
        ExperimentSpec.from_dict(raw)


def test_bad_stack_mode_rejected():
    raw = copy.deepcopy(VALID_SPEC)
    raw["stack"]["mode"] = "possessed"
    with pytest.raises(ContractError, match=r"stack\.mode"):
        ExperimentSpec.from_dict(raw)


def test_policy_coverage_gaps():
    assert POLICY.covers(["math", "code"]) == []
    assert POLICY.covers(["math", "history"]) == ["history"]


def test_policy_rejects_empty_model():
    with pytest.raises(ContractError, match="model"):
        OraclePolicy.from_dict({"math": ""})
