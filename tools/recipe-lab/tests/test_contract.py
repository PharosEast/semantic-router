"""Contract tests: eager validation with actionable errors."""

from __future__ import annotations

import copy
import hashlib

import pytest
import yaml

from recipe_lab.contract import ContractError, ExperimentSpec, OraclePolicy, dataset_content_hash
from tests.fixtures import POLICY, VALID_SPEC


def test_valid_spec_roundtrip():
    spec = ExperimentSpec.from_dict(copy.deepcopy(VALID_SPEC))
    assert spec.objective.metric_key == "capability.score"
    assert spec.objective.direction == "maximize"
    assert spec.objective.replay_metric == "routed_accuracy"
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


def test_replay_metric_default_and_reject():
    # A1: replay_metric defaults to routed_accuracy; unknown values rejected
    raw = copy.deepcopy(VALID_SPEC)
    del raw["objective"]["replay_metric"]
    spec = ExperimentSpec.from_dict(raw)
    assert spec.objective.replay_metric == "routed_accuracy"
    raw["objective"]["replay_metric"] = "latency_p99"
    with pytest.raises(ContractError, match="replay_metric"):
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
    with pytest.raises(ContractError, match="stack.mode"):
        ExperimentSpec.from_dict(raw)


def test_policy_coverage_gaps():
    assert POLICY.covers(["math", "code"]) == []
    assert POLICY.covers(["math", "history"]) == ["history"]


def test_policy_rejects_empty_model():
    with pytest.raises(ContractError, match="model"):
        OraclePolicy.from_dict({"math": ""})


# ---- A4: frozen dataset identity ----


def _write_frozen_spec(tmp_path, content: bytes = b'{"case_id":"c1"}\n{"case_id":"c2"}\n'):
    """A spec + dataset pair: workload.content_hash covers the exact bytes."""
    dataset = tmp_path / "datasets" / "dev.jsonl"
    dataset.parent.mkdir(parents=True, exist_ok=True)
    dataset.write_bytes(content)
    spec = copy.deepcopy(VALID_SPEC)
    spec["workload"] = {
        "manifest": "datasets/dev.jsonl",
        "content_hash": dataset_content_hash(dataset),
        "profile": "quick",
    }
    spec_path = tmp_path / "spec.yaml"
    spec_path.write_text(yaml.safe_dump(spec), encoding="utf-8")
    return spec_path, dataset


def test_dataset_content_hash_is_sha256_of_raw_bytes(tmp_path):
    path = tmp_path / "d.jsonl"
    path.write_bytes(b"abc\n")
    assert dataset_content_hash(path) == "sha256:" + hashlib.sha256(b"abc\n").hexdigest()


def test_verify_workload_passes_on_exact_bytes(tmp_path):
    spec_path, _ = _write_frozen_spec(tmp_path)
    spec = ExperimentSpec.from_yaml(spec_path)
    assert spec.verify_workload() == spec.workload.content_hash


def test_verify_workload_rejects_one_byte_change(tmp_path):
    spec_path, dataset = _write_frozen_spec(tmp_path)
    dataset.write_bytes(dataset.read_bytes() + b'{"case_id":"c3"}\n')  # dataset drifted
    spec = ExperimentSpec.from_yaml(spec_path)
    with pytest.raises(ContractError, match="dataset hash mismatch: spec=.* file="):
        spec.verify_workload()


def test_verify_workload_missing_manifest_is_actionable(tmp_path):
    spec_path, _ = _write_frozen_spec(tmp_path)
    spec = ExperimentSpec.from_yaml(spec_path)
    (tmp_path / "datasets" / "dev.jsonl").unlink()
    with pytest.raises(ContractError, match="dataset manifest not found"):
        spec.verify_workload()


def test_require_dataset_file_rejects_drifted_cases(tmp_path):
    spec_path, dataset = _write_frozen_spec(tmp_path)
    spec = ExperimentSpec.from_yaml(spec_path)
    drifted = tmp_path / "other.jsonl"
    drifted.write_bytes(b'{"case_id":"x"}\n')
    with pytest.raises(ContractError, match="cases hash mismatch"):
        spec.require_dataset_file(drifted, "cases")
    with pytest.raises(ContractError, match="cannot read cases"):
        spec.require_dataset_file(tmp_path / "missing.jsonl", "cases")
    # the exact dataset bytes pass
    assert spec.require_dataset_file(dataset, "cases").startswith("sha256:")
