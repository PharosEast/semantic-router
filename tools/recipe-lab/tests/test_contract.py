"""Contract tests: eager validation with actionable errors."""

from __future__ import annotations

import copy
import hashlib
import json

import pytest
import yaml
from recipe_lab.contract import (
    ContractError,
    ExperimentSpec,
    OraclePolicy,
    check_runnable_split,
    dataset_content_hash,
)

from tests.fixtures import POLICY, VALID_SPEC, VALID_SPLITS_SPEC


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


def test_objective_constraints_parse_and_roundtrip():
    # E: constraints are economic conditions declared on the objective
    raw = copy.deepcopy(VALID_SPEC)
    raw["objective"]["constraints"] = [
        {
            "name": "cost-not-worse",
            "metric": "total_spend_usd",
            "op": "lte",
            "bound": "baseline",
            "tolerance_pct": 5.0,
        }
    ]
    spec = ExperimentSpec.from_dict(raw)
    [constraint] = spec.objective.constraints
    assert (
        constraint.op == "lte"
        and constraint.bound == "baseline"
        and constraint.tolerance_pct == 5.0
    )
    spec2 = ExperimentSpec.from_dict(spec.to_dict())
    assert spec2.to_dict() == spec.to_dict()
    # a numeric bound is legal too
    raw["objective"]["constraints"] = [
        {
            "name": "keep-recall",
            "metric": "guard.pii.recall",
            "op": "gte",
            "bound": 0.95,
        }
    ]
    spec = ExperimentSpec.from_dict(raw)
    assert spec.objective.constraints[0].bound == 0.95


def test_objective_constraints_reject_invalid():
    raw = copy.deepcopy(VALID_SPEC)
    base = {
        "name": "cost-not-worse",
        "metric": "total_spend_usd",
        "op": "lte",
        "bound": "baseline",
    }
    for field, value, match in [
        ("op", "lt", "op"),
        ("bound", "guess", "bound"),
        ("tolerance_pct", -1, "tolerance_pct"),
        ("name", "", "name"),
        ("metric", "", "metric"),
    ]:
        bad = dict(base)
        bad[field] = value
        raw["objective"]["constraints"] = [bad]
        with pytest.raises(ContractError, match=match):
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


def test_evaluation_section_is_retired():
    # A5: replay and live are separate commands; the contract refuses the key
    raw = copy.deepcopy(VALID_SPEC)
    raw["evaluation"] = {"search_tier": "replay", "final_tier": "live"}
    with pytest.raises(ContractError, match="'evaluation' is retired"):
        ExperimentSpec.from_dict(raw)
    assert (
        "evaluation"
        not in ExperimentSpec.from_dict(copy.deepcopy(VALID_SPEC)).to_dict()
    )


# ---- B: three-way dataset splits ----


def test_legacy_workload_roundtrips_and_is_dev_only():
    spec = ExperimentSpec.from_dict(copy.deepcopy(VALID_SPEC))
    assert spec.workload.legacy
    assert spec.workload.split_hash("dev") == VALID_SPEC["workload"]["content_hash"]
    assert spec.workload.to_dict() == VALID_SPEC["workload"]


def test_splits_workload_roundtrip():
    spec = ExperimentSpec.from_dict(copy.deepcopy(VALID_SPLITS_SPEC))
    assert not spec.workload.legacy
    assert spec.workload.split_rule == "group by source_id; seed 42; 40/30/30"
    assert sorted(spec.workload.splits) == ["calib", "dev", "frozen_test"]
    assert spec.workload.split_hash("frozen_test").startswith("sha256:3")
    out = spec.to_dict()["workload"]
    assert set(out["splits"]) == {"calib", "dev", "frozen_test"}
    assert "manifest" not in out  # legacy keys gone


def test_splits_require_split_rule():
    raw = copy.deepcopy(VALID_SPLITS_SPEC)
    del raw["workload"]["split_rule"]
    with pytest.raises(ContractError, match="split_rule"):
        ExperimentSpec.from_dict(raw)


def test_splits_require_all_three():
    raw = copy.deepcopy(VALID_SPLITS_SPEC)
    del raw["workload"]["splits"]["frozen_test"]
    with pytest.raises(
        ContractError, match="exactly \\['calib', 'dev', 'frozen_test'\\]"
    ):
        ExperimentSpec.from_dict(raw)


def test_check_runnable_split_quarantines_frozen_test():
    with pytest.raises(ContractError, match="frozen_test is quarantined"):
        check_runnable_split("frozen_test")
    check_runnable_split("dev")
    check_runnable_split("calib")


def _write_splits_dataset(tmp_path, overlap_with_dev=False):
    """Three disjoint slice files; optionally leak a dev case into calib."""
    rows = {
        "calib": [{"case_id": "c1", "prompt": "p1"}, {"case_id": "c2", "prompt": "p2"}],
        "dev": [{"case_id": "c3", "prompt": "p3"}],
        "frozen_test": [{"case_id": "c4", "prompt": "p4"}],
    }
    if overlap_with_dev:
        rows["calib"].append({"case_id": "c3", "prompt": "p3"})
    spec = copy.deepcopy(VALID_SPLITS_SPEC)
    for name, case_rows in rows.items():
        target = tmp_path / f"{name}.jsonl"
        target.write_text(
            "".join(json.dumps(r) + "\n" for r in case_rows), encoding="utf-8"
        )
        spec["workload"]["splits"][name]["manifest"] = f"{name}.jsonl"
        spec["workload"]["splits"][name]["content_hash"] = dataset_content_hash(target)
    spec_path = tmp_path / "spec.yaml"
    spec_path.write_text(yaml.safe_dump(spec), encoding="utf-8")
    return spec_path


def test_verify_workload_checks_all_splits_and_disjointness(tmp_path):
    spec_path = _write_splits_dataset(tmp_path)
    spec = ExperimentSpec.from_yaml(spec_path)
    assert spec.verify_workload()["dev"] == spec.workload.split_hash("dev")  # type: ignore[index]


def test_verify_workload_rejects_overlapping_splits(tmp_path):
    spec_path = _write_splits_dataset(tmp_path, overlap_with_dev=True)
    spec = ExperimentSpec.from_yaml(spec_path)
    with pytest.raises(ContractError, match="overlap between 'calib' and 'dev'"):
        spec.verify_workload()


def test_verify_workload_split_selects_one_slice(tmp_path):
    spec_path = _write_splits_dataset(tmp_path)
    spec = ExperimentSpec.from_yaml(spec_path)
    assert spec.verify_workload(split="dev") == spec.workload.split_hash("dev")
    # unknown split names are refused
    with pytest.raises(ContractError, match="unknown workload split"):
        spec.verify_workload(split="train")


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
    with pytest.raises(ContractError, match=r"stack.mode"):
        ExperimentSpec.from_dict(raw)


def test_policy_coverage_gaps():
    assert POLICY.covers(["math", "code"]) == []
    assert POLICY.covers(["math", "history"]) == ["history"]


def test_policy_rejects_empty_model():
    with pytest.raises(ContractError, match="model"):
        OraclePolicy.from_dict({"math": ""})


# ---- A4: frozen dataset identity ----


def _write_frozen_spec(
    tmp_path, content: bytes = b'{"case_id":"c1"}\n{"case_id":"c2"}\n'
):
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
    assert (
        dataset_content_hash(path) == "sha256:" + hashlib.sha256(b"abc\n").hexdigest()
    )


def test_verify_workload_passes_on_exact_bytes(tmp_path):
    spec_path, _ = _write_frozen_spec(tmp_path)
    spec = ExperimentSpec.from_yaml(spec_path)
    assert spec.verify_workload() == {"dev": spec.workload.split_hash("dev")}
    assert spec.verify_workload(split="dev") == spec.workload.split_hash("dev")


def test_verify_workload_rejects_one_byte_change(tmp_path):
    spec_path, dataset = _write_frozen_spec(tmp_path)
    dataset.write_bytes(dataset.read_bytes() + b'{"case_id":"c3"}\n')  # dataset drifted
    spec = ExperimentSpec.from_yaml(spec_path)
    with pytest.raises(ContractError, match=r"dataset hash mismatch.*spec=.*file="):
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
