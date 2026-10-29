"""Ledger tests: verdict-tier isolation, self-sufficiency of live records."""

from __future__ import annotations

import pytest

from recipe_lab.contract import ContractError, ExperimentSpec
from recipe_lab.ledger import Ledger, LedgerRecord
from tests.fixtures import VALID_SPEC


def base_record(**overrides) -> dict:
    record = {
        "round": 3,
        "ts": "2026-10-26T10:00:00Z",
        "candidate": {
            "kind": "config",
            "ref": "config-rev@abc123",
            "change_summary": "domain threshold 0.30 -> 0.28",
            "provenance": {"proposer": "threshold-search", "insight_refs": []},
        },
        "tier": "replay",
        "baseline_ref": "config-revision@main",
        "metrics": {"capability.score": 0.794},
        "verdict": "RANKED",
        "dataset_hash": "sha256:" + "1" * 64,
    }
    record.update(overrides)
    return record


def live_record(**overrides) -> dict:
    record = base_record(
        tier="live",
        verdict="PASS",
        gates={"pii-recall-floor": "pass"},
        bench_refs=["sr-bench-run-1"],
        metrics={"capability.score": 0.831},
    )
    record.update(overrides)
    return record


def test_replay_record_roundtrip(tmp_path):
    path = tmp_path / "ledger.jsonl"
    ledger = Ledger(path)
    stored = ledger.append(base_record())
    assert len(ledger.read()) == 1
    assert ledger.read()[0] == stored
    assert stored["verdict"] == "RANKED"


def test_replay_rejects_three_state_verdict(tmp_path):
    ledger = Ledger(tmp_path / "ledger.jsonl")
    with pytest.raises(ContractError, match="RANKED"):
        ledger.append(base_record(verdict="PASS"))


def test_live_rejects_ranked_verdict(tmp_path):
    ledger = Ledger(tmp_path / "ledger.jsonl")
    with pytest.raises(ContractError, match="RANKED"):
        ledger.append(live_record(verdict="RANKED"))


def test_live_requires_gates_and_bench_refs(tmp_path):
    ledger = Ledger(tmp_path / "ledger.jsonl")
    with pytest.raises(ContractError, match="gates"):
        ledger.append(live_record(gates=None))
    with pytest.raises(ContractError, match="bench_refs"):
        ledger.append(live_record(bench_refs=[]))
    # INCONCLUSIVE means incomplete evidence by definition: empty bench_refs allowed
    stored = ledger.append(live_record(verdict="INCONCLUSIVE", bench_refs=[]))
    assert stored["verdict"] == "INCONCLUSIVE"


def test_missing_required_field_rejected(tmp_path):
    ledger = Ledger(tmp_path / "ledger.jsonl")
    record = base_record()
    del record["dataset_hash"]
    with pytest.raises(ContractError, match="dataset_hash"):
        ledger.append(record)


def test_bool_round_rejected(tmp_path):
    ledger = Ledger(tmp_path / "ledger.jsonl")
    with pytest.raises(ContractError, match="round"):
        ledger.append(base_record(round=True))


def test_filter_and_top_replay(tmp_path):
    path = tmp_path / "ledger.jsonl"
    ledger = Ledger(path)
    ledger.append(base_record(metrics={"capability.score": 0.75}, round=1))
    ledger.append(base_record(metrics={"capability.score": 0.81}, round=2, coverage_warning=True))
    ledger.append(base_record(metrics={"capability.score": 0.79}, round=3))
    ledger.append(
        base_record(metrics={"capability.score": 0.95}, round=4, candidate={"kind": "config", "ref": "x", "provenance": {"proposer": "agent:1", "insight_refs": []}})
    )
    assert len(ledger.filter(tier="replay", verdict="RANKED")) == 4
    assert len(ledger.filter(round_no=2)) == 1
    # top_replay skips coverage_warning and picks the highest metric
    best = ledger.top_replay("capability.score")
    assert best is not None and best["metrics"]["capability.score"] == 0.95


def test_dataclass_record_from_dict(tmp_path):
    ledger = Ledger(tmp_path / "ledger.jsonl")
    record = LedgerRecord.from_dict(base_record())
    stored = ledger.append(record)
    assert stored["round"] == 3


def test_corrupt_line_raises(tmp_path):
    path = tmp_path / "ledger.jsonl"
    path.write_text('{"round": 1}\nnot-json\n', encoding="utf-8")
    ledger = Ledger(path)
    with pytest.raises(ContractError, match="line 2"):
        ledger.read()


def test_spec_metric_key_roundtrip(tmp_path):
    # the ledger stores metrics keyed by the objective metric_key
    spec = ExperimentSpec.from_dict(VALID_SPEC)
    ledger = Ledger(tmp_path / "ledger.jsonl")
    record = base_record(metrics={spec.objective.metric_key: 0.8})
    ledger.append(record)
    best = ledger.top_replay(spec.objective.metric_key)
    assert best is not None and best["metrics"][spec.objective.metric_key] == 0.8


# ---- A5: timeout status never ranks ----


def test_timeout_record_roundtrip_and_never_ranks(tmp_path):
    ledger = Ledger(tmp_path / "ledger.jsonl")
    stored = ledger.append(
        base_record(metrics={}, reasons=["trial timeout: ..."], coverage_warning=True, status="timeout")
    )
    assert stored["status"] == "timeout"
    ledger.append(base_record(metrics={"capability.score": 0.77}))
    best = ledger.top_replay("capability.score")
    assert best is not None and best["metrics"]["capability.score"] == 0.77


def test_timeout_status_is_the_only_allowed_one(tmp_path):
    ledger = Ledger(tmp_path / "ledger.jsonl")
    with pytest.raises(ContractError, match="status must be 'timeout'"):
        ledger.append(base_record(status="crashed"))
