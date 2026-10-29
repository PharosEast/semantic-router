"""CLI tests for the A4 contract gates: dataset hash command + verification."""

from __future__ import annotations

import hashlib
import json

import pytest
import yaml

from recipe_lab.cli import main
from tests.fixtures import VALID_SPEC


def _write_frozen_spec(tmp_path):
    """Spec YAML whose workload hash covers tmp_path/cases.jsonl bytes."""
    cases = tmp_path / "cases.jsonl"
    cases.write_bytes(b'{"case_id":"c1","prompt":"p","truth":"math"}\n')
    spec = dict(VALID_SPEC)
    spec["workload"] = {
        "manifest": "cases.jsonl",
        "content_hash": "sha256:" + hashlib.sha256(cases.read_bytes()).hexdigest(),
        "profile": "quick",
    }
    spec_path = tmp_path / "spec.yaml"
    spec_path.write_text(yaml.safe_dump(spec), encoding="utf-8")
    return str(spec_path), cases


def test_hash_command_prints_content_hash(tmp_path, capsys):
    cases = tmp_path / "cases.jsonl"
    cases.write_bytes(b'{"case_id":"c1"}\n')
    assert main(["hash", "--file", str(cases)]) == 0
    parsed = json.loads(capsys.readouterr().out)
    assert parsed["content_hash"] == "sha256:" + hashlib.sha256(b'{"case_id":"c1"}\n').hexdigest()


def test_hash_command_missing_file_exits_2(tmp_path, capsys):
    assert main(["hash", "--file", str(tmp_path / "nope.jsonl")]) == 2
    assert "cannot read dataset file" in capsys.readouterr().err


def test_attribute_cases_hash_mismatch_exits_2(tmp_path, capsys):
    spec_path, _ = _write_frozen_spec(tmp_path)
    records = tmp_path / "records.jsonl"
    records.write_text(
        json.dumps(
            {"case_id": "c1", "truth": "math", "matched_decision": "math", "routed": "local/model-math", "answer_ok": True}
        )
        + "\n",
        encoding="utf-8",
    )
    drifted = tmp_path / "drifted.jsonl"
    drifted.write_bytes(b'{"case_id":"c9","prompt":"p","truth":"math"}\n')
    assert main(["attribute", "--spec", spec_path, "--records", str(records), "--cases", str(drifted)]) == 2
    assert "cases hash mismatch" in capsys.readouterr().err


def test_attribute_missing_manifest_exits_2(tmp_path, capsys):
    spec_path, cases = _write_frozen_spec(tmp_path)
    cases.unlink()  # the workload manifest itself is gone
    records = tmp_path / "records.jsonl"
    records.write_text("", encoding="utf-8")
    assert main(["attribute", "--spec", spec_path, "--records", str(records)]) == 2
    assert "dataset manifest not found" in capsys.readouterr().err


def test_attribute_cases_hash_match_passes(tmp_path, capsys):
    spec_path, cases = _write_frozen_spec(tmp_path)
    records = tmp_path / "records.jsonl"
    records.write_text(
        json.dumps(
            {"case_id": "c1", "truth": "math", "matched_decision": "math", "routed": "local/model-math", "answer_ok": True}
        )
        + "\n",
        encoding="utf-8",
    )
    assert main(["attribute", "--spec", spec_path, "--records", str(records), "--cases", str(cases)]) == 0


# ---- A1: replay_metric contract + ledger top --spec ----


def _replay_record(ref: str, metric: str, value: float) -> dict:
    return {
        "round": 1,
        "ts": "2026-10-29T00:00:00Z",
        "candidate": {"kind": "config", "ref": ref, "change_summary": "s", "provenance": {"proposer": "p", "insight_refs": []}},
        "tier": "replay",
        "baseline_ref": "base@main",
        "metrics": {metric: value},
        "verdict": "RANKED",
        "dataset_hash": "sha256:" + "0" * 64,
        "reasons": [],
        "gates": None,
        "bench_refs": [],
        "attribution_delta": None,
        "attribution_ref": None,
        "coverage_warning": False,
        "notes": "",
    }


def _write_a1_spec(tmp_path) -> str:
    spec = dict(VALID_SPEC)
    spec_path = tmp_path / "spec.yaml"
    spec_path.write_text(yaml.safe_dump(spec), encoding="utf-8")
    return str(spec_path)


def test_ledger_top_spec_uses_replay_metric(tmp_path, capsys):
    ledger = tmp_path / "ledger.jsonl"
    ledger.write_text(
        json.dumps(_replay_record("a.yaml", "routed_accuracy", 0.80)) + "\n"
        + json.dumps(_replay_record("b.yaml", "routed_accuracy", 0.91)) + "\n",
        encoding="utf-8",
    )
    spec_path = _write_a1_spec(tmp_path)
    assert main(["ledger", "top", "--file", str(ledger), "--spec", spec_path]) == 0
    assert json.loads(capsys.readouterr().out)["candidate"]["ref"] == "b.yaml"


def test_ledger_top_metric_key_and_spec_are_exclusive(tmp_path, capsys):
    ledger = tmp_path / "ledger.jsonl"
    ledger.write_text("", encoding="utf-8")
    with pytest.raises(SystemExit) as exc:
        main(["ledger", "top", "--file", str(ledger), "--metric-key", "accuracy", "--spec", "s.yaml"])
    assert exc.value.code == 2  # argparse usage error
    with pytest.raises(SystemExit) as exc2:  # neither given
        main(["ledger", "top", "--file", str(ledger)])
    assert exc2.value.code == 2
    capsys.readouterr()
