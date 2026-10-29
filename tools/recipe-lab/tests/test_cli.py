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


def test_run_round_beyond_budget_exits_2(tmp_path, capsys):
    # A5: budget.rounds is a hard cap; the check fires before any stack work
    spec_path, cases = _write_frozen_spec(tmp_path)
    rc = main(
        [
            "run", "--spec", spec_path, "--round", "6",
            "--cases", str(cases),
            "--bank", str(tmp_path / "bank.json"),
            "--ledger", str(tmp_path / "ledger.jsonl"),
            "--candidates", str(tmp_path / "cands"),
            "--router-bin", "/bin/true",
            "--mock-backend", str(tmp_path / "mock.py"),
            "--output-dir", str(tmp_path / "out"),
        ]
    )
    assert rc == 2
    assert "budget exhausted" in capsys.readouterr().err


# ---- B: --split handling + the split command ----


def test_attribute_rejects_frozen_test_split(tmp_path, capsys):
    spec_path, cases = _write_frozen_spec(tmp_path)
    records = tmp_path / "records.jsonl"
    records.write_text("", encoding="utf-8")
    assert main(["attribute", "--spec", spec_path, "--records", str(records), "--split", "frozen_test"]) == 2
    assert "frozen_test is quarantined" in capsys.readouterr().err


def test_attribute_rejects_calib_on_legacy_workload(tmp_path, capsys):
    spec_path, _ = _write_frozen_spec(tmp_path)  # legacy single-dataset form
    records = tmp_path / "records.jsonl"
    records.write_text("", encoding="utf-8")
    assert main(["attribute", "--spec", spec_path, "--records", str(records), "--split", "calib"]) == 2
    assert "unknown workload split 'calib'" in capsys.readouterr().err


def test_attribute_dev_split_prints_legacy_warning(tmp_path, capsys):
    spec_path, cases = _write_frozen_spec(tmp_path)
    records = tmp_path / "records.jsonl"
    records.write_text(
        json.dumps(
            {"case_id": "c1", "truth": "math", "matched_decision": "math", "routed": "local/model-math", "answer_ok": True}
        )
        + "\n",
        encoding="utf-8",
    )
    out_dir = tmp_path / "out"
    assert main(["attribute", "--spec", spec_path, "--records", str(records), "--cases", str(cases), "--output-dir", str(out_dir)]) == 0
    assert "legacy single-dataset form" in capsys.readouterr().err


def test_split_command_outputs_spec_ready_block(tmp_path, capsys):
    cases = tmp_path / "cases.jsonl"
    rows = []
    for i in range(24):
        rows.append(json.dumps({"case_id": f"c{i}", "prompt": f"prompt {i}", "truth": "math", "source_id": f"src{i % 6}"}))
    cases.write_text("\n".join(rows) + "\n", encoding="utf-8")
    out_dir = tmp_path / "slices"
    assert main(["split", "--cases", str(cases), "--output-dir", str(out_dir)]) == 0
    parsed = json.loads(capsys.readouterr().out)
    assert parsed["split_rule"] == "group by source_id; seed 42; 40/30/30"
    assert sorted(parsed["splits"]) == ["calib", "dev", "frozen_test"]
    for name, spec_block in parsed["splits"].items():
        assert spec_block["manifest"] == f"{name}.jsonl"
        assert spec_block["content_hash"].startswith("sha256:")
        assert (out_dir / f"{name}.jsonl").is_file()
    # totals match: no case lost or duplicated
    total = sum(
        sum(1 for _ in (out_dir / f"{name}.jsonl").open(encoding="utf-8") if _.strip())
        for name in parsed["splits"]
    )
    assert total == 24
