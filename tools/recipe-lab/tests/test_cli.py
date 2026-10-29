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
