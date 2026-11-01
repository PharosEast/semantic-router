"""C9 bundle tests: packaging is a final-verify-only, byte-stable operation.

Covers the acceptance flow end to end through the CLI: a final live PASS
record with passing transfer guards bundles into the nine artifacts; byte
stability holds across repeated runs; and every rejection path (not final,
not PASS, transfer guard not passing, unknown record, failing config
validate, private content) exits 2 without writing a bundle.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import yaml
from recipe_lab.cli import main
from recipe_lab.contract import API_VERSION
from recipe_lab.ledger import Ledger
from recipe_lab.report import render_promotion_draft

EXIT_CONTRACT_ERROR = 2

FROZEN_HASH = "sha256:" + "1" * 64
SMALL = "mock/small"
LARGE = "mock/large"

CONFIG_YAML = """\
providers:
  vllm:
    endpoints: ["http://127.0.0.1:8001/v1"]
routing:
  default_model: mock/small
"""
SKELETON_YAML = """\
providers:
  vllm:
    endpoints: ["http://127.0.0.1:8001/v1"]
"""
PROFILE_JSON = '{"schema": "test-profile", "n": 3}\n'


def _write_spec(tmp_path: Path) -> str:
    spec = {
        "api_version": API_VERSION,
        "objective": {
            "metric_key": "capability.score",
            "direction": "maximize",
            "min_improvement_pct": 2.0,
        },
        "workloads": [
            {
                "name": "holdout-tune",
                "profile": "quick",
                "splits": {
                    "calib": {
                        "manifest": "calib.jsonl",
                        "content_hash": "sha256:" + "0" * 64,
                    }
                },
            },
            {
                "name": "holdout-transfer",
                "profile": "quick",
                "role": "transfer",
                "splits": {
                    "frozen_test": {
                        "manifest": "frozen_test.jsonl",
                        "content_hash": FROZEN_HASH,
                    }
                },
            },
        ],
        "attribution": {"oracle_policy": {"math": LARGE}},
        "budget": {"rounds": 1, "max_minutes_per_trial": 5},
        "models": [
            {"name": SMALL, "endpoint": "http://127.0.0.1:8001/v1"},
            {"name": LARGE, "endpoint": "http://127.0.0.1:8002/v1"},
        ],
        "cost_basis": {
            "source": "provider_pricing",
            "assumed_pricing": {
                name: {"prompt_per_1m": 0.1, "completion_per_1m": 0.4}
                for name in (SMALL, LARGE)
            },
        },
    }
    spec_path = tmp_path / "spec.yaml"
    spec_path.write_text(yaml.safe_dump(spec), encoding="utf-8")
    return str(spec_path)


def _pass_record(**overrides) -> dict:
    record = {
        "round": 3,
        "ts": "2026-01-01T00:00:00Z",
        "candidate": {
            "kind": "config",
            "ref": "candidate-lambda-0.30.yaml",
            "change_summary": "bootstrap lambda 0.30",
            "provenance": {"proposer": "bootstrap", "insight_refs": []},
        },
        "tier": "live",
        "baseline_ref": "run-base-1",
        "metrics": {"capability.score": 0.83},
        "verdict": "PASS",
        "dataset_hash": FROZEN_HASH,
        "reasons": [
            "paired CI lower bound 1.20pp >= min 1.00pp (200 paired cases, delta 2.00pp)"
        ],
        "gates": {"privacy": "pass", "no_pii": "pass"},
        "bench_refs": ["run-base-1", "run-cand-1"],
        "constraints": {
            "cost-cap": {
                "name": "cost-cap",
                "metric": "cost_per_question",
                "op": "lte",
                "candidate": 1.2e-05,
                "bound_value": 2.0e-05,
                "status": "pass",
            }
        },
        "paired": {
            "paired_cases": 200,
            "quality_delta": 0.02,
            "quality_delta_ci95": [0.012, 0.03],
            "quality_delta_bootstrap_ci95": [0.011, 0.029],
            "wins": 120,
            "losses": 60,
            "ties": 20,
        },
        "workload": "holdout-transfer",
        "split": "frozen_test",
        "notes": (
            'baseline=0.81; repetitions=3; routing_distribution='
            '{"mock/small": 120, "mock/large": 80}'
        ),
    }
    record.update(overrides)
    return record


def _write_ledger(tmp_path: Path, final_record: dict) -> tuple[str, str]:
    """A replay record outside the campaign hash plus the final PASS record."""
    ledger_path = tmp_path / "ledger.jsonl"
    ledger = Ledger(ledger_path)
    ledger.append(
        {
            "round": 2,
            "ts": "2025-12-31T00:00:00Z",
            "candidate": {
                "kind": "config",
                "ref": "candidate-lambda-0.10.yaml",
                "change_summary": "",
                "provenance": {"proposer": "bootstrap", "insight_refs": []},
            },
            "tier": "replay",
            "baseline_ref": "run-base-0",
            "metrics": {"routed_accuracy": 0.8},
            "verdict": "RANKED",
            "dataset_hash": "sha256:" + "2" * 64,
            "reasons": ["replay ranking only"],
        }
    )
    appended = ledger.append(final_record)
    return str(ledger_path), appended["record_id"]


def _write_inputs(tmp_path: Path) -> dict:
    config = tmp_path / "candidate-lambda-0.30.yaml"
    config.write_text(CONFIG_YAML, encoding="utf-8")
    skeleton = tmp_path / "scaffold-lambda-0.30.yaml"
    skeleton.write_text(SKELETON_YAML, encoding="utf-8")
    profile = tmp_path / "profile.json"
    profile.write_text(PROFILE_JSON, encoding="utf-8")
    validate_bin = tmp_path / "fake-vllm-sr"
    validate_bin.write_text("#!/bin/sh\necho 'config ok'\n", encoding="utf-8")
    validate_bin.chmod(0o755)
    return {
        "config": str(config),
        "skeleton": str(skeleton),
        "profile": str(profile),
        "validate_bin": str(validate_bin),
    }


def _bundle_args(tmp_path: Path, spec_path: str, ledger_path: str, record_id: str,
                 inputs: dict, out: Path, extra: list[str] | None = None) -> list[str]:
    return [
        "bundle",
        "--spec", spec_path,
        "--ledger", ledger_path,
        "--record", record_id,
        "--out", str(out),
        "--config", inputs["config"],
        "--skeleton", inputs["skeleton"],
        "--profile", inputs["profile"],
        "--validate-bin", inputs["validate_bin"],
        "--router-commit", "abc1234",
        "--model-revision", f"{SMALL}=rev-a",
        "--model-eval-script", inputs["skeleton"],
    ] + (extra or [])


BUNDLE_FILES = {
    "config.yaml",
    "validate.txt",
    "config.diff",
    "report.md",
    "profile.json",
    "ledger.jsonl",
    "provenance.json",
    "PROMOTION.md",
    "REVIEW.md",
}


def test_bundle_happy_path(tmp_path, capsys):
    spec_path = _write_spec(tmp_path)
    ledger_path, record_id = _write_ledger(tmp_path, _pass_record())
    inputs = _write_inputs(tmp_path)
    out = tmp_path / "bundle"
    rc = main(_bundle_args(tmp_path, spec_path, ledger_path, record_id, inputs, out))
    assert rc == 0
    assert {p.name for p in out.iterdir()} == BUNDLE_FILES

    # config.yaml and profile.json are byte-exact copies
    assert (out / "config.yaml").read_bytes() == Path(inputs["config"]).read_bytes()
    assert (out / "profile.json").read_text(encoding="utf-8") == PROFILE_JSON
    # validate.txt carries the validator output
    assert (out / "validate.txt").read_text(encoding="utf-8") == "config ok\n"
    # config.diff is a unified diff against the skeleton
    diff_text = (out / "config.diff").read_text(encoding="utf-8")
    assert diff_text.startswith("--- skeleton")
    assert "+  default_model: mock/small" in diff_text

    ledger_record = Ledger(ledger_path).read()[-1]
    # PROMOTION.md is the record-draft output for this record
    assert (out / "PROMOTION.md").read_text(encoding="utf-8") == render_promotion_draft(
        ledger_record
    )
    # REVIEW.md is the V5 template prefilled with the record identity
    review = (out / "REVIEW.md").read_text(encoding="utf-8")
    assert f"record: {record_id}" in review
    assert "decision (promote / hold / rework):" in review

    # ledger.jsonl is the campaign excerpt: only records sharing the dataset hash
    excerpt = [
        json.loads(line)
        for line in (out / "ledger.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert [row["record_id"] for row in excerpt] == [record_id]

    provenance = json.loads((out / "provenance.json").read_text(encoding="utf-8"))
    assert provenance["kind"] == "recipe-lab-bundle"
    assert provenance["record_id"] == record_id
    assert provenance["router_commit"] == "abc1234"
    assert provenance["models"] == {SMALL: "rev-a", LARGE: None}
    assert provenance["artifacts"]["config"]["sha256"] == (
        "sha256:" + hashlib.sha256(Path(inputs["config"]).read_bytes()).hexdigest()
    )
    assert provenance["run_ids"]["bench_refs"] == ["run-base-1", "run-cand-1"]
    assert provenance["datasets"]["workloads"]["holdout-transfer"]["frozen_test"] == FROZEN_HASH
    provenance_text = (out / "provenance.json").read_text(encoding="utf-8")
    # provenance cites basenames and hashes only - no absolute paths leak in
    assert str(Path(ledger_path).parent) not in provenance_text

    report = (out / "report.md").read_text(encoding="utf-8")
    assert "# recipe-lab acceptance bundle report" in report
    assert f"`{record_id}`" in report
    assert "paired cases: 200" in report
    assert "quality delta CI95: [0.012, 0.03]" in report
    assert "- mock/small: 120" in report  # observed routing distribution
    assert "## Cache and tools" in report
    assert "assumed pricing" in report
    assert "verdict: **PASS**" in report
    assert "holdout-transfer: 1x PASS" in report


def test_bundle_is_byte_stable(tmp_path):
    spec_path = _write_spec(tmp_path)
    ledger_path, record_id = _write_ledger(tmp_path, _pass_record())
    inputs = _write_inputs(tmp_path)
    out1 = tmp_path / "bundle-1"
    out2 = tmp_path / "bundle-2"
    assert main(_bundle_args(tmp_path, spec_path, ledger_path, record_id, inputs, out1)) == 0
    assert main(_bundle_args(tmp_path, spec_path, ledger_path, record_id, inputs, out2)) == 0
    files1 = {p.name: p.read_bytes() for p in out1.iterdir()}
    files2 = {p.name: p.read_bytes() for p in out2.iterdir()}
    assert set(files1) == BUNDLE_FILES
    assert files1 == files2


@pytest.mark.parametrize("gate_value", ["fail", "incomplete"])
def test_bundle_rejects_guard_not_passing(tmp_path, capsys, gate_value):
    spec_path = _write_spec(tmp_path)
    ledger_path, record_id = _write_ledger(
        tmp_path, _pass_record(gates={"privacy": gate_value})
    )
    inputs = _write_inputs(tmp_path)
    out = tmp_path / "bundle"
    rc = main(_bundle_args(tmp_path, spec_path, ledger_path, record_id, inputs, out))
    assert rc == EXIT_CONTRACT_ERROR
    assert "transfer guards not passing" in capsys.readouterr().err
    assert not out.exists()


def test_bundle_rejects_non_final_record(tmp_path, capsys):
    spec_path = _write_spec(tmp_path)
    record = _pass_record()
    record.pop("workload")
    record["split"] = "dev"
    ledger_path, record_id = _write_ledger(tmp_path, record)
    inputs = _write_inputs(tmp_path)
    out = tmp_path / "bundle"
    rc = main(_bundle_args(tmp_path, spec_path, ledger_path, record_id, inputs, out))
    assert rc == EXIT_CONTRACT_ERROR
    assert "not a final-verify record" in capsys.readouterr().err
    assert not out.exists()


def test_bundle_rejects_replay_record(tmp_path, capsys):
    spec_path = _write_spec(tmp_path)
    ledger_path, record_id = _write_ledger(tmp_path, _pass_record())
    replay_id = Ledger(ledger_path).read()[0]["record_id"]
    assert replay_id != record_id
    inputs = _write_inputs(tmp_path)
    out = tmp_path / "bundle"
    rc = main(_bundle_args(tmp_path, spec_path, ledger_path, replay_id, inputs, out))
    assert rc == EXIT_CONTRACT_ERROR
    assert "not a live-tier record" in capsys.readouterr().err
    assert not out.exists()


def test_bundle_rejects_fail_verdict(tmp_path, capsys):
    spec_path = _write_spec(tmp_path)
    ledger_path, _ = _write_ledger(tmp_path, _pass_record(verdict="FAIL"))
    record_id = Ledger(ledger_path).read()[-1]["record_id"]
    inputs = _write_inputs(tmp_path)
    out = tmp_path / "bundle"
    rc = main(_bundle_args(tmp_path, spec_path, ledger_path, record_id, inputs, out))
    assert rc == EXIT_CONTRACT_ERROR
    assert "only PASS records" in capsys.readouterr().err
    assert not out.exists()


def test_bundle_rejects_unknown_record(tmp_path, capsys):
    spec_path = _write_spec(tmp_path)
    ledger_path, _ = _write_ledger(tmp_path, _pass_record())
    inputs = _write_inputs(tmp_path)
    out = tmp_path / "bundle"
    rc = main(
        _bundle_args(tmp_path, spec_path, ledger_path, "rec-000000000000", inputs, out)
    )
    assert rc == EXIT_CONTRACT_ERROR
    assert "not found" in capsys.readouterr().err
    assert not out.exists()


def test_bundle_rejects_failing_validate(tmp_path, capsys):
    spec_path = _write_spec(tmp_path)
    ledger_path, record_id = _write_ledger(tmp_path, _pass_record())
    inputs = _write_inputs(tmp_path)
    bad_bin = tmp_path / "bad-vllm-sr"
    bad_bin.write_text("#!/bin/sh\necho 'bad config' >&2\nexit 1\n", encoding="utf-8")
    bad_bin.chmod(0o755)
    inputs["validate_bin"] = str(bad_bin)
    out = tmp_path / "bundle"
    rc = main(_bundle_args(tmp_path, spec_path, ledger_path, record_id, inputs, out))
    assert rc == EXIT_CONTRACT_ERROR
    assert "config validate" in capsys.readouterr().err
    assert not out.exists()


def test_bundle_rejects_private_path_in_config(tmp_path, capsys):
    spec_path = _write_spec(tmp_path)
    ledger_path, record_id = _write_ledger(tmp_path, _pass_record())
    inputs = _write_inputs(tmp_path)
    leaked = tmp_path / "leaked.yaml"
    leaked.write_text(
        CONFIG_YAML + "  # reviewed against C:\\Users\\alice\\notes.md\n",
        encoding="utf-8",
    )
    inputs["config"] = str(leaked)
    out = tmp_path / "bundle"
    rc = main(_bundle_args(tmp_path, spec_path, ledger_path, record_id, inputs, out))
    assert rc == EXIT_CONTRACT_ERROR
    err = capsys.readouterr().err
    assert "private content" in err
    assert "windows-user-path" in err
    assert "alice" not in err  # the matched content is never echoed
    assert not out.exists()


def test_bundle_rejects_custom_private_pattern(tmp_path, capsys):
    spec_path = _write_spec(tmp_path)
    ledger_path, record_id = _write_ledger(tmp_path, _pass_record())
    inputs = _write_inputs(tmp_path)
    leaked = tmp_path / "leaked.yaml"
    leaked.write_text(
        CONFIG_YAML + "  # endpoint vllm-internal.example:8000\n", encoding="utf-8"
    )
    inputs["config"] = str(leaked)
    out = tmp_path / "bundle"
    rc = main(
        _bundle_args(
            tmp_path, spec_path, ledger_path, record_id, inputs, out,
            extra=["--private-pattern", "vllm-internal"],
        )
    )
    assert rc == EXIT_CONTRACT_ERROR
    err = capsys.readouterr().err
    assert "private content" in err
    assert "custom[0]" in err
    assert not out.exists()
