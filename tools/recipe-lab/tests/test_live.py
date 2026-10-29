"""Live-tier orchestration tests: guard parsing, metric pairing, verdict flow."""

from __future__ import annotations

import hashlib
import json

import pytest

from recipe_lab.cli import main
from recipe_lab.contract import ContractError
from recipe_lab.live import _pair_metrics, guards_from_rows, live_verdict_from_runs
from recipe_lab.srbench import SrbenchCli
from tests.fixtures import VALID_SPEC
from tests.test_contract import _write_splits_dataset


def test_guards_from_rows_validates():
    guards = guards_from_rows([{"name": "pii", "metric": "guard.pii", "value": 0.97, "floor": 0.95}])
    assert guards[0].passed is True
    # a missing floor is legal (guard not evaluated -> incomplete), a missing
    # name is not - the record would be unidentifiable
    with pytest.raises(ContractError, match="name"):
        guards_from_rows([{"metric": "m", "value": 0.9, "floor": 0.8}])


def test_pair_metrics_sectioned_payload():
    payload = {"baseline": {"m": {"capability.score": 0.80}}, "candidate": {"capability.score": 0.86}}
    base, cand = _pair_metrics(payload, "capability.score", [])
    assert (base, cand) == (0.80, 0.86)
    # sr-bench compare's actual shape: *_targets arrays with accuracy leaves
    srbench_payload = {
        "baseline_targets": [{"id": "single", "accuracy": 1.0}],
        "candidate_targets": [{"id": "mom", "accuracy": 0.833}],
    }
    base, cand = _pair_metrics(srbench_payload, "accuracy", [])
    assert (base, cand) == (1.0, 0.833)


def test_pair_metrics_prefers_routing_arm_over_reference_lane():
    # a candidate run keeps its "no routing" reference single lane before the
    # routing arm; the comparison arm is the router (mom), not the reference
    payload = {
        "baseline_targets": [{"id": "single", "accuracy": 0.5}],
        "candidate_targets": [
            {"id": "single", "accuracy": 0.5},
            {"id": "mom", "accuracy": 0.6667},
        ],
    }
    incomplete: list[str] = []
    base, cand = _pair_metrics(payload, "accuracy", incomplete)
    assert (base, cand) == (0.5, 0.6667)
    assert incomplete == []
    # without a mom kind, the last target carrying the metric is the arm
    payload["candidate_targets"] = [
        {"id": "single", "accuracy": 0.5},
        {"id": "router", "accuracy": 0.7},
    ]
    base, cand = _pair_metrics(payload, "accuracy", [])
    assert (base, cand) == (0.5, 0.7)


def test_pair_metrics_flat_two_values():
    base, cand = _pair_metrics({"a": 0.80, "b": 0.86}, "x", [])
    assert base is None and cand is None  # different key, nothing matched
    incomplete: list[str] = []
    base, cand = _pair_metrics({"capability.score": 0.80, "nested": {"capability.score": 0.86}}, "capability.score", incomplete)
    assert (base, cand) == (0.80, 0.86)
    assert incomplete == []


def test_pair_metrics_ambiguous_is_incomplete():
    incomplete: list[str] = []
    base, cand = _pair_metrics({"capability.score": 0.8}, "capability.score", incomplete)
    assert base is None and cand is None
    assert "ambiguous" in incomplete[0]


def _spec():
    from recipe_lab.contract import ExperimentSpec

    return ExperimentSpec.from_dict(VALID_SPEC)


def test_live_verdict_pass_flow():
    class FakeCli:
        def compare(self, base, cand):
            return {"baseline": {"capability.score": 0.80}, "candidate": {"capability.score": 0.86}}

    result = live_verdict_from_runs(
        spec=_spec(),
        cli=FakeCli(),
        baseline_run_id="run-a",
        candidate_run_id="run-b",
        guards=guards_from_rows([{"name": "pii", "metric": "g", "value": 0.97, "floor": 0.95}]),
    )
    assert result["verdict"] == "PASS"
    assert result["gates"] == {"pii": "pass"}
    assert result["bench_refs"] == ["run-a", "run-b"]


def test_live_verdict_ambiguous_is_inconclusive():
    class FakeCli:
        def compare(self, base, cand):
            return {"capability.score": 0.8}  # only one value -> ambiguous

    result = live_verdict_from_runs(
        spec=_spec(),
        cli=FakeCli(),
        baseline_run_id="run-a",
        candidate_run_id="run-b",
        guards=[],
    )
    assert result["verdict"] == "INCONCLUSIVE"
    assert result["gates"] == {}


# ---- B: --split quarantine gate on the CLI ----


def test_live_verdict_rejects_frozen_test_without_final(tmp_path, capsys):
    spec_path = str(_write_splits_dataset(tmp_path))
    assert (
        main(
            [
                "live-verdict",
                "--spec",
                spec_path,
                "--baseline-run-id",
                "run-a",
                "--candidate-run-id",
                "run-b",
                "--split",
                "frozen_test",
            ]
        )
        == 2
    )
    assert "frozen_test is quarantined" in capsys.readouterr().err


def test_live_verdict_final_cites_frozen_test(monkeypatch, tmp_path, capsys):
    from recipe_lab import cli

    spec_path = str(_write_splits_dataset(tmp_path))

    def fake_verdict(**kwargs):
        return {
            "verdict": "PASS",
            "reasons": [],
            "gates": {},
            "bench_refs": ["run-a", "run-b"],
            "metrics": {"baseline": 0.5, "candidate": 0.6},
        }

    monkeypatch.setattr(cli, "live_verdict_from_runs", fake_verdict)
    ledger_path = tmp_path / "ledger.jsonl"
    assert (
        main(
            [
                "live-verdict",
                "--spec",
                spec_path,
                "--baseline-run-id",
                "run-a",
                "--candidate-run-id",
                "run-b",
                "--split",
                "frozen_test",
                "--final",
                "--ledger",
                str(ledger_path),
            ]
        )
        == 0
    )
    parsed = json.loads(capsys.readouterr().out)
    assert parsed["verdict"] == "PASS"
    rows = [json.loads(line) for line in ledger_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert rows[-1]["split"] == "frozen_test"
    expected = "sha256:" + hashlib.sha256((tmp_path / "frozen_test.jsonl").read_bytes()).hexdigest()
    assert rows[-1]["dataset_hash"] == expected
