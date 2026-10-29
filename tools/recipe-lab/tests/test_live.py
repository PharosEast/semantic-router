"""Live-tier orchestration tests: guard measurements, metric pairing, verdict flow."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from recipe_lab.cli import main
from recipe_lab.contract import ContractError, ExperimentSpec
from recipe_lab.live import _pair_metrics, live_verdict_from_runs, measurements_from_rows, resolve_gates
from recipe_lab.srbench import SrbenchCli
from tests.fixtures import VALID_SPEC
from tests.test_contract import _write_splits_dataset


# ---- A3: the contract is the only source of guard floors ----


def test_measurements_from_rows_rejects_floors_and_unknown_keys():
    # a floor inside the measurement file is a contract bypass
    with pytest.raises(ContractError, match="floor"):
        measurements_from_rows([{"name": "pii", "candidate": 0.97, "baseline": 0.96, "floor": 0.95}])
    with pytest.raises(ContractError, match="non-measurement keys"):
        measurements_from_rows([{"name": "pii", "candidate": 0.97, "metric": "g"}])
    with pytest.raises(ContractError, match="name"):
        measurements_from_rows([{"candidate": 0.97}])
    with pytest.raises(ContractError, match="candidate"):
        measurements_from_rows([{"name": "pii", "candidate": "high"}])


def test_resolve_gates_numeric_floor_pass_and_breach():
    from recipe_lab.contract import Gate

    gates = [Gate(name="pii-recall-floor", metric="guard.pii.recall", floor=0.95)]
    guards, undeclared = resolve_gates(gates, [{"name": "pii-recall-floor", "candidate": 0.97, "baseline": 0.5}])
    assert guards[0].floor == 0.95 and guards[0].value == 0.97 and guards[0].passed
    guards, _ = resolve_gates(gates, [{"name": "pii-recall-floor", "candidate": 0.90, "baseline": 0.5}])
    assert guards[0].passed is False


def test_resolve_gates_baseline_floor_uses_measured_baseline():
    gates = ExperimentSpec.from_dict(VALID_SPEC).gates
    guards, undeclared = resolve_gates(gates, [{"name": "pii-recall-floor", "candidate": 0.97, "baseline": 0.96}])
    assert undeclared == []
    assert guards[0].floor == 0.96 and guards[0].value == 0.97 and guards[0].passed


def test_resolve_gates_missing_measurement_is_unevaluated():
    gates = ExperimentSpec.from_dict(VALID_SPEC).gates
    # no measurement at all
    guards, _ = resolve_gates(gates, [])
    assert not guards[0].evaluated
    # baseline floor without a measured baseline value
    guards, _ = resolve_gates(gates, [{"name": "pii-recall-floor", "candidate": 0.97}])
    assert not guards[0].evaluated
    # candidate value missing
    guards, _ = resolve_gates(gates, [{"name": "pii-recall-floor", "baseline": 0.96}])
    assert not guards[0].evaluated


def test_resolve_gates_reports_undeclared_without_veto():
    gates = ExperimentSpec.from_dict(VALID_SPEC).gates
    guards, undeclared = resolve_gates(
        gates,
        [
            {"name": "pii-recall-floor", "candidate": 0.97, "baseline": 0.96},
            {"name": "unheard-of", "candidate": 0.1, "baseline": 0.9},
        ],
    )
    assert undeclared == ["unheard-of"]
    assert guards[0].passed  # the declared gate still resolves normally


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
        # VALID_SPEC declares floor: baseline -> the measured baseline is the floor
        measurements=[{"name": "pii-recall-floor", "candidate": 0.97, "baseline": 0.96}],
    )
    assert result["verdict"] == "PASS"
    assert result["gates"] == {"pii-recall-floor": "pass"}
    assert result["undeclared_measurements"] == []
    assert result["bench_refs"] == ["run-a", "run-b"]


def test_live_verdict_baseline_floor_breach_fails():
    class FakeCli:
        def compare(self, base, cand):
            return {"baseline": {"capability.score": 0.80}, "candidate": {"capability.score": 0.86}}

    result = live_verdict_from_runs(
        spec=_spec(),
        cli=FakeCli(),
        baseline_run_id="run-a",
        candidate_run_id="run-b",
        measurements=[{"name": "pii-recall-floor", "candidate": 0.90, "baseline": 0.96}],
    )
    assert result["verdict"] == "FAIL"
    assert result["gates"] == {"pii-recall-floor": "fail"}
    assert "guard floor violated" in result["reasons"][0]


def test_live_verdict_missing_measurement_is_inconclusive():
    class FakeCli:
        def compare(self, base, cand):
            return {"baseline": {"capability.score": 0.80}, "candidate": {"capability.score": 0.86}}

    result = live_verdict_from_runs(
        spec=_spec(),
        cli=FakeCli(),
        baseline_run_id="run-a",
        candidate_run_id="run-b",
        measurements=[],
    )
    assert result["verdict"] == "INCONCLUSIVE"
    assert result["gates"] == {"pii-recall-floor": "incomplete"}
    assert "guard evaluation incomplete" in result["reasons"][0]


def test_live_verdict_undeclared_measurement_does_not_veto():
    class FakeCli:
        def compare(self, base, cand):
            return {"baseline": {"capability.score": 0.80}, "candidate": {"capability.score": 0.86}}

    result = live_verdict_from_runs(
        spec=_spec(),
        cli=FakeCli(),
        baseline_run_id="run-a",
        candidate_run_id="run-b",
        measurements=[{"name": "pii-recall-floor", "candidate": 0.97, "baseline": 0.96}, {"name": "extra", "candidate": 0.1}],
    )
    assert result["verdict"] == "PASS"
    assert result["undeclared_measurements"] == ["extra"]
    assert any("undeclared guard measurements" in reason for reason in result["reasons"])


def test_live_verdict_empty_gates_is_explicit():
    spec_dict = dict(VALID_SPEC)
    spec_dict["gates"] = []

    class FakeCli:
        def compare(self, base, cand):
            return {"baseline": {"capability.score": 0.80}, "candidate": {"capability.score": 0.86}}

    result = live_verdict_from_runs(
        spec=ExperimentSpec.from_dict(spec_dict),
        cli=FakeCli(),
        baseline_run_id="run-a",
        candidate_run_id="run-b",
        measurements=[{"name": "anything", "candidate": 0.5, "baseline": 0.4}],
    )
    assert result["gates"] == {}
    assert result["gates_note"] == "no guard declared"
    assert result["undeclared_measurements"] == ["anything"]


def test_live_verdict_ambiguous_is_inconclusive():
    class FakeCli:
        def compare(self, base, cand):
            return {"capability.score": 0.8}  # only one value -> ambiguous

    result = live_verdict_from_runs(
        spec=_spec(),
        cli=FakeCli(),
        baseline_run_id="run-a",
        candidate_run_id="run-b",
        measurements=[{"name": "pii-recall-floor", "candidate": 0.97, "baseline": 0.96}],
    )
    assert result["verdict"] == "INCONCLUSIVE"
    assert result["gates"] == {"pii-recall-floor": "pass"}


# ---- A3: CLI rejects measurement floors with exit code 2 ----


def test_live_verdict_rejects_floor_in_measurement_file(tmp_path, capsys):
    spec_path = str(_write_splits_dataset(tmp_path))
    measurements = tmp_path / "measurements.jsonl"
    measurements.write_text(
        json.dumps({"name": "pii-recall-floor", "candidate": 0.97, "baseline": 0.96, "floor": 0.9}) + "\n",
        encoding="utf-8",
    )
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
                "--guard-measurements",
                str(measurements),
            ]
        )
        == 2
    )
    assert "floor" in capsys.readouterr().err


# ---- G2: --insight-refs on live-verdict ----


def _seed_replay_ledger(tmp_path) -> tuple[str, str]:
    """An empty ledger with one replay record; returns (ledger_path, record_id)."""
    from recipe_lab.ledger import Ledger, LedgerRecord

    ledger_path = tmp_path / "seed-ledger.jsonl"
    stored = Ledger(ledger_path).append(
        LedgerRecord(
            round=1,
            ts="2026-10-29T00:00:00Z",
            candidate={"kind": "config", "ref": "seed.yaml", "change_summary": "s", "provenance": {"proposer": "p", "insight_refs": []}},
            tier="replay",
            baseline_ref="base@main",
            metrics={"routed_accuracy": 0.9},
            verdict="RANKED",
            dataset_hash="sha256:" + "0" * 64,
        )
    )
    return str(ledger_path), stored["record_id"]


def _live_verdict_args(spec_path: str, ledger: str | None = None) -> list[str]:
    args = ["live-verdict", "--spec", spec_path, "--baseline-run-id", "run-a", "--candidate-run-id", "run-b"]
    if ledger:
        args += ["--ledger", ledger]
    return args


def test_live_verdict_insight_refs_four_forms_merge_and_dedupe(monkeypatch, tmp_path, capsys):
    from recipe_lab import cli

    spec_path = str(_write_splits_dataset(tmp_path))
    seed_ledger, record_id = _seed_replay_ledger(tmp_path)
    ledger_path = Path(seed_ledger)  # rec- refs resolve against --ledger itself
    attr_file = tmp_path / "attr.json"
    attr_file.write_text("{}", encoding="utf-8")
    diag_file = tmp_path / "diagnosis.json"
    diag_file.write_text("{}", encoding="utf-8")
    sidecar = tmp_path / "cand.json"
    sidecar.write_text(
        json.dumps({"kind": "config", "ref": "cand-1.yaml", "change_summary": "s", "provenance": {"proposer": "agent", "insight_refs": ["bench:run-a"]}}),
        encoding="utf-8",
    )
    attr_hash = hashlib.sha256(attr_file.read_bytes()).hexdigest()
    diag_hash = hashlib.sha256(diag_file.read_bytes()).hexdigest()
    refs = ",".join(
        [
            record_id,
            "bench:run-a",  # already on the sidecar -> deduped
            f"attr:{attr_file}@sha256:{attr_hash}",
            f"diag:{diag_file}@sha256:{diag_hash}#threshold.domain_code_complex",
        ]
    )
    monkeypatch.setattr(
        cli,
        "live_verdict_from_runs",
        lambda **kwargs: {"verdict": "PASS", "reasons": [], "improvement_pct": 1.0, "gates": {}, "bench_refs": ["run-a", "run-b"], "metrics": {"baseline": 0.5, "candidate": 0.6}},
    )
    args = _live_verdict_args(spec_path, str(ledger_path)) + [
        "--candidate", str(sidecar),
        "--insight-refs", refs,
    ]
    assert main(args) == 0
    rows = [json.loads(line) for line in ledger_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert rows[-1]["candidate"]["provenance"]["insight_refs"] == [
        "bench:run-a",
        record_id,
        f"attr:{attr_file}@sha256:{attr_hash}",
        f"diag:{diag_file}@sha256:{diag_hash}#threshold.domain_code_complex",
    ]


def test_live_verdict_unknown_record_ref_exits_2(tmp_path, capsys):
    seed_ledger, _ = _seed_replay_ledger(tmp_path)
    spec_path = str(_write_splits_dataset(tmp_path))
    rc = main(_live_verdict_args(spec_path, seed_ledger) + ["--insight-refs", "rec-000000000000"])
    assert rc == 2
    assert "not found in ledger" in capsys.readouterr().err


def test_live_verdict_hash_mismatch_exits_2(tmp_path, capsys):
    spec_path = str(_write_splits_dataset(tmp_path))
    attr_file = tmp_path / "attr.json"
    attr_file.write_text("{}", encoding="utf-8")
    bad_hash = "sha256:" + "f" * 64
    rc = main(_live_verdict_args(spec_path) + ["--insight-refs", f"attr:{attr_file}@{bad_hash}"])
    assert rc == 2
    assert "hash mismatch" in capsys.readouterr().err


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
