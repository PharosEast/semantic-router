"""Live-tier orchestration tests: guard measurements, metric pairing, verdict flow."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest
import yaml
from recipe_lab.cli import main
from recipe_lab.contract import ContractError, ExperimentSpec
from recipe_lab.ledger import Ledger, LedgerRecord
from recipe_lab.live import (
    _pair_metrics,
    live_verdict_from_runs,
    measurements_from_rows,
    resolve_gates,
    verify_live,
)

from tests.fixtures import VALID_SPEC
from tests.test_contract import _write_splits_dataset

# ---- A3: the contract is the only source of guard floors ----

# VALID_SPEC's legacy workload hash (tests.fixtures); every FakeCli run doc
# carries it so the M1 dataset binding passes unless a test overrides it.
_LEGACY_HASH = "sha256:" + "0" * 64
_RUN_DOC = {
    "id": "run-x",
    "status": "completed",
    "manifest": {"dataset": {"sha256": _LEGACY_HASH}},
}


def test_measurements_from_rows_rejects_floors_and_unknown_keys():
    # a floor inside the measurement file is a contract bypass
    with pytest.raises(ContractError, match="floor"):
        measurements_from_rows(
            [{"name": "pii", "candidate": 0.97, "baseline": 0.96, "floor": 0.95}]
        )
    with pytest.raises(ContractError, match="non-measurement keys"):
        measurements_from_rows([{"name": "pii", "candidate": 0.97, "metric": "g"}])
    with pytest.raises(ContractError, match="name"):
        measurements_from_rows([{"candidate": 0.97}])
    with pytest.raises(ContractError, match="candidate"):
        measurements_from_rows([{"name": "pii", "candidate": "high"}])


def test_resolve_gates_numeric_floor_pass_and_breach():
    from recipe_lab.contract import Gate

    gates = [Gate(name="pii-recall-floor", metric="guard.pii.recall", floor=0.95)]
    guards, _undeclared = resolve_gates(
        gates, [{"name": "pii-recall-floor", "candidate": 0.97, "baseline": 0.5}]
    )
    assert guards[0].floor == 0.95 and guards[0].value == 0.97 and guards[0].passed
    guards, _ = resolve_gates(
        gates, [{"name": "pii-recall-floor", "candidate": 0.90, "baseline": 0.5}]
    )
    assert guards[0].passed is False


def test_resolve_gates_baseline_floor_uses_measured_baseline():
    gates = ExperimentSpec.from_dict(VALID_SPEC).gates
    guards, undeclared = resolve_gates(
        gates, [{"name": "pii-recall-floor", "candidate": 0.97, "baseline": 0.96}]
    )
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
    payload = {
        "baseline": {"m": {"capability.score": 0.80}},
        "candidate": {"capability.score": 0.86},
    }
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
    base, cand = _pair_metrics(
        {"capability.score": 0.80, "nested": {"capability.score": 0.86}},
        "capability.score",
        incomplete,
    )
    assert (base, cand) == (0.80, 0.86)
    assert incomplete == []


def test_pair_metrics_ambiguous_is_incomplete():
    incomplete: list[str] = []
    base, cand = _pair_metrics(
        {"capability.score": 0.8}, "capability.score", incomplete
    )
    assert base is None and cand is None
    assert "ambiguous" in incomplete[0]


def _spec():
    from recipe_lab.contract import ExperimentSpec

    return ExperimentSpec.from_dict(VALID_SPEC)


def test_live_verdict_pass_flow():
    class FakeCli:
        def show(self, run_id):
            return dict(_RUN_DOC)

        def compare(self, base, cand):
            return {
                "baseline": {"capability.score": 0.80},
                "candidate": {"capability.score": 0.86},
            }

    result = live_verdict_from_runs(
        spec=_spec(),
        cli=FakeCli(),
        baseline_run_id="run-a",
        candidate_run_id="run-b",
        # VALID_SPEC declares floor: baseline -> the measured baseline is the floor
        measurements=[
            {"name": "pii-recall-floor", "candidate": 0.97, "baseline": 0.96}
        ],
    )
    assert result["verdict"] == "PASS"
    assert result["gates"] == {"pii-recall-floor": "pass"}
    assert result["undeclared_measurements"] == []
    assert result["bench_refs"] == ["run-a", "run-b"]


def test_live_verdict_baseline_floor_breach_fails():
    class FakeCli:
        def show(self, run_id):
            return dict(_RUN_DOC)

        def compare(self, base, cand):
            return {
                "baseline": {"capability.score": 0.80},
                "candidate": {"capability.score": 0.86},
            }

    result = live_verdict_from_runs(
        spec=_spec(),
        cli=FakeCli(),
        baseline_run_id="run-a",
        candidate_run_id="run-b",
        measurements=[
            {"name": "pii-recall-floor", "candidate": 0.90, "baseline": 0.96}
        ],
    )
    assert result["verdict"] == "FAIL"
    assert result["gates"] == {"pii-recall-floor": "fail"}
    assert "guard floor violated" in result["reasons"][0]


def test_live_verdict_missing_measurement_is_inconclusive():
    class FakeCli:
        def show(self, run_id):
            return dict(_RUN_DOC)

        def compare(self, base, cand):
            return {
                "baseline": {"capability.score": 0.80},
                "candidate": {"capability.score": 0.86},
            }

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
        def show(self, run_id):
            return dict(_RUN_DOC)

        def compare(self, base, cand):
            return {
                "baseline": {"capability.score": 0.80},
                "candidate": {"capability.score": 0.86},
            }

    result = live_verdict_from_runs(
        spec=_spec(),
        cli=FakeCli(),
        baseline_run_id="run-a",
        candidate_run_id="run-b",
        measurements=[
            {"name": "pii-recall-floor", "candidate": 0.97, "baseline": 0.96},
            {"name": "extra", "candidate": 0.1},
        ],
    )
    assert result["verdict"] == "PASS"
    assert result["undeclared_measurements"] == ["extra"]
    assert any(
        "undeclared guard measurements" in reason for reason in result["reasons"]
    )


def test_live_verdict_empty_gates_is_explicit():
    spec_dict = dict(VALID_SPEC)
    spec_dict["gates"] = []

    class FakeCli:
        def show(self, run_id):
            return dict(_RUN_DOC)

        def compare(self, base, cand):
            return {
                "baseline": {"capability.score": 0.80},
                "candidate": {"capability.score": 0.86},
            }

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
        def show(self, run_id):
            return dict(_RUN_DOC)

        def compare(self, base, cand):
            return {"capability.score": 0.8}  # only one value -> ambiguous

    result = live_verdict_from_runs(
        spec=_spec(),
        cli=FakeCli(),
        baseline_run_id="run-a",
        candidate_run_id="run-b",
        measurements=[
            {"name": "pii-recall-floor", "candidate": 0.97, "baseline": 0.96}
        ],
    )
    assert result["verdict"] == "INCONCLUSIVE"
    assert result["gates"] == {"pii-recall-floor": "pass"}


# ---- A3: CLI rejects measurement floors with exit code 2 ----


def test_live_verdict_rejects_floor_in_measurement_file(tmp_path, capsys):
    spec_path = str(_write_splits_dataset(tmp_path))
    measurements = tmp_path / "measurements.jsonl"
    measurements.write_text(
        json.dumps(
            {
                "name": "pii-recall-floor",
                "candidate": 0.97,
                "baseline": 0.96,
                "floor": 0.9,
            }
        )
        + "\n",
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


# ---- I: verify - sr-bench-driven AB/BA orchestration ----


def _verify_spec():
    spec_dict = copy.deepcopy(VALID_SPEC)
    spec_dict["objective"]["constraints"] = [
        {
            "name": "cost-not-worse",
            "metric": "total_spend_usd",
            "op": "lte",
            "bound": "baseline",
            "tolerance_pct": 0.0,
        },
    ]
    return ExperimentSpec.from_dict(spec_dict)


class VerifyStub:
    """Scripted stand-in for SrbenchCli: deterministic run documents."""

    def __init__(
        self,
        *,
        preview_status: str = "completed",
        candidate_statuses: list[str] | None = None,
        baseline_statuses: list[str] | None = None,
        compare_values: dict | None = None,
        baseline_doc: dict | None = None,
    ):
        self.baseline_doc = baseline_doc or {
            "id": "run-base",
            "status": "completed",
            "manifest": {"manifest_id": "m1", "dataset": {"sha256": _LEGACY_HASH}},
        }
        self.preview_status = preview_status
        self.candidate_statuses = list(candidate_statuses or [])
        self.baseline_statuses = list(baseline_statuses or [])
        self.compare_values = dict(compare_values or {})
        self.preview_report_payload = {
            "summary": {
                "targets": [
                    {"selected_models": {"local/model-a": 7, "local/model-b": 5}}
                ]
            }
        }
        self.events: list[tuple] = []
        self.manifest_runs = 0
        self.compares: list[tuple] = []
        self._cand = 0

    def show(self, run_id):
        self.events.append(("show", run_id))
        return self.baseline_doc

    def candidate_plan(
        self, baseline_run_id, *, targets, mode="live", name=None, hypothesis=""
    ):
        self.events.append(("plan", mode))
        if mode == "preview":
            return {"id": "run-preview", "status": self.preview_status}
        self._cand += 1
        status = (
            self.candidate_statuses.pop(0) if self.candidate_statuses else "completed"
        )
        return {"id": f"run-cand-{self._cand}", "status": status}

    def report(self, run_id, out_path):
        self.events.append(("report", run_id))
        return self.preview_report_payload

    def run_manifest(self, manifest_path):
        self.manifest_runs += 1
        self.events.append(("manifest",))
        status = (
            self.baseline_statuses.pop(0) if self.baseline_statuses else "completed"
        )
        return {"id": f"run-base-trial-{self.manifest_runs}", "status": status}

    def compare(self, base, cand):
        self.compares.append((base, cand))
        return dict(self.compare_values)


def _compare_values(
    base: float, cand: float, spend: tuple[float, float] = (5.0, 4.8)
) -> dict:
    # default spend does not violate the cost-not-worse constraint (lte
    # baseline 5.0), so tests can target the metric/gate path in isolation
    return {
        "baseline": {"capability.score": base, "total_spend_usd": spend[0]},
        "candidate": {"capability.score": cand, "total_spend_usd": spend[1]},
    }


def test_verify_live_pass_flow(tmp_path):
    stub = VerifyStub(
        candidate_statuses=["completed", "completed"],
        baseline_statuses=["completed", "completed"],
        compare_values=_compare_values(0.80, 0.86),
    )
    result = verify_live(
        spec=_verify_spec(),
        cli=stub,
        baseline_run_id="run-base",
        targets=["mom-router"],
        repetitions=2,
        measurements=[
            {"name": "pii-recall-floor", "candidate": 0.97, "baseline": 0.96}
        ],
        work_dir=tmp_path,
    )
    assert result["verdict"] == "PASS"
    assert result["metrics"] == {"baseline": 0.80, "candidate": 0.86}
    assert result["gates"] == {"pii-recall-floor": "pass"}
    assert result["preview"]["distribution"] == {"local/model-a": 7, "local/model-b": 5}
    # AB/BA: preview first, then each pair runs baseline and candidate arms
    # in alternating order (schedule: base0, cand0, cand1, base1)
    assert stub.events == [
        ("show", "run-base"),
        ("plan", "preview"),
        ("report", "run-preview"),
        ("manifest",),
        ("plan", "live"),
        ("plan", "live"),
        ("manifest",),
    ]
    assert stub.manifest_runs == 2
    assert len(stub.compares) == 2
    assert result["abba_summary"]["baseline"]["median"] == 0.80
    assert result["abba_summary"]["candidate"]["median"] == 0.86
    assert (
        len(result["bench_refs"]) == 1 + 1 + 4
    )  # baseline + preview + both arms of both reps
    assert "run-base" in result["bench_refs"] and "run-preview" in result["bench_refs"]


def test_verify_live_incomplete_pair_is_inconclusive(tmp_path):
    stub = VerifyStub(
        candidate_statuses=["completed", "failed"],
        baseline_statuses=["completed", "completed"],
        compare_values=_compare_values(0.80, 0.86),
    )
    result = verify_live(
        spec=_verify_spec(),
        cli=stub,
        baseline_run_id="run-base",
        targets=["mom-router"],
        repetitions=2,
        measurements=[
            {"name": "pii-recall-floor", "candidate": 0.97, "baseline": 0.96}
        ],
        work_dir=tmp_path,
    )
    assert result["verdict"] == "INCONCLUSIVE"
    assert any("pair 1" in reason for reason in result["reasons"])  # reps are 0-indexed
    # only the complete pair is compared; no numbers are invented for the broken one
    assert len(stub.compares) == 1


def test_verify_live_baseline_arm_failure_is_inconclusive(tmp_path):
    stub = VerifyStub(
        baseline_statuses=["failed", "completed"],
        compare_values=_compare_values(0.80, 0.86),
    )
    result = verify_live(
        spec=_verify_spec(),
        cli=stub,
        baseline_run_id="run-base",
        targets=["mom-router"],
        repetitions=2,
        measurements=[
            {"name": "pii-recall-floor", "candidate": 0.97, "baseline": 0.96}
        ],
        work_dir=tmp_path,
    )
    assert result["verdict"] == "INCONCLUSIVE"
    assert any("baseline ended 'failed'" in reason for reason in result["reasons"])


def test_verify_live_constraint_median_over_pairs(tmp_path):
    # spend medians 5.0 vs 5.75 across two pairs -> constraint violated -> FAIL
    stub = VerifyStub(
        candidate_statuses=["completed", "completed"],
        baseline_statuses=["completed", "completed"],
    )
    stub.compare_values_by_call = {
        0: _compare_values(0.80, 0.86, spend=(5.0, 5.5)),
        1: _compare_values(0.80, 0.86, spend=(5.0, 6.0)),
    }

    def compare(base, cand):
        index = len(stub.compares)
        stub.compares.append((base, cand))
        return dict(stub.compare_values_by_call[index])

    stub.compare = compare
    result = verify_live(
        spec=_verify_spec(),
        cli=stub,
        baseline_run_id="run-base",
        targets=["mom-router"],
        repetitions=2,
        measurements=[
            {"name": "pii-recall-floor", "candidate": 0.97, "baseline": 0.96}
        ],
        work_dir=tmp_path,
    )
    assert result["verdict"] == "FAIL"
    evidence = result["constraints"][0]
    assert evidence["baseline"] == 5.0 and evidence["candidate"] == 5.75
    assert evidence["status"] == "violated"
    assert any(
        "constraint cost-not-worse violated" in reason for reason in result["reasons"]
    )


def test_verify_live_constraint_unmeasured_pair_stays_unevaluated(tmp_path):
    # one pair broken -> the constraint is measured in 1 of 2 pairs -> None
    stub = VerifyStub(
        candidate_statuses=["completed", "failed"],
        baseline_statuses=["completed", "completed"],
        compare_values=_compare_values(0.80, 0.86),
    )
    result = verify_live(
        spec=_verify_spec(),
        cli=stub,
        baseline_run_id="run-base",
        targets=["mom-router"],
        repetitions=2,
        measurements=[
            {"name": "pii-recall-floor", "candidate": 0.97, "baseline": 0.96}
        ],
        work_dir=tmp_path,
    )
    assert result["verdict"] == "INCONCLUSIVE"
    evidence = result["constraints"][0]
    assert evidence["baseline"] is None and evidence["candidate"] is None
    assert any("measured in 1 of 2 pairs" in reason for reason in result["reasons"])


def test_verify_live_preview_failure_is_diagnostic_only(tmp_path):
    stub = VerifyStub(
        preview_status="failed", compare_values=_compare_values(0.80, 0.86)
    )
    result = verify_live(
        spec=_verify_spec(),
        cli=stub,
        baseline_run_id="run-base",
        targets=["mom-router"],
        repetitions=1,
        measurements=[
            {"name": "pii-recall-floor", "candidate": 0.97, "baseline": 0.96}
        ],
        work_dir=tmp_path,
    )
    assert result["verdict"] == "PASS"  # a failed preview produces no quality numbers
    assert any("preview" in reason for reason in result["reasons"])
    assert "distribution" not in result["preview"]


def test_verify_live_baseline_must_be_completed(tmp_path):
    stub = VerifyStub(
        baseline_doc={
            "id": "run-base",
            "status": "failed",
            "manifest": {"manifest_id": "m1"},
        }
    )
    with pytest.raises(ContractError, match="not completed"):
        verify_live(
            spec=_verify_spec(),
            cli=stub,
            baseline_run_id="run-base",
            targets=["mom-router"],
            repetitions=1,
            measurements=[],
            work_dir=tmp_path,
        )


def test_verify_live_baseline_must_expose_manifest(tmp_path):
    stub = VerifyStub(baseline_doc={"id": "run-base", "status": "completed"})
    with pytest.raises(ContractError, match="does not expose its frozen manifest"):
        verify_live(
            spec=_verify_spec(),
            cli=stub,
            baseline_run_id="run-base",
            targets=["mom-router"],
            repetitions=1,
            measurements=[],
            work_dir=tmp_path,
        )


def test_verify_live_needs_at_least_one_repetition(tmp_path):
    with pytest.raises(ContractError, match="at least one repetition"):
        verify_live(
            spec=_verify_spec(),
            cli=VerifyStub(),
            baseline_run_id="run-base",
            targets=["mom-router"],
            repetitions=0,
            measurements=[],
            work_dir=tmp_path,
        )


def test_verify_cli_requires_targets(tmp_path, capsys):
    spec_path = str(_write_splits_dataset(tmp_path))
    candidate = tmp_path / "candidate.json"
    candidate.write_text(
        json.dumps(
            {
                "kind": "config",
                "ref": "config-rev",
                "change_summary": "t",
                "provenance": {},
            }
        ),
        encoding="utf-8",
    )
    assert (
        main(
            [
                "verify",
                "--spec",
                spec_path,
                "--baseline-run-id",
                "run-a",
                "--candidate",
                str(candidate),
            ]
        )
        == 2
    )
    assert "MoM target ids" in capsys.readouterr().err


# ---- G2: --insight-refs on live-verdict ----


def _seed_replay_ledger(tmp_path) -> tuple[str, str]:
    """An empty ledger with one replay record; returns (ledger_path, record_id)."""
    from recipe_lab.ledger import Ledger, LedgerRecord

    ledger_path = tmp_path / "seed-ledger.jsonl"
    stored = Ledger(ledger_path).append(
        LedgerRecord(
            round=1,
            ts="2026-10-29T00:00:00Z",
            candidate={
                "kind": "config",
                "ref": "seed.yaml",
                "change_summary": "s",
                "provenance": {"proposer": "p", "insight_refs": []},
            },
            tier="replay",
            baseline_ref="base@main",
            metrics={"routed_accuracy": 0.9},
            verdict="RANKED",
            dataset_hash="sha256:" + "0" * 64,
        )
    )
    return str(ledger_path), stored["record_id"]


def _live_verdict_args(spec_path: str, ledger: str | None = None) -> list[str]:
    args = [
        "live-verdict",
        "--spec",
        spec_path,
        "--baseline-run-id",
        "run-a",
        "--candidate-run-id",
        "run-b",
    ]
    if ledger:
        args += ["--ledger", ledger]
    return args


def test_live_verdict_insight_refs_four_forms_merge_and_dedupe(
    monkeypatch, tmp_path, capsys
):
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
        json.dumps(
            {
                "kind": "config",
                "ref": "cand-1.yaml",
                "change_summary": "s",
                "provenance": {"proposer": "agent", "insight_refs": ["bench:run-a"]},
            }
        ),
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
        lambda **kwargs: {
            "verdict": "PASS",
            "reasons": [],
            "improvement_pct": 1.0,
            "gates": {},
            "constraints": [],
            "bench_refs": ["run-a", "run-b"],
            "metrics": {"baseline": 0.5, "candidate": 0.6},
        },
    )
    args = [
        *_live_verdict_args(spec_path, str(ledger_path)),
        "--candidate",
        str(sidecar),
        "--insight-refs",
        refs,
    ]
    assert main(args) == 0
    rows = [
        json.loads(line)
        for line in ledger_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert rows[-1]["candidate"]["provenance"]["insight_refs"] == [
        "bench:run-a",
        record_id,
        f"attr:{attr_file}@sha256:{attr_hash}",
        f"diag:{diag_file}@sha256:{diag_hash}#threshold.domain_code_complex",
    ]


def test_live_verdict_unknown_record_ref_exits_2(tmp_path, capsys):
    seed_ledger, _ = _seed_replay_ledger(tmp_path)
    spec_path = str(_write_splits_dataset(tmp_path))
    rc = main(
        [
            *_live_verdict_args(spec_path, seed_ledger),
            "--insight-refs",
            "rec-000000000000",
        ]
    )
    assert rc == 2
    assert "not found in ledger" in capsys.readouterr().err


def test_live_verdict_hash_mismatch_exits_2(tmp_path, capsys):
    spec_path = str(_write_splits_dataset(tmp_path))
    attr_file = tmp_path / "attr.json"
    attr_file.write_text("{}", encoding="utf-8")
    bad_hash = "sha256:" + "f" * 64
    rc = main(
        [
            *_live_verdict_args(spec_path),
            "--insight-refs",
            f"attr:{attr_file}@{bad_hash}",
        ]
    )
    assert rc == 2
    assert "hash mismatch" in capsys.readouterr().err


# ---- E: economic constraints (live part) ----


def _spec_with_constraints(**overrides) -> ExperimentSpec:
    spec_dict = copy.deepcopy(VALID_SPEC)
    spec_dict["objective"]["constraints"] = [
        {
            "name": "cost-not-worse",
            "metric": "total_spend_usd",
            "op": "lte",
            "bound": "baseline",
            "tolerance_pct": 0.0,
        }
    ]
    spec_dict["objective"].update(overrides)
    return ExperimentSpec.from_dict(spec_dict)


def _compare_with_costs(candidate_cost):
    class FakeCli:
        def show(self, run_id):
            return dict(_RUN_DOC)

        def compare(self, base, cand):
            return {
                "baseline": {"capability.score": 0.80, "total_spend_usd": 5.0},
                "candidate": {
                    "capability.score": 0.86,
                    "total_spend_usd": candidate_cost,
                },
            }

    return FakeCli()


def test_live_verdict_constraint_within_baseline_passes():
    result = live_verdict_from_runs(
        spec=_spec_with_constraints(),
        cli=_compare_with_costs(5.0),
        baseline_run_id="run-a",
        candidate_run_id="run-b",
        measurements=[
            {"name": "pii-recall-floor", "candidate": 0.97, "baseline": 0.96}
        ],
    )
    assert result["verdict"] == "PASS"
    [constraint] = result["constraints"]
    assert constraint["status"] == "pass"
    assert constraint["bound_value"] == 5.0


def test_live_verdict_constraint_violated_fails():
    result = live_verdict_from_runs(
        spec=_spec_with_constraints(),
        cli=_compare_with_costs(6.0),
        baseline_run_id="run-a",
        candidate_run_id="run-b",
        measurements=[
            {"name": "pii-recall-floor", "candidate": 0.97, "baseline": 0.96}
        ],
    )
    assert result["verdict"] == "FAIL"
    assert "constraint cost-not-worse violated" in result["reasons"][0]
    assert result["constraints"][0]["status"] == "violated"


def test_live_verdict_unknown_cost_is_inconclusive_not_zero():
    # verified: sr-bench serializes unknown usage as null; an unevaluable
    # constraint is missing evidence, never a violation (and never free)
    result = live_verdict_from_runs(
        spec=_spec_with_constraints(),
        cli=_compare_with_costs(None),
        baseline_run_id="run-a",
        candidate_run_id="run-b",
        measurements=[
            {"name": "pii-recall-floor", "candidate": 0.97, "baseline": 0.96}
        ],
    )
    assert result["verdict"] == "INCONCLUSIVE"
    assert "not evaluable" in result["reasons"][0]
    assert result["constraints"][0]["status"] == "not_evaluated"


def test_guard_violation_is_listed_before_constraint_violation():
    result = live_verdict_from_runs(
        spec=_spec_with_constraints(),
        cli=_compare_with_costs(6.0),
        baseline_run_id="run-a",
        candidate_run_id="run-b",
        measurements=[{"name": "pii-recall-floor", "candidate": 0.5, "baseline": 0.96}],
    )
    assert result["verdict"] == "FAIL"
    assert result["reasons"][0].startswith("guard floor violated")
    assert any(
        "constraint cost-not-worse violated" in reason for reason in result["reasons"]
    )


def test_numeric_bound_and_tolerance_relax_in_permissive_direction():
    from recipe_lab.verdict import evaluate_verdict

    # tolerance 10% on an lte baseline bound of 5.0 -> 5.5 allowed
    evidence = [
        {
            "name": "c",
            "metric": "m",
            "op": "lte",
            "bound": "baseline",
            "tolerance_pct": 10.0,
            "baseline": 5.0,
            "candidate": 5.4,
        }
    ]
    result = evaluate_verdict(
        baseline_metric=0.8,
        candidate_metric=0.9,
        direction="maximize",
        min_improvement_pct=0.0,
        guards=[],
        incomplete_reasons=[],
        constraints=evidence,
    )
    assert result.verdict == "PASS" and result.constraints[0]["bound_value"] == 5.5
    # gte with 10% tolerance on a numeric bound of 0.9 -> 0.81 required
    evidence = [
        {
            "name": "c",
            "metric": "m",
            "op": "gte",
            "bound": 0.9,
            "tolerance_pct": 10.0,
            "baseline": 1.0,
            "candidate": 0.85,
        }
    ]
    result = evaluate_verdict(
        baseline_metric=0.8,
        candidate_metric=0.9,
        direction="maximize",
        min_improvement_pct=0.0,
        guards=[],
        incomplete_reasons=[],
        constraints=evidence,
    )
    assert result.verdict == "PASS" and result.constraints[0][
        "bound_value"
    ] == pytest.approx(0.81)


def test_live_cli_record_carries_constraint_evidence(monkeypatch, tmp_path, capsys):
    from recipe_lab import cli

    spec_path = _write_splits_dataset(tmp_path)
    spec_dict = yaml.safe_load(Path(spec_path).read_text(encoding="utf-8"))
    spec_dict["objective"]["constraints"] = [
        {
            "name": "cost-not-worse",
            "metric": "total_spend_usd",
            "op": "lte",
            "bound": "baseline",
            "tolerance_pct": 0.0,
        }
    ]
    spec_path = tmp_path / "spec-constraints.yaml"
    spec_path.write_text(yaml.safe_dump(spec_dict), encoding="utf-8")

    def fake_verdict(**kwargs):
        return {
            "verdict": "PASS",
            "reasons": [],
            "gates": {"pii-recall-floor": "pass"},
            "constraints": [
                {
                    "name": "cost-not-worse",
                    "metric": "total_spend_usd",
                    "op": "lte",
                    "bound": "baseline",
                    "tolerance_pct": 0.0,
                    "baseline": 5.0,
                    "candidate": 5.0,
                    "bound_value": 5.0,
                    "status": "pass",
                }
            ],
            "bench_refs": ["run-a", "run-b"],
            "metrics": {"baseline": 0.5, "candidate": 0.6},
        }

    monkeypatch.setattr(cli, "live_verdict_from_runs", fake_verdict)
    ledger_path = tmp_path / "ledger.jsonl"
    assert main([*_live_verdict_args(str(spec_path), str(ledger_path)), "--final"]) == 0
    rows = [
        json.loads(line)
        for line in ledger_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    constraint = rows[-1]["constraints"]["cost-not-worse"]
    assert constraint["status"] == "pass"
    assert constraint["candidate"] == 5.0


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
            "constraints": [],
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
    rows = [
        json.loads(line)
        for line in ledger_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert rows[-1]["split"] == "frozen_test"
    expected = (
        "sha256:"
        + hashlib.sha256((tmp_path / "frozen_test.jsonl").read_bytes()).hexdigest()
    )
    assert rows[-1]["dataset_hash"] == expected


# ---- M1: live verdicts bind the runs to the contract's own frozen split ----


def test_live_verdict_rejects_run_from_other_dataset():
    class FakeCli:
        def show(self, run_id):
            return {
                "id": run_id,
                "status": "completed",
                "manifest": {"dataset": {"sha256": "f" * 64}},
            }

        def compare(self, base, cand):
            raise AssertionError("compare must not run on a dataset mismatch")

    with pytest.raises(ContractError) as excinfo:
        live_verdict_from_runs(
            spec=_spec(),
            cli=FakeCli(),
            baseline_run_id="run-a",
            candidate_run_id="run-b",
            measurements=[],
        )
    assert "scored dataset" in str(excinfo.value)
    assert "baseline run run-a" in str(excinfo.value)


def test_live_verdict_rejects_run_without_dataset_identity():
    class FakeCli:
        def show(self, run_id):
            return {"id": run_id, "status": "completed", "manifest": {}}

        def compare(self, base, cand):
            raise AssertionError("compare must not run without a dataset identity")

    with pytest.raises(ContractError) as excinfo:
        live_verdict_from_runs(
            spec=_spec(),
            cli=FakeCli(),
            baseline_run_id="run-a",
            candidate_run_id="run-b",
            measurements=[],
        )
    assert "no dataset sha256" in str(excinfo.value)


def test_live_verdict_accepts_bare_hexdigest_dataset_hash():
    # sr-bench records a bare hexdigest; the A4 sha256: prefix is normalized.
    class FakeCli:
        def show(self, run_id):
            return {
                "id": run_id,
                "status": "completed",
                "manifest": {"dataset": {"sha256": "0" * 64}},
            }

        def compare(self, base, cand):
            return {
                "baseline": {"capability.score": 0.80},
                "candidate": {"capability.score": 0.86},
            }

    result = live_verdict_from_runs(
        spec=_spec(),
        cli=FakeCli(),
        baseline_run_id="run-a",
        candidate_run_id="run-b",
        measurements=[
            {"name": "pii-recall-floor", "candidate": 0.97, "baseline": 0.96}
        ],
    )
    assert result["verdict"] == "PASS"
    assert result["bench_refs"] == ["run-a", "run-b"]


def test_verify_live_rejects_baseline_from_other_dataset(tmp_path):
    stub = VerifyStub(
        baseline_doc={
            "id": "run-base",
            "status": "completed",
            "manifest": {"dataset": {"sha256": "f" * 64}},
        },
    )
    with pytest.raises(ContractError) as excinfo:
        verify_live(
            spec=_verify_spec(),
            cli=stub,
            baseline_run_id="run-base",
            targets=["mom-router"],
            repetitions=1,
            measurements=[],
            work_dir=tmp_path,
        )
    assert "scored dataset" in str(excinfo.value)


def test_legacy_verdict_record_uses_null_split(tmp_path):
    # M1: a legacy single-dataset contract records split: null + the
    # legacy_workload marker, while a three-way contract keeps its split.
    ledger_path = tmp_path / "ledger.jsonl"
    ledger = Ledger(ledger_path)
    record = LedgerRecord(
        round=1,
        ts="2026-10-31T00:00:00Z",
        candidate={"kind": "config", "ref": "run-b"},
        tier="live",
        baseline_ref="run-a",
        metrics={"capability.score": 0.86},
        verdict="PASS",
        dataset_hash=_LEGACY_HASH,
        gates={},
        bench_refs=["run-a", "run-b"],
        split=None,
        legacy_workload=True,
    )
    ledger.append(record)
    rows = [
        json.loads(line)
        for line in ledger_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert rows[0].get("split") is None
    assert rows[0]["legacy_workload"] is True
    roundtrip = LedgerRecord.from_dict(rows[0])
    assert roundtrip.legacy_workload is True and roundtrip.split is None
