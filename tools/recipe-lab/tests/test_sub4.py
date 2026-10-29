"""Sub-issue 4 tests: ledger-only report replay, untrusted next header,
provenance manifest emission."""

from __future__ import annotations

import json

from recipe_lab.next import render_next_context
from recipe_lab.provenance import build_run_manifest
from recipe_lab.report import render_run_report

REPLAY = {
    "round": 1, "ts": "t", "tier": "replay", "verdict": "RANKED",
    "candidate": {"kind": "config", "ref": "c1.yaml", "provenance": {"proposer": "threshold-search"}},
    "metrics": {"accuracy": 0.85}, "dataset_hash": "sha256:aa",
    "coverage_warning": False,
}
LIVE = {
    "round": 2, "ts": "t", "tier": "live", "verdict": "PASS",
    "candidate": {"kind": "config", "ref": "c1.yaml", "provenance": {"proposer": "threshold-search"}},
    "metrics": {"accuracy": 0.9}, "dataset_hash": "sha256:aa",
    "gates": {"pii": "pass"}, "bench_refs": ["run-a", "run-b"],
    "coverage_warning": False,
}


def test_report_replays_ledger_and_flags_missing_evidence():
    report = render_run_report([REPLAY, LIVE])
    assert "**PASS**" in report and "RANKED" in report
    assert "replayed from the recipe-lab ledger only" in report.lower()
    broken = dict(LIVE, gates=None)
    report = render_run_report([broken])
    assert "Replay inconsistencies" in report


def test_next_context_carries_untrusted_header():
    context = render_next_context([REPLAY, LIVE])
    assert context.startswith("# recipe-lab knowledge context")
    assert "untrusted reference material" in context
    assert "c1.yaml" in context
    assert "Discipline reminders" in context


def test_provenance_manifest_from_ledger(tmp_path):
    ledger = tmp_path / "ledger.jsonl"
    ledger.write_text(
        json.dumps(REPLAY) + "\n" + json.dumps(LIVE) + "\n", encoding="utf-8"
    )
    manifest = build_run_manifest(str(ledger), [REPLAY, LIVE])
    assert manifest["kind"] == "recipe-lab-run"
    assert manifest["dataset_ref"]["content_hash"] == "sha256:aa"
    assert manifest["evidence"]["bench_refs"] == ["run-a", "run-b"]
    assert manifest["best_replay_candidate"]["ref"] == "c1.yaml"
    assert manifest["rounds"]["verdict_counts"] == {"PASS": 1, "RANKED": 1}
    # ledger content hash is verifiable
    import hashlib

    expected = f"sha256:{hashlib.sha256(ledger.read_bytes()).hexdigest()}"
    assert manifest["evidence"]["ledger"]["content_hash"] == expected
