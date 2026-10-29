"""Sub-issue 4 tests: ledger-only report replay, untrusted next header,
provenance manifest emission."""

from __future__ import annotations

import json

from recipe_lab.next import render_next_context
from recipe_lab.provenance import build_run_manifest
from recipe_lab.report import render_run_report

REPLAY = {
    "record_id": "rec-111111111111",
    "round": 1, "ts": "t", "tier": "replay", "verdict": "RANKED",
    "candidate": {"kind": "config", "ref": "c1.yaml", "provenance": {"proposer": "threshold-search"}},
    "metrics": {"accuracy": 0.85}, "dataset_hash": "sha256:aa",
    "coverage_warning": False,
}
LIVE = {
    "record_id": "rec-222222222222",
    "round": 2, "ts": "t", "tier": "live", "verdict": "PASS",
    "candidate": {"kind": "config", "ref": "c1.yaml", "provenance": {"proposer": "threshold-search"}},
    "metrics": {"accuracy": 0.9}, "dataset_hash": "sha256:aa",
    "gates": {"pii": "pass"}, "bench_refs": ["run-a", "run-b"],
    "coverage_warning": False,
}


def test_report_replays_ledger_and_flags_missing_evidence():
    report = render_run_report([REPLAY, LIVE])
    assert "**PASS**" in report and "RANKED" in report
    assert "rec-111111111111" in report  # G1: every row cites its record id
    assert "replayed from the recipe-lab ledger only" in report.lower()
    broken = dict(LIVE, gates=None)
    report = render_run_report([broken])
    assert "Replay inconsistencies" in report


def test_report_flags_mixed_dataset_hashes():
    other = dict(REPLAY, round=3, dataset_hash="sha256:bb")
    report = render_run_report([REPLAY, LIVE, other])
    assert "Cross-dataset warning" in report
    assert "NOT directly comparable" in report
    assert "sha256:bb" in report


def test_promotion_draft_from_pass_record():
    from recipe_lab.report import render_promotion_draft

    draft = render_promotion_draft(LIVE)
    assert "DRAFT" in draft and "human review" in draft.lower()
    assert "rec-222222222222" in draft  # G1: the draft cites the ledger record
    assert "c1.yaml" in draft and "run-a" in draft
    assert "Standard holdout" in draft


def test_next_context_carries_untrusted_header():
    context = render_next_context([REPLAY, LIVE], "accuracy")
    assert context.startswith("# recipe-lab knowledge context")
    assert "untrusted reference material" in context
    assert "c1.yaml" in context
    assert "rec-111111111111" in context  # G1: the tail cites record ids
    assert "Discipline reminders" in context


def test_next_best_replay_uses_metric_not_recency():
    # A2: the buggy selection let the last RANKED record win regardless of
    # metric; an earlier candidate with a higher metric must win instead.
    early_high = dict(REPLAY, record_id="rec-aaaaaaaaaaaa", round=1, metrics={"accuracy": 0.9})
    late_low = dict(REPLAY, record_id="rec-bbbbbbbbbbbb", round=2, metrics={"accuracy": 0.7})
    context = render_next_context([early_high, late_low, LIVE], "accuracy")
    assert "rec-aaaaaaaaaaaa" in context
    assert "rec-bbbbbbbbbbbb" not in context.split("## Best replay-tier candidate")[1].split("##")[0]


def test_next_best_replay_excludes_flagged_records():
    # A2: coverage_warning records and timeout trials never rank.
    good = dict(REPLAY, record_id="rec-cccccccccccc", round=1, metrics={"accuracy": 0.8})
    flagged = dict(REPLAY, record_id="rec-dddddddddddd", round=2, metrics={"accuracy": 0.99}, coverage_warning=True)
    timeout = dict(REPLAY, record_id="rec-eeeeeeeeeeee", round=3, metrics={"accuracy": 0.98}, status="timeout")
    context = render_next_context([flagged, timeout, good, LIVE], "accuracy")
    best_section = context.split("## Best replay-tier candidate")[1].split("##")[0]
    assert "rec-cccccccccccc" in best_section
    assert "rec-dddddddddddd" not in best_section
    assert "rec-eeeeeeeeeeee" not in best_section


def test_next_without_replay_records(tmp_path):
    from recipe_lab.next import write_next_context

    ledger = tmp_path / "ledger.jsonl"
    ledger.write_text(json.dumps(LIVE) + "\n", encoding="utf-8")
    output = tmp_path / "ctx.md"
    context = write_next_context(str(ledger), str(output), metric_key="accuracy")
    assert "none yet: run the replay tier first" in context
    assert output.read_text(encoding="utf-8") == context


def test_provenance_manifest_from_ledger(tmp_path):
    ledger = tmp_path / "ledger.jsonl"
    ledger.write_text(
        json.dumps(REPLAY) + "\n" + json.dumps(LIVE) + "\n", encoding="utf-8"
    )
    manifest = build_run_manifest(str(ledger), [REPLAY, LIVE])
    assert manifest["kind"] == "recipe-lab-run"
    assert manifest["dataset_ref"]["content_hash"] == "sha256:aa"
    assert manifest["evidence"]["ledger"]["record_ids"] == ["rec-111111111111", "rec-222222222222"]
    assert manifest["evidence"]["bench_refs"] == ["run-a", "run-b"]
    assert manifest["best_replay_candidate"]["ref"] == "c1.yaml"
    assert manifest["rounds"]["verdict_counts"] == {"PASS": 1, "RANKED": 1}
    # ledger content hash is verifiable
    import hashlib

    expected = f"sha256:{hashlib.sha256(ledger.read_bytes()).hexdigest()}"
    assert manifest["evidence"]["ledger"]["content_hash"] == expected
