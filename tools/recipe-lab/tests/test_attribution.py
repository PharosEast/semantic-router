"""Attribution tests: the E1/E2/E3 taxonomy and the evidence-first rule."""

from __future__ import annotations

import json

import pytest
from recipe_lab.attribution import (
    CORRECT,
    DECISION_MISMATCH,
    E1_SIGNAL,
    E2_POLICY,
    E3_CAPABILITY,
    NO_EVIDENCE,
    POLICY_GAP,
    SUBOPTIMAL,
    CaseInput,
    attribute,
    classify_case,
    load_records,
    render_markdown,
    summarize,
    write_jsonl,
)
from recipe_lab.contract import ContractError

from tests.fixtures import ATTRIBUTION_CASES, POLICY, make_case


def test_all_classes_classified():
    rows = attribute(
        [CaseInput.from_dict(c, i) for i, c in enumerate(ATTRIBUTION_CASES)],
        POLICY,
        dataset_hash="sha256:" + "2" * 64,
    )
    by_id = {r["case_id"]: r for r in rows}
    # string-only truth: a mismatched decision is decision_mismatch, not E1 (F1)
    assert by_id["case-mm"]["error_class"] == DECISION_MISMATCH
    assert by_id["case-e2"]["error_class"] == E2_POLICY
    assert by_id["case-e3"]["error_class"] == E3_CAPABILITY
    assert by_id["case-sub"]["error_class"] == SUBOPTIMAL
    assert by_id["case-correct"]["error_class"] == CORRECT
    assert by_id["case-noev"]["error_class"] == NO_EVIDENCE
    assert by_id["case-gap"]["error_class"] == POLICY_GAP
    # dataset hash is stamped on every row
    assert all(r["dataset_hash"] == "sha256:" + "2" * 64 for r in rows)


def test_e1_never_produced_from_string_truth():
    # F1: without signal-level truth, no record can be classified as E1
    rows = attribute(
        [CaseInput.from_dict(c, i) for i, c in enumerate(ATTRIBUTION_CASES)],
        POLICY,
    )
    assert all(r["error_class"] != E1_SIGNAL for r in rows)
    assert rows[0]["matched_decision"] == "code"  # renamed row field is present


def test_legacy_pred_field_still_readable():
    # records written before F1 use "pred"; they must keep loading
    row = classify_case(
        {
            "case_id": "x",
            "truth": "math",
            "pred": "code",
            "routed": "m",
            "answer_ok": False,
        },
        POLICY,
    )
    assert row["error_class"] == DECISION_MISMATCH
    assert row["matched_decision"] == "code"


def test_evidence_first_no_evidence_with_deviation():
    # answer unknown + routing deviates -> NO_EVIDENCE (never counted as an error)
    row = classify_case(
        make_case("x", "math", "math", "local/model-code", None), POLICY
    )
    assert row["error_class"] == NO_EVIDENCE


def test_policy_gap_precedes_everything():
    row = classify_case(
        make_case("x", "no-such-label", "also-wrong", "m", None), POLICY
    )
    assert row["error_class"] == POLICY_GAP
    assert row["expected"] is None


def test_summary_math():
    summary = summarize(
        attribute(
            [CaseInput.from_dict(c, i) for i, c in enumerate(ATTRIBUTION_CASES)],
            POLICY,
        )
    )
    assert summary["total"] == 7
    assert summary["answered"] == 4  # correct + e2 + e3 + suboptimal
    assert summary["correct"] == 1
    assert summary["accuracy"] == pytest.approx(0.25)
    assert summary["class_counts"][DECISION_MISMATCH] == 1
    assert summary["class_counts"][E1_SIGNAL] == 0  # E1 needs signal-level truth (F1)
    # decision mismatch pair: truth=math matched as code
    assert {"truth": "math", "matched_decision": "code", "count": 1} in summary[
        "decision_mismatch_top"
    ]
    assert summary["e1_confusion_top"] == []
    # E2 deviation: routed code, expected model-math
    assert {
        "routed": "local/model-code",
        "expected": "local/model-math",
        "count": 1,
    } in summary["e2_deviations_top"]
    assert summary["e1_by_family"] == {}


def test_report_renders_family_aware_hints():
    rows = attribute(
        [CaseInput.from_dict(c, i) for i, c in enumerate(ATTRIBUTION_CASES)],
        POLICY,
    )
    text = render_markdown(summarize(rows), dataset_hash="sha256:abc")
    assert "Misroute attribution report" in text
    assert "sha256:abc" in text
    assert "decision_mismatch confusion pairs" in text
    assert "`math` but decision matched `code`" in text
    assert "E1 by signal family" in text
    # E1 needs signal-level truth (F2): empty for now, with a hint instead
    assert "- none (E1 needs signal-level truth, F2)" in text
    assert "E2 routing deviations" in text


def test_write_jsonl_roundtrip(tmp_path):
    rows = attribute(
        [CaseInput.from_dict(c, i) for i, c in enumerate(ATTRIBUTION_CASES)],
        POLICY,
        dataset_hash="sha256:" + "3" * 64,
    )
    out = write_jsonl(rows, tmp_path / "attribution.jsonl")
    loaded = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert loaded == rows


def test_load_records_rejects_bad_answer_ok(tmp_path):
    path = tmp_path / "records.jsonl"
    path.write_text(
        json.dumps(
            {
                "case_id": "a",
                "truth": "math",
                "pred": "math",
                "routed": "m",
                "answer_ok": "yes",
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ContractError, match="answer_ok"):
        load_records(path)


def test_load_records_rejects_missing_field(tmp_path):
    path = tmp_path / "records.jsonl"
    path.write_text(json.dumps({"case_id": "a", "truth": "math"}), encoding="utf-8")
    with pytest.raises(ContractError, match="required field"):
        load_records(path)
