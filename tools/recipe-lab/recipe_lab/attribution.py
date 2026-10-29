"""Misroute attribution: error taxonomy over a labeled frozen dataset.

Classes (recipe-lab design §6, per signal family):

- DECISION_MISMATCH  matched decision != expected decision, judged without
                     signal-level truth (the decision-level class; E1 is not
                     used for this until F2 introduces signal-level truth)
- E1_SIGNAL          signal model label != ground truth; requires
                     signal-level truth, which arrives with F2 (multi-signal
                     oracle), so it is defined but not produced yet
- E2_POLICY          decision right, routed model != oracle-policy
                     expectation, answer wrong
- E3_CAPABILITY      routing exactly as intended, answer still wrong
- SUBOPTIMAL         routing deviated from the policy but the answer was
                     still correct (counted separately, never as an error)
- NO_EVIDENCE        answer correctness unknown (no saved answer /
                     interrupted run); evidence-first: incomplete evidence is
                     never counted as an error
- CORRECT            routed as intended and answered correctly
- POLICY_GAP         ground-truth label missing from the declared oracle
                     policy

Input records (one JSON object per line) follow a minimal schema; the
router_replay adapter produces exactly this shape:

  {"case_id": str, "truth": str, "matched_decision": str, "routed": str,
   "family": "vela_classify" | "vela_decisions" | "d3" | ...,
   "answer_ok": true | false | null}

The row field is named matched_decision (F1): with string-only truth, a
mismatched decision is decision_mismatch, never E1_signal. Records written
before F1 remain readable: a top-level "pred" field is accepted as
matched_decision.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .contract import ContractError, OraclePolicy

E1_SIGNAL = "E1_signal"
DECISION_MISMATCH = "decision_mismatch"
E2_POLICY = "E2_policy"
E3_CAPABILITY = "E3_capability"
SUBOPTIMAL = "suboptimal_routing"
NO_EVIDENCE = "no_evidence"
CORRECT = "correct"
POLICY_GAP = "policy_gap"

ERROR_CLASSES = (E1_SIGNAL, DECISION_MISMATCH, E2_POLICY, E3_CAPABILITY)
ALL_CLASSES = (
    CORRECT,
    E1_SIGNAL,
    DECISION_MISMATCH,
    E2_POLICY,
    E3_CAPABILITY,
    SUBOPTIMAL,
    NO_EVIDENCE,
    POLICY_GAP,
)

REQUIRED_FIELDS = ("case_id", "truth", "matched_decision", "routed", "answer_ok")


@dataclass(frozen=True)
class CaseInput:
    case_id: str
    truth: str
    matched_decision: str
    routed: str
    answer_ok: bool | None
    family: str | None = None

    @classmethod
    def from_dict(cls, raw: dict, lineno: int) -> "CaseInput":
        if "matched_decision" not in raw and "pred" in raw:
            # pre-F1 field name; kept readable for compatibility (F1)
            raw = {**raw, "matched_decision": raw["pred"]}
        for name in REQUIRED_FIELDS:
            if name not in raw:
                raise ContractError(f"attribution record line {lineno} is missing required field: {name}")
        answer_ok = raw["answer_ok"]
        if answer_ok is not None and not isinstance(answer_ok, bool):
            raise ContractError(
                f"attribution record line {lineno}: answer_ok must be true, false or null (no evidence), "
                f"got {answer_ok!r}"
            )
        for name in ("case_id", "truth", "matched_decision", "routed"):
            if not isinstance(raw[name], str) or not raw[name].strip():
                raise ContractError(f"attribution record line {lineno}: {name} must be a non-empty string")
        family = raw.get("family")
        if family is not None and (not isinstance(family, str) or not family.strip()):
            raise ContractError(f"attribution record line {lineno}: family must be a non-empty string when present")
        return cls(
            case_id=raw["case_id"],
            truth=raw["truth"],
            matched_decision=raw["matched_decision"],
            routed=raw["routed"],
            answer_ok=answer_ok,
            family=family,
        )


def classify_case(case: "CaseInput | dict", policy: OraclePolicy) -> dict[str, Any]:
    """Classify one case into the attribution taxonomy (branch-explicit).

    Accepts a CaseInput or a raw record dict."""
    if isinstance(case, dict):
        case = CaseInput.from_dict(case, 0)
    expected = policy.expected(case.truth)
    row: dict[str, Any] = {
        "case_id": case.case_id,
        "truth": case.truth,
        "matched_decision": case.matched_decision,
        "routed": case.routed,
        "expected": expected,
        "family": case.family,
        "answer_ok": case.answer_ok,
    }
    if expected is None:
        row["error_class"] = POLICY_GAP
        return row
    if case.matched_decision != case.truth:
        # no signal-level truth in this format: a mismatched decision is not
        # an E1 signal error (F1; E1 needs F2's signal-level truth)
        row["error_class"] = DECISION_MISMATCH
        return row
    if case.answer_ok is None:
        row["error_class"] = NO_EVIDENCE
        return row
    routed_as_expected = case.routed == expected
    if case.answer_ok:
        row["error_class"] = CORRECT if routed_as_expected else SUBOPTIMAL
    else:
        row["error_class"] = E3_CAPABILITY if routed_as_expected else E2_POLICY
    return row


def load_records(path: str | Path) -> list[CaseInput]:
    path = Path(path)
    if not path.is_file():
        raise ContractError(f"attribution records file not found: {path}")
    cases: list[CaseInput] = []
    with path.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ContractError(f"attribution records line {lineno} is not valid JSON: {exc}") from exc
            cases.append(CaseInput.from_dict(raw, lineno))
    return cases


def attribute(cases: list[CaseInput], policy: OraclePolicy, dataset_hash: str = "") -> list[dict[str, Any]]:
    rows = [classify_case(c, policy) for c in cases]
    for row in rows:
        if dataset_hash:
            row["dataset_hash"] = dataset_hash
    return rows


def summarize(rows: list[dict[str, Any]], top_k: int = 5) -> dict[str, Any]:
    counts = Counter(row["error_class"] for row in rows)
    answered_classes = (CORRECT, E2_POLICY, E3_CAPABILITY, SUBOPTIMAL)
    n_answered = sum(counts[c] for c in answered_classes)
    n_correct = counts[CORRECT]
    accuracy = (n_correct / n_answered) if n_answered else None

    e1_rows = [r for r in rows if r["error_class"] == E1_SIGNAL]
    confusion = Counter((r["truth"], r["matched_decision"]) for r in e1_rows)
    mm_rows = [r for r in rows if r["error_class"] == DECISION_MISMATCH]
    mm_confusion = Counter((r["truth"], r["matched_decision"]) for r in mm_rows)
    e2_rows = [r for r in rows if r["error_class"] == E2_POLICY]
    deviations = Counter((r["routed"], r["expected"]) for r in e2_rows)
    e3_by_family = Counter(r.get("family") or "unknown" for r in rows if r["error_class"] == E3_CAPABILITY)
    e1_by_family = Counter(r.get("family") or "unknown" for r in e1_rows)

    return {
        "total": len(rows),
        "class_counts": {cls: counts.get(cls, 0) for cls in ALL_CLASSES},
        "answered": n_answered,
        "correct": n_correct,
        "accuracy": accuracy,
        "e1_by_family": dict(e1_by_family),
        "e3_by_family": dict(e3_by_family),
        "e1_confusion_top": [
            {"truth": t, "matched_decision": m, "count": n} for (t, m), n in confusion.most_common(top_k)
        ],
        "decision_mismatch_top": [
            {"truth": t, "matched_decision": m, "count": n} for (t, m), n in mm_confusion.most_common(top_k)
        ],
        "e2_deviations_top": [
            {"routed": r, "expected": e, "count": n} for (r, e), n in deviations.most_common(top_k)
        ],
    }


def render_markdown(summary: dict[str, Any], dataset_hash: str = "") -> str:
    counts = summary["class_counts"]
    accuracy = summary["accuracy"]
    accuracy_text = f"{accuracy * 100:.2f}%" if accuracy is not None else "N/A (no answered cases)"
    lines = [
        "# Misroute attribution report",
        "",
        f"- dataset_hash: `{dataset_hash or 'n/a'}`",
        f"- total cases: {summary['total']}",
        f"- routed-outcome accuracy: **{accuracy_text}**"
        f" ({summary['correct']}/{summary['answered']} answered cases)",
        "",
        "## Class distribution",
        "",
        "| class | count |",
        "|---|---|",
    ]
    for cls in ALL_CLASSES:
        lines.append(f"| {cls} | {counts[cls]} |")
    lines += ["", "## E1 by signal family", ""]
    if summary["e1_by_family"]:
        for family, n in sorted(summary["e1_by_family"].items()):
            hint = "threshold neighborhoods" if family == "vela_classify" else "question-text edits (signal-question engineering, later proposer)"
            lines.append(f"- `{family}`: {n} cases -> directional hint: {hint}")
    else:
        lines.append("- none")
    lines += ["", "## E1 confusion pairs (top)", ""]
    if summary["e1_confusion_top"]:
        for item in summary["e1_confusion_top"]:
            lines.append(f"- `{item['truth']}` -> `{item['matched_decision']}`: {item['count']}")
    else:
        lines.append("- none (E1 needs signal-level truth, F2)")
    lines += ["", "## decision_mismatch confusion pairs (top)", ""]
    if summary["decision_mismatch_top"]:
        for item in summary["decision_mismatch_top"]:
            lines.append(
                f"- expected `{item['truth']}` but decision matched `{item['matched_decision']}`: {item['count']}"
            )
    else:
        lines.append("- none")
    lines += ["", "## E2 routing deviations (top)", ""]
    if summary["e2_deviations_top"]:
        for item in summary["e2_deviations_top"]:
            lines.append(f"- routed `{item['routed']}` but policy expects `{item['expected']}`: {item['count']}")
    else:
        lines.append("- none")
    lines += ["", "## E3 by signal family", ""]
    if summary["e3_by_family"]:
        for family, n in sorted(summary["e3_by_family"].items()):
            lines.append(f"- `{family}`: {n} cases (routing not at fault; candidate new-signal area)")
    else:
        lines.append("- none")
    return "\n".join(lines) + "\n"


def write_jsonl(rows: list[dict[str, Any]], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    return path
