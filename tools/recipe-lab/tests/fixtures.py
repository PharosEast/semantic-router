"""Shared synthetic fixtures for recipe-lab tests (no network, no stack)."""

from __future__ import annotations

from recipe_lab.contract import OraclePolicy

POLICY = OraclePolicy.from_dict(
    {
        "math": "local/model-math",
        "code": "local/model-code",
        "reasoning": "local/model-large",
    }
)

VALID_SPEC: dict = {
    "api_version": "recipe-lab/v1alpha1",
    "objective": {
        "metric_key": "capability.score",
        "direction": "maximize",
        "min_improvement_pct": 1.0,
        "replay_metric": "routed_accuracy",
    },
    "workload": {
        "manifest": "datasets/dev-mmlu-pro.yaml",
        "content_hash": "sha256:" + "0" * 64,
        "profile": "quick",
    },
    "attribution": {
        "oracle_policy": {"math": "local/model-math", "code": "local/model-code"},
    },
    "gates": [
        {"name": "pii-recall-floor", "metric": "guard.pii.recall", "floor": "baseline"},
    ],
    "budget": {"rounds": 5, "max_minutes_per_trial": 60},
    "stack": {
        "mode": "external",
        "endpoint": "http://127.0.0.1:8899",
        "baseline_ref": "config-revision@main",
    },
}

# B: the preferred three-way workload form (splits must be disjoint)
VALID_SPLITS_SPEC: dict = {
    **VALID_SPEC,
    "workload": {
        "profile": "standard",
        "split_rule": "group by source_id; seed 42; 40/30/30",
        "splits": {
            "calib": {"manifest": "calib.jsonl", "content_hash": "sha256:" + "1" * 64},
            "dev": {"manifest": "dev.jsonl", "content_hash": "sha256:" + "2" * 64},
            "frozen_test": {"manifest": "frozen_test.jsonl", "content_hash": "sha256:" + "3" * 64},
        },
    },
}


def make_case(case_id: str, truth: str, matched_decision: str, routed: str, answer_ok, family=None) -> dict:
    return {
        "case_id": case_id,
        "truth": truth,
        "matched_decision": matched_decision,
        "routed": routed,
        "answer_ok": answer_ok,
        **({"family": family} if family is not None else {}),
    }


ATTRIBUTION_CASES: list[dict] = [
    # decision mismatch: expected math but the decision matched code (F1)
    make_case("case-mm", "math", "code", "local/model-code", False, family="d3"),
    make_case("case-e2", "math", "math", "local/model-code", False, family="d3"),
    make_case("case-e3", "math", "math", "local/model-math", False, family="d3"),
    make_case("case-sub", "math", "math", "local/model-code", True, family="d3"),
    make_case("case-correct", "math", "math", "local/model-math", True, family="d3"),
    make_case("case-noev", "math", "math", "local/model-math", None, family="d3"),
    make_case("case-gap", "unknown-label", "unknown-label", "local/model-math", True),
]


# ---- sub-issue 7: BUG-3 pilot-shaped diagnostic rows (D) ----

def pilot_difficulty_values() -> list[float]:
    """344 ascending difficulty scores shaped like the BUG-3 pilot (appendix
    A of sub-issue-7): range [0.08, 1.87], 74 values below 1, nothing at or
    above 2.5, and p75 anchored exactly at 1.75 so the calibrated proposal
    matches the PR #5568 golden result deterministically."""
    values = [round(0.08 + i * (0.91 / 73), 4) for i in range(74)]
    values += [round(1.0 + i * (0.74 / 182), 4) for i in range(183)]
    values += [1.75, 1.75]  # indices 257/258: the p75 interpolation anchors
    values += [round(1.7501 + i * ((1.87 - 1.7501) / 84), 4) for i in range(85)]
    assert len(values) == 344
    return sorted(values)


def pilot_rows() -> list[dict]:
    """Unlabeled diagnostic rows for the agent-recipe fixture: the strong
    lanes (domain_code_complex / complex_general) never fire, every row
    carries a difficulty score plus a per-choice domain confidence."""
    decisions = ["simple_math_fast_path", "domain_code", "medium_general", "simple_general"]
    rows = []
    for index, value in enumerate(pilot_difficulty_values()):
        rows.append(
            {
                "request_id": f"req-{index:04d}",
                "matched_decision": decisions[index % 4],
                "signal_values": {
                    "decision:difficulty": value,
                    "decision:domain:code": round(0.80 + (index % 20) * 0.01, 4),
                },
                "signal_confidences": {},
            }
        )
    return rows


def pilot_rows_jsonl() -> str:
    import json

    return "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in pilot_rows())
