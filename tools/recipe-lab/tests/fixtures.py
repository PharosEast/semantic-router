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
    "evaluation": {"search_tier": "replay", "final_tier": "live"},
}


def make_case(case_id: str, truth: str, pred: str, routed: str, answer_ok, family=None) -> dict:
    return {
        "case_id": case_id,
        "truth": truth,
        "pred": pred,
        "routed": routed,
        "answer_ok": answer_ok,
        **({"family": family} if family is not None else {}),
    }


ATTRIBUTION_CASES: list[dict] = [
    make_case("case-e1", "math", "code", "local/model-code", False, family="d3"),
    make_case("case-e2", "math", "math", "local/model-code", False, family="d3"),
    make_case("case-e3", "math", "math", "local/model-math", False, family="d3"),
    make_case("case-sub", "math", "math", "local/model-code", True, family="d3"),
    make_case("case-correct", "math", "math", "local/model-math", True, family="d3"),
    make_case("case-noev", "math", "math", "local/model-math", None, family="d3"),
    make_case("case-gap", "unknown-label", "unknown-label", "local/model-math", True),
]
