"""Three-state verdict tests: priority rules, guard veto, honesty."""

from __future__ import annotations

from recipe_lab.verdict import GuardResult, evaluate_verdict


def guard(name: str, value: float | None, floor: float | None = 0.95) -> GuardResult:
    return GuardResult(name=name, metric=f"guard.{name}", value=value, floor=floor)


BASE = {
    "baseline_metric": 0.80,
    "direction": "maximize",
    "min_improvement_pct": 1.0,
    "guards": [],
}


def test_pass_when_improvement_meets_bar():
    result = evaluate_verdict(**BASE, candidate_metric=0.85, incomplete_reasons=[])
    assert result.verdict == "PASS"
    assert result.improvement_pct == 6.25


def test_fail_when_below_bar():
    result = evaluate_verdict(**BASE, candidate_metric=0.805, incomplete_reasons=[])
    assert result.verdict == "FAIL"
    assert "below min" in result.reasons[0]


def test_inconclusive_outranks_fail():
    # a broken run must never be recorded as "the recipe is worse"
    result = evaluate_verdict(
        baseline_metric=0.80,
        candidate_metric=0.70,  # much worse
        direction="maximize",
        min_improvement_pct=1.0,
        guards=[],
        incomplete_reasons=["trial 2 timed out", "snapshot hash changed"],
    )
    assert result.verdict == "INCONCLUSIVE"
    assert len(result.reasons) == 2


def test_guard_veto_forces_fail_despite_improvement():
    result = evaluate_verdict(
        baseline_metric=0.80,
        candidate_metric=0.90,
        direction="maximize",
        min_improvement_pct=1.0,
        guards=[guard("pii-recall", 0.90, 0.95)],
        incomplete_reasons=[],
    )
    assert result.verdict == "FAIL"
    assert "guard floor violated" in result.reasons[0]


def test_unevaluated_guard_is_incomplete_not_fail():
    result = evaluate_verdict(
        baseline_metric=0.80,
        candidate_metric=0.90,
        direction="maximize",
        min_improvement_pct=1.0,
        guards=[guard("pii-recall", None)],
        incomplete_reasons=[],
    )
    assert result.verdict == "INCONCLUSIVE"
    assert "pii-recall" in result.reasons[0]


def test_missing_metrics_are_inconclusive():
    for kwargs in ({"candidate_metric": None}, {"baseline_metric": None}):
        merged = {
            **BASE,
            **kwargs,
            "candidate_metric": kwargs.get("candidate_metric", 0.9),
            "baseline_metric": kwargs.get("baseline_metric", 0.8),
        }
        result = evaluate_verdict(**merged, incomplete_reasons=[])
        assert result.verdict == "INCONCLUSIVE"


def test_minimize_direction():
    result = evaluate_verdict(
        baseline_metric=10.0,
        candidate_metric=9.0,
        direction="minimize",
        min_improvement_pct=5.0,
        guards=[],
        incomplete_reasons=[],
    )
    assert result.verdict == "PASS"  # 10% lower latency


def test_guard_passing_keeps_pass():
    result = evaluate_verdict(
        baseline_metric=0.80,
        candidate_metric=0.85,
        direction="maximize",
        min_improvement_pct=1.0,
        guards=[guard("pii-recall", 0.97), guard("jailbreak-recall", 0.99)],
        incomplete_reasons=[],
    )
    assert result.verdict == "PASS"
