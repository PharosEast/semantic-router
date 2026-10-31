"""Three-state verdict tests: priority rules, guard veto, honesty."""

from __future__ import annotations

import math

import pytest
from recipe_lab.verdict import (
    GuardResult,
    PairedStats,
    compute_paired_stats,
    evaluate_verdict,
)


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


# ---- M2: paired statistics decide from the confidence interval ----


def paired(
    *,
    paired_cases: int = 400,
    quality_delta: float = 0.05,
    ci: tuple[float, float] | None = (0.03, 0.07),
) -> PairedStats:
    half = paired_cases // 4
    return PairedStats(
        paired_cases=paired_cases,
        quality_delta=quality_delta,
        quality_delta_ci95=ci,
        quality_delta_bootstrap_ci95=ci,
        wins=half + 1,
        losses=max(half - 1, 0),
        ties=paired_cases - 2 * half,
    )


def test_paired_pass_when_ci_lower_bound_meets_bar():
    result = evaluate_verdict(
        baseline_metric=0.80,
        candidate_metric=0.85,
        direction="maximize",
        min_improvement_pct=0.0,
        guards=[],
        incomplete_reasons=[],
        stats=paired(quality_delta=0.05, ci=(0.03, 0.07)),
        min_cases=200,
        min_improvement_pp=2.0,
    )
    assert result.verdict == "PASS"
    assert "lower bound 3.00pp" in result.reasons[0]


def test_paired_pass_boundary_exact_lower_bound():
    result = evaluate_verdict(
        baseline_metric=0.80,
        candidate_metric=0.85,
        direction="maximize",
        min_improvement_pct=0.0,
        guards=[],
        incomplete_reasons=[],
        stats=paired(quality_delta=0.05, ci=(0.02, 0.07)),
        min_cases=200,
        min_improvement_pp=2.0,
    )
    assert result.verdict == "PASS"


def test_paired_fail_when_ci_upper_bound_below_bar():
    result = evaluate_verdict(
        baseline_metric=0.80,
        candidate_metric=0.82,
        direction="maximize",
        min_improvement_pct=0.0,
        guards=[],
        incomplete_reasons=[],
        stats=paired(quality_delta=0.01, ci=(0.005, 0.015)),
        min_cases=200,
        min_improvement_pp=2.0,
    )
    assert result.verdict == "FAIL"
    assert "upper bound 1.50pp below min 2.00pp" in result.reasons[0]


def test_paired_inconclusive_when_interval_spans_threshold():
    result = evaluate_verdict(
        baseline_metric=0.80,
        candidate_metric=0.83,
        direction="maximize",
        min_improvement_pct=0.0,
        guards=[],
        incomplete_reasons=[],
        stats=paired(quality_delta=0.03, ci=(0.01, 0.05)),
        min_cases=200,
        min_improvement_pp=2.0,
    )
    assert result.verdict == "INCONCLUSIVE"
    assert "spans the threshold 2.00pp" in result.reasons[0]


def test_paired_inconclusive_when_interval_spans_zero():
    # pp bar 0: the interval covering 0 means the improvement could be noise
    result = evaluate_verdict(
        baseline_metric=0.80,
        candidate_metric=0.81,
        direction="maximize",
        min_improvement_pct=0.0,
        guards=[],
        incomplete_reasons=[],
        stats=paired(quality_delta=0.005, ci=(-0.02, 0.03)),
        min_cases=200,
        min_improvement_pp=0.0,
    )
    assert result.verdict == "INCONCLUSIVE"
    assert "spans the threshold 0.00pp" in result.reasons[0]


def test_paired_inconclusive_when_insufficient_cases():
    result = evaluate_verdict(
        baseline_metric=17 / 60,
        candidate_metric=18 / 60,
        direction="maximize",
        min_improvement_pct=0.0,
        guards=[],
        incomplete_reasons=[],
        stats=paired(paired_cases=60, quality_delta=1 / 60),
        min_cases=200,
        min_improvement_pp=1.0,
    )
    assert result.verdict == "INCONCLUSIVE"
    assert "insufficient paired sample: 60 paired cases < min_cases 200" in (
        result.reasons[0]
    )


def test_paired_inconclusive_with_relative_threshold():
    # a relative bar cannot be applied to an absolute interval
    result = evaluate_verdict(
        baseline_metric=0.80,
        candidate_metric=0.85,
        direction="maximize",
        min_improvement_pct=1.0,
        guards=[],
        incomplete_reasons=[],
        stats=paired(),
        min_cases=200,
        min_improvement_pp=None,
    )
    assert result.verdict == "INCONCLUSIVE"
    assert "min_improvement_pp" in result.reasons[0]


def test_paired_minimize_direction_is_inconclusive():
    result = evaluate_verdict(
        baseline_metric=10.0,
        candidate_metric=9.0,
        direction="minimize",
        min_improvement_pct=0.0,
        guards=[],
        incomplete_reasons=[],
        stats=paired(),
        min_cases=200,
        min_improvement_pp=1.0,
    )
    assert result.verdict == "INCONCLUSIVE"
    assert "maximize objectives only" in result.reasons[0]


def test_paired_inconclusive_without_confidence_interval():
    result = evaluate_verdict(
        baseline_metric=0.80,
        candidate_metric=0.85,
        direction="maximize",
        min_improvement_pct=0.0,
        guards=[],
        incomplete_reasons=[],
        stats=paired(ci=None),
        min_cases=200,
        min_improvement_pp=2.0,
    )
    assert result.verdict == "INCONCLUSIVE"
    assert "no confidence interval" in result.reasons[0]


def test_pp_bar_without_paired_stats_is_inconclusive():
    # a pp bar is defined on the paired CI; point estimates cannot carry it
    result = evaluate_verdict(
        baseline_metric=0.80,
        candidate_metric=0.90,
        direction="maximize",
        min_improvement_pct=0.0,
        guards=[],
        incomplete_reasons=[],
        stats=None,
        min_improvement_pp=2.0,
    )
    assert result.verdict == "INCONCLUSIVE"
    assert "requires paired CI evidence" in result.reasons[0]


def test_guard_veto_outranks_paired_sample_shortfall():
    result = evaluate_verdict(
        baseline_metric=0.80,
        candidate_metric=0.85,
        direction="maximize",
        min_improvement_pct=0.0,
        guards=[guard("pii-recall", 0.90, 0.95)],
        incomplete_reasons=[],
        stats=paired(paired_cases=60),
        min_cases=200,
        min_improvement_pp=2.0,
    )
    assert result.verdict == "FAIL"
    assert "guard floor violated" in result.reasons[0]


# ---- M2: compute_paired_stats (sr-bench field shape) ----


def test_compute_paired_stats_plain_matches_hoeffding():
    diffs = [1.0] * 120 + [0.0] * 80
    s = compute_paired_stats(diffs)
    expected_delta = 0.6
    radius = math.sqrt(2 * math.log(2 / 0.05) / 200)
    assert s.paired_cases == 200
    assert s.quality_delta == pytest.approx(expected_delta)
    assert s.quality_delta_ci95[0] == pytest.approx(expected_delta - radius)
    assert s.quality_delta_ci95[1] == pytest.approx(expected_delta + radius)
    assert (s.wins, s.losses, s.ties) == (120, 0, 80)


def test_compute_paired_stats_grouped_weighted_macro():
    groups = {
        "a": [1.0] * 50 + [0.0] * 50,  # n=100, mean 0.5
        "b": [1.0] * 10 + [0.0] * 10,  # n=20, mean 0.5
    }
    weights = {"a": 3.0, "b": 1.0}  # normalized: a=0.75, b=0.25
    diffs = list(groups["a"]) + list(groups["b"])
    s = compute_paired_stats(diffs, groups=groups, weights=weights)
    radius = math.sqrt(2 * math.log(2 / 0.05) * (0.75**2 / 100 + 0.25**2 / 20))
    assert s.paired_cases == 120
    assert s.quality_delta == pytest.approx(0.5)
    assert s.quality_delta_ci95[0] == pytest.approx(0.5 - radius)
    assert s.quality_delta_ci95[1] == pytest.approx(0.5 + radius)
    assert (s.wins, s.losses, s.ties) == (60, 0, 60)


def test_compute_paired_stats_bootstrap_deterministic_and_srbench_aligned():
    diffs = [1.0, 0.0] * 100
    s1 = compute_paired_stats(diffs)
    s2 = compute_paired_stats(diffs)
    assert s1 == s2  # same fixed seed -> identical intervals
    # degenerate diffs pin the bootstrap quantiles (2000 samples: [49]/[1949])
    sure = compute_paired_stats([1.0] * 200)
    assert sure.quality_delta_bootstrap_ci95 == (1.0, 1.0)


def test_compute_paired_stats_rejects_bad_inputs():
    with pytest.raises(ValueError, match="at least one"):
        compute_paired_stats([])
    with pytest.raises(ValueError, match="together"):
        compute_paired_stats([1.0], groups={"a": [1.0]})
    with pytest.raises(ValueError, match="missing"):
        compute_paired_stats([1.0], groups={"a": [1.0]}, weights={})
    with pytest.raises(ValueError, match="cover"):
        compute_paired_stats([1.0, 1.0], groups={"a": [1.0]}, weights={"a": 1.0})
