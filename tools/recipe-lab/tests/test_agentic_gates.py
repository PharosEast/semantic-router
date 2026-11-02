"""AG7: agentic adjudication - stability gates, session-level R, cost.

Truth tables cover the four stability gates (pass / fail / unevaluated),
the session-level random-mixture closed form, the cost side-by-side
summary and the contract's `agentic_gates` field validation.
"""

from __future__ import annotations

import json

import pytest

from recipe_lab.agentic import (
    TOOL_LOOP_PHASE,
    AgenticGateResult,
    agentic_cost_summary,
    agentic_gate_results,
    session_random_mix_stats,
)
from recipe_lab.contract import ContractError, DEFAULT_AGENTIC_GATES, Workload
from recipe_lab.verdict import evaluate_verdict
from recipe_lab.session_attribution import analyze_sessions


def _continuity(
    *,
    tool_loop_switches: int = 0,
    mean: float | None = 0.0,
    multi_request_tasks: int = 2,
) -> dict:
    by_phase = {TOOL_LOOP_PHASE: tool_loop_switches} if tool_loop_switches else {}
    return {
        "session_mode": "session_aware",
        "multi_request_tasks": multi_request_tasks,
        "model_switches": tool_loop_switches,
        "mean_switches_per_task": mean,
        "model_switches_by_phase": by_phase,
    }


def _results(
    gates,
    *,
    candidate_continuity=None,
    baseline_continuity=None,
    candidate_events=None,
    baseline_events=None,
):
    return {
        (result.name): result
        for result in agentic_gate_results(
            gates,
            candidate_continuity=candidate_continuity,
            baseline_continuity=baseline_continuity,
            candidate_events=candidate_events,
            baseline_events=baseline_events,
        )
    }


def _verdict(gate_results):
    return evaluate_verdict(
        baseline_metric=0.80,
        candidate_metric=0.86,
        direction="maximize",
        min_improvement_pct=1.0,
        guards=[],
        incomplete_reasons=[],
        agentic_gates=gate_results,
    )


# ---- truth table: the four stability gates ----


def test_tool_loop_switch_gate_truth_table():
    gates = {"max_tool_loop_switches": 0}
    clean = _results(gates, candidate_continuity=_continuity())
    assert clean["max_tool_loop_switches"].passed
    assert _verdict(list(clean.values())).verdict == "PASS"

    switching = _results(
        gates, candidate_continuity=_continuity(tool_loop_switches=2)
    )
    assert not switching["max_tool_loop_switches"].passed
    failed = _verdict(list(switching.values()))
    assert failed.verdict == "FAIL"
    assert any("max_tool_loop_switches" in reason for reason in failed.reasons)

    # a missing continuity section is missing evidence, never a pass
    missing = _results(gates, candidate_continuity=None)
    assert not missing["max_tool_loop_switches"].evaluated
    undecided = _verdict(list(missing.values()))
    assert undecided.verdict == "INCONCLUSIVE"
    assert any(
        "agentic stability gate evaluation incomplete" in reason
        for reason in undecided.reasons
    )


def test_selection_rejected_gate_truth_table():
    gates = {"max_selection_rejected": 0}
    passed = _results(gates, candidate_events={"selection_failed": 0})
    assert passed["max_selection_rejected"].passed

    rejected = _results(gates, candidate_events={"selection_failed": 1})
    assert not rejected["max_selection_rejected"].passed
    assert _verdict(list(rejected.values())).verdict == "FAIL"

    # no replay records provided -> the gate stays unevaluated
    unevaluated = _results(gates, candidate_events=None)
    assert not unevaluated["max_selection_rejected"].evaluated
    assert _verdict(list(unevaluated.values())).verdict == "INCONCLUSIVE"


def test_tool_loop_interrupted_gate_resolves_baseline():
    gates = {"max_tool_loop_interrupted": "baseline"}
    # candidate at or under the baseline interruption count passes
    better = _results(
        {"max_tool_loop_interrupted": "baseline"},
        candidate_events={"tool_loop_interrupted": 1},
        baseline_events={"tool_loop_interrupted": 2},
    )
    assert better["max_tool_loop_interrupted"].bound == 2.0
    assert better["max_tool_loop_interrupted"].passed
    assert _verdict(list(better.values())).verdict == "PASS"

    worse = _results(
        gates,
        candidate_events={"tool_loop_interrupted": 3},
        baseline_events={"tool_loop_interrupted": 2},
    )
    assert not worse["max_tool_loop_interrupted"].passed
    assert _verdict(list(worse.values())).verdict == "FAIL"

    # the baseline arm's evidence is missing -> unevaluated, never a pass
    no_baseline = _results(gates, candidate_events={"tool_loop_interrupted": 0})
    assert not no_baseline["max_tool_loop_interrupted"].evaluated
    assert _verdict(list(no_baseline.values())).verdict == "INCONCLUSIVE"


def test_mean_switches_per_task_gate_truth_table():
    gates = {"max_mean_switches_per_task": 1.0}
    passed = _results(gates, candidate_continuity=_continuity(mean=0.5))
    assert passed["max_mean_switches_per_task"].passed

    failed = _results(gates, candidate_continuity=_continuity(mean=1.5))
    assert not failed["max_mean_switches_per_task"].passed
    assert _verdict(list(failed.values())).verdict == "FAIL"

    # sr-bench reports null when no task was multi-request: unevaluated
    unobservable = _results(gates, candidate_continuity=_continuity(mean=None))
    assert not unobservable["max_mean_switches_per_task"].evaluated
    assert _verdict(list(unobservable.values())).verdict == "INCONCLUSIVE"


def test_all_default_gates_evaluated_together():
    continuity = _continuity()
    events = analyze_sessions([])["event_counts"]
    results = agentic_gate_results(
        DEFAULT_AGENTIC_GATES,
        candidate_continuity=continuity,
        baseline_continuity=continuity,
        candidate_events=events,
        baseline_events=events,
    )
    assert {r.name for r in results} == set(DEFAULT_AGENTIC_GATES)
    assert _verdict(results).verdict == "PASS"


def test_agentic_gate_result_shape():
    gate = AgenticGateResult(name="g", value=1.0, bound=1.0)
    assert gate.evaluated and gate.passed
    assert gate.to_dict() == {"name": "g", "value": 1.0, "bound": 1.0}


# ---- session-level R arm: the closed form ----


def test_random_mix_delta_is_the_share_weighted_closed_form():
    # two tasks, both references judged both; shares 0.5/0.5
    candidate = {"t1": 1.0, "t2": 0.0}
    references = {
        "strong": {"t1": 1.0, "t2": 1.0},
        "weak": {"t1": 0.0, "t2": 0.0},
    }
    stats = session_random_mix_stats(
        candidate, references, {"strong": 1.0, "weak": 1.0}
    )
    assert stats is not None
    assert stats["paired_tasks"] == 2
    assert stats["shares"] == {"strong": 0.5, "weak": 0.5}
    assert stats["candidate_accuracy"] == pytest.approx(0.5)
    assert stats["expected_r_accuracy"] == pytest.approx(0.5)
    assert stats["delta"] == pytest.approx(0.0)

    # the bootstrap mean converges to the analytic delta (C8's cross-check)
    assert stats["bootstrap_mean"] == pytest.approx(stats["delta"], abs=0.02)


def test_random_mix_strictly_above_r_matches_hand_computation():
    candidate = {f"t{i}": 1.0 for i in range(10)}
    references = {
        "strong": {f"t{i}": 1.0 for i in range(10)},
        "weak": {f"t{i}": 0.0 for i in range(10)},
    }
    stats = session_random_mix_stats(
        candidate, references, {"strong": 1.0, "weak": 3.0}
    )
    assert stats is not None
    # R answers every task with strong at 1/4, weak at 3/4 -> E[R] = 0.25
    assert stats["expected_r_accuracy"] == pytest.approx(0.25)
    assert stats["delta"] == pytest.approx(0.75)


def test_random_mix_requires_full_pairing():
    # pairing is the intersection with every weighted reference arm, and
    # the closed form runs over that intersection (the C8 convention)
    candidate = {"t1": 1.0, "t2": 1.0}
    references = {"strong": {"t1": 1.0}}  # t2 unjudged by the reference
    stats = session_random_mix_stats(candidate, references, {"strong": 1.0})
    assert stats is not None
    assert stats["paired_tasks"] == 1
    assert stats["delta"] == pytest.approx(0.0)
    assert session_random_mix_stats({}, {"strong": {"t1": 1.0}}, {"strong": 1.0}) is None
    assert (
        session_random_mix_stats(
            {"t1": 1.0}, {"strong": {"t2": 1.0}}, {"strong": 1.0}
        )
        is None
    )
    # a zero-weight reference is not needed for the pairing
    stats = session_random_mix_stats(
        candidate,
        {"strong": {"t1": 1.0, "t2": 1.0}},
        {"strong": 1.0, "weak": 0.0},
    )
    assert stats is not None
    assert stats["paired_tasks"] == 2


# ---- cost side by side ----


def _compare_with_costs(base_total, cand_total, base_neutral, cand_neutral):
    return {
        "comparisons": [
            {
                "baseline_total_cost_usd": base_total,
                "candidate_total_cost_usd": cand_total,
                "cache_neutral_baseline_cost_usd": base_neutral,
                "cache_neutral_candidate_cost_usd": cand_neutral,
                "total_cost_comparison_reason": "both arms measured",
                "cache_neutral_cost_basis": "counterfactual",
            }
        ]
    }


def test_cost_summary_sums_pairs_and_recomputes_savings():
    summary = agentic_cost_summary(
        [
            _compare_with_costs(5.0, 4.0, 5.0, 5.0),
            _compare_with_costs(5.0, 4.0, 5.0, 5.0),
        ]
    )
    assert summary["complete"] is True
    assert summary["actual_cost_usd"] == {"baseline": 10.0, "candidate": 8.0}
    assert summary["cache_neutral_cost_usd"] == {
        "baseline": 10.0,
        "candidate": 10.0,
    }
    assert summary["saving_percent"]["total"] == pytest.approx(20.0)
    assert summary["saving_percent"]["cache_neutral"] == pytest.approx(0.0)
    assert summary["notes"]["total_cost_reason"] == "both arms measured"


def test_cost_summary_keeps_unknown_sides_unknown():
    comparison = _compare_with_costs(5.0, 4.0, 5.0, 5.0)
    comparison["comparisons"][0]["cache_neutral_candidate_cost_usd"] = None
    summary = agentic_cost_summary([comparison])
    assert summary["complete"] is False
    assert summary["actual_cost_usd"] == {"baseline": None, "candidate": None}
    assert any(
        "cache_neutral_candidate_cost_usd is unknown" in r
        for r in summary["reasons"]
    )


def test_cost_summary_reports_missing_comparison_entry():
    summary = agentic_cost_summary([{"comparisons": []}])
    assert summary["complete"] is False
    assert any("no single comparisons[] entry" in r for r in summary["reasons"])


# ---- truncation (W3): early-finish tasks make the evidence unreliable ----


def test_truncated_tasks_turn_the_verdict_inconclusive():
    from recipe_lab.verdict import evidence_quality_reasons

    reasons = evidence_quality_reasons(
        "candidate",
        truncation_rate=0.5,
        missing_result_rate=None,
        no_answer_rate=None,
        max_truncation_pct=5.0,
        max_no_answer_pct=5.0,
    )
    verdict = evaluate_verdict(
        baseline_metric=0.80,
        candidate_metric=0.86,
        direction="maximize",
        min_improvement_pct=1.0,
        guards=[],
        incomplete_reasons=reasons,
    )
    assert verdict.verdict == "INCONCLUSIVE"
    assert any("truncation rate" in reason for reason in verdict.reasons)


# ---- contract: the agentic_gates workload field ----


def _agentic_raw(**overrides):
    raw = {
        "profile": "quick",
        "kind": "agentic",
        "benchmark": "terminal-bench-2.1",
        "repetitions": 1,
        "role": "tune",
        "split_rule": "group by task; seed 42",
        "splits": {
            split: {"manifest": "cases.jsonl", "content_hash": "a" * 64}
            for split in ("calib", "dev", "frozen_test")
        },
    }
    raw.update(overrides)
    return raw


def test_contract_agentic_gates_round_trip():
    workload = Workload.from_dict(
        _agentic_raw(
            agentic_gates={"max_mean_switches_per_task": 0.5},
        )
    )
    assert workload.agentic_gates == {"max_mean_switches_per_task": 0.5}
    resolved = workload.resolved_agentic_gates()
    assert resolved["max_mean_switches_per_task"] == 0.5
    assert resolved["max_tool_loop_switches"] == 0
    assert resolved["max_tool_loop_interrupted"] == "baseline"
    assert workload.to_dict()["agentic_gates"] == {
        "max_mean_switches_per_task": 0.5
    }
    # the defaults round-trip nothing
    default = Workload.from_dict(_agentic_raw())
    assert default.agentic_gates is None
    assert "agentic_gates" not in default.to_dict()
    assert default.resolved_agentic_gates() == DEFAULT_AGENTIC_GATES


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"kind": "mcq", "agentic_gates": {"max_selection_rejected": 0}}, "agentic"),
        ({"agentic_gates": {}}, "non-empty"),
        ({"agentic_gates": {"unknown_gate": 1}}, "unknown gates"),
        ({"agentic_gates": {"max_tool_loop_switches": -1}}, ">= 0"),
        (
            {"agentic_gates": {"max_selection_rejected": "baseline"}},
            'may pin "baseline"',
        ),
    ],
)
def test_contract_agentic_gates_validation(overrides, message):
    with pytest.raises(ContractError, match=message):
        Workload.from_dict(_agentic_raw(**overrides))


def test_contract_serialization_round_trips_through_json():
    workload = Workload.from_dict(
        _agentic_raw(agentic_gates={"max_tool_loop_interrupted": 2})
    )
    restored = json.loads(json.dumps(workload.to_dict()))
    reloaded = Workload.from_dict(restored)
    assert reloaded.agentic_gates == {"max_tool_loop_interrupted": 2}
