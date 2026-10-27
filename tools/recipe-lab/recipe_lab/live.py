"""Live-tier orchestration (Sub-issue 3).

The live tier's evaluation engine is sr-bench; this module only wires the
evidence together:

1. the caller (or an upstream script) runs sr-bench for baseline and
   candidate and holds two run IDs;
2. `live_verdict_from_runs` compares them, reads the objective metric,
   applies the three-state verdict with the guard floors, and returns the
   live-tier evidence block (bench_refs included - the ledger record is
   self-sufficient for Sub-issue 4's report replay).

Guard results come from model_eval guard tooling produced offline (the
tool does not own guard evaluation); they arrive as GuardResult values.
"""

from __future__ import annotations

from typing import Any

from .contract import ContractError, ExperimentSpec
from .srbench import SrbenchCli, extract_metric_values
from .verdict import GuardResult, evaluate_verdict


def guards_from_rows(rows: list[dict[str, Any]]) -> list[GuardResult]:
    """Build GuardResult values from a guards JSONL (name/metric/value/floor)."""
    guards: list[GuardResult] = []
    for index, row in enumerate(rows, 1):
        for key in ("name", "metric"):
            if not isinstance(row.get(key), str) or not row[key].strip():
                raise ContractError(f"guards row {index}: {key} must be a non-empty string")
        value = row.get("value")
        floor = row.get("floor")
        guards.append(
            GuardResult(
                name=row["name"],
                metric=row["metric"],
                value=None if value is None else float(value),
                floor=None if floor is None else float(floor),
            )
        )
    return guards


def _pair_metrics(
    comparison: Any,
    metric_key: str,
    incomplete: list[str],
) -> tuple[float | None, float | None]:
    """Pair baseline/candidate metric values from a compare payload.

    Preferred shape: {"baseline": {...}, "candidate": {...}} sections. If the
    payload is flatter, collect every numeric occurrence; exactly two are
    read in traversal order, anything else is ambiguous evidence and becomes
    INCONCLUSIVE rather than a guess.
    """
    if isinstance(comparison, dict):
        base_section = comparison.get("baseline", comparison.get("baseline_targets"))
        cand_section = comparison.get("candidate", comparison.get("candidate_targets"))
        if isinstance(base_section, (dict, list)) and isinstance(cand_section, (dict, list)):
            base_values = extract_metric_values(base_section, metric_key)
            cand_values = extract_metric_values(cand_section, metric_key)
            return (
                base_values[0] if base_values else None,
                cand_values[0] if cand_values else None,
            )
    values = extract_metric_values(comparison, metric_key)
    if len(values) == 2:
        return values[0], values[1]
    incomplete.append(
        f"ambiguous compare payload for {metric_key!r}: {len(values)} numeric values found"
    )
    return None, None


def live_verdict_from_runs(
    *,
    spec: ExperimentSpec,
    cli: SrbenchCli,
    baseline_run_id: str,
    candidate_run_id: str,
    guards: list[GuardResult],
    incomplete_reasons: list[str] | None = None,
) -> dict[str, Any]:
    """Compare two sr-bench runs and decide the live-tier verdict."""
    comparison = cli.compare(baseline_run_id, candidate_run_id)
    incomplete = list(incomplete_reasons or [])
    baseline_metric, candidate_metric = _pair_metrics(comparison, spec.objective.metric_key, incomplete)

    verdict = evaluate_verdict(
        baseline_metric=baseline_metric,
        candidate_metric=candidate_metric,
        direction=spec.objective.direction,
        min_improvement_pct=spec.objective.min_improvement_pct,
        guards=guards,
        incomplete_reasons=incomplete,
    )
    return {
        "verdict": verdict.verdict,
        "reasons": verdict.reasons,
        "improvement_pct": verdict.improvement_pct,
        "metrics": {
            "baseline": baseline_metric,
            "candidate": candidate_metric,
        },
        "gates": {g.name: ("pass" if g.passed else ("incomplete" if not g.evaluated else "fail")) for g in guards},
        "bench_refs": [baseline_run_id, candidate_run_id],
    }
