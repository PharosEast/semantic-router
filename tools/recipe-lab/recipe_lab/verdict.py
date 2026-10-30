"""Three-state verdict for the live tier (Sub-issue 3).

Ported from InferRSI's evaluation semantics, adapted to sr-bench evidence:

- INCONCLUSIVE: evidence is incomplete (an interrupted trial, missing
  metrics, an unfinished guard evaluation, an unevaluable constraint, a
  changed snapshot hash). This outranks FAIL - a broken run must never be
  recorded as "the recipe is worse".
- FAIL: evidence complete but the candidate does not beat the baseline by
  at least `min_improvement_pct`, OR a guard floor is violated (safety
  veto), OR an economic constraint is violated (E).
- PASS: evidence complete, guard floors hold, constraints hold,
  improvement reaches the bar.

The verdict is a pure function of its inputs: deterministic, unit-testable,
no I/O.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

VERDICT_PASS = "PASS"
VERDICT_FAIL = "FAIL"
VERDICT_INCONCLUSIVE = "INCONCLUSIVE"
VERDICTS = (VERDICT_PASS, VERDICT_FAIL, VERDICT_INCONCLUSIVE)


@dataclass(frozen=True)
class GuardResult:
    """One guard floor check (e.g. pii-recall-floor)."""

    name: str
    metric: str
    value: float | None  # None = the guard evaluation did not complete
    floor: float | None

    @property
    def evaluated(self) -> bool:
        return self.value is not None and self.floor is not None

    @property
    def passed(self) -> bool:
        return self.evaluated and self.value >= self.floor


@dataclass(frozen=True)
class VerdictResult:
    verdict: str
    reasons: list[str] = field(default_factory=list)
    improvement_pct: float | None = None
    constraints: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict,
            "reasons": list(self.reasons),
            "improvement_pct": self.improvement_pct,
            "constraints": [dict(c) for c in self.constraints],
        }


def _improvement_pct(baseline: float, candidate: float, direction: str) -> float:
    if direction == "minimize":
        if baseline == 0:
            return 0.0 if candidate == 0 else -100.0
        return (baseline - candidate) / baseline * 100.0
    if baseline == 0:
        return 0.0 if candidate == 0 else 100.0
    return (candidate - baseline) / baseline * 100.0


def _resolve_constraint(constraint: dict[str, Any]) -> dict[str, Any]:
    """Status and resolved bound for one constraint evidence entry (E).

    `constraint` carries the declared fields (name/metric/op/bound/
    tolerance_pct) plus the measured `baseline` and `candidate` values
    (None = unknown, e.g. sr-bench serializes unknown spend as null -
    unknown usage must never count as zero). A "baseline" bound resolves
    against the measured baseline value; tolerance_pct relaxes the bound
    in the permissive direction. Not-measurable constraints are
    unevaluated, never failed.
    """
    name = constraint["name"]
    op = constraint["op"]
    bound = constraint["bound"]
    tolerance = float(constraint.get("tolerance_pct", 0.0))
    baseline = constraint.get("baseline")
    candidate = constraint.get("candidate")
    out = dict(constraint)
    out["bound_value"] = None
    out["status"] = "not_evaluated"
    if candidate is None or (bound == "baseline" and baseline is None):
        return out
    bound_value = float(baseline) if bound == "baseline" else float(bound)
    # lte allows a worst case above the bound, gte below it
    bound_value = (
        bound_value * (1 + tolerance / 100.0)
        if op == "lte"
        else bound_value * (1 - tolerance / 100.0)
    )
    out["bound_value"] = bound_value
    violated = candidate > bound_value if op == "lte" else candidate < bound_value
    out["status"] = "violated" if violated else "pass"
    if violated:
        out["reason"] = (
            f"constraint {name} violated (candidate {candidate:.6f} {'>' if op == 'lte' else '<'} bound {bound_value:.6f})"
        )
    return out


def evaluate_verdict(
    *,
    baseline_metric: float | None,
    candidate_metric: float | None,
    direction: str,
    min_improvement_pct: float,
    guards: list[GuardResult],
    incomplete_reasons: list[str],
    constraints: list[dict[str, Any]] | None = None,
) -> VerdictResult:
    """Decide PASS / FAIL / INCONCLUSIVE from live-tier evidence.

    `incomplete_reasons` lists everything that makes the evidence unreliable
    (timeout, missing metric, snapshot mismatch, ...). Guards that did not
    complete and constraints that cannot be evaluated also count as
    incomplete evidence, not as failures.

    Guards are safety vetoes and outrank constraints (E): when both a guard
    and a constraint are violated the guard reasons are listed first.
    """
    reasons = list(incomplete_reasons)
    unevaluated = [g.name for g in guards if not g.evaluated]
    if unevaluated:
        reasons.append("guard evaluation incomplete: " + ", ".join(unevaluated))
    resolved_constraints = [_resolve_constraint(c) for c in (constraints or [])]
    for constraint in resolved_constraints:
        if constraint["status"] == "not_evaluated":
            reasons.append(
                f"constraint {constraint['name']} not evaluable: "
                f"{'candidate' if constraint.get('candidate') is None else 'baseline'} "
                f"{constraint['metric']!r} is unknown (missing evidence, never counted as zero)"
            )
    if reasons:
        return VerdictResult(
            VERDICT_INCONCLUSIVE, reasons, constraints=resolved_constraints
        )

    if baseline_metric is None:
        return VerdictResult(
            VERDICT_INCONCLUSIVE,
            ["baseline metric missing"],
            constraints=resolved_constraints,
        )
    if candidate_metric is None:
        return VerdictResult(
            VERDICT_INCONCLUSIVE,
            ["candidate metric missing"],
            constraints=resolved_constraints,
        )

    improvement = _improvement_pct(baseline_metric, candidate_metric, direction)
    failed_guards = [g for g in guards if not g.passed]
    violated_constraints = [
        c for c in resolved_constraints if c["status"] == "violated"
    ]
    if failed_guards:
        details = "; ".join(
            f"{g.name}: {g.value:.4f} < floor {g.floor:.4f}" for g in failed_guards
        )
        # safety veto first; any simultaneously violated constraint is still listed
        return VerdictResult(
            VERDICT_FAIL,
            [f"guard floor violated ({details})"]
            + [c["reason"] for c in violated_constraints],
            improvement_pct=round(improvement, 6),
            constraints=resolved_constraints,
        )
    if violated_constraints:
        return VerdictResult(
            VERDICT_FAIL,
            [c["reason"] for c in violated_constraints],
            improvement_pct=round(improvement, 6),
            constraints=resolved_constraints,
        )
    if improvement < min_improvement_pct:
        return VerdictResult(
            VERDICT_FAIL,
            [f"improvement {improvement:.2f}% below min {min_improvement_pct:.2f}%"],
            improvement_pct=round(improvement, 6),
            constraints=resolved_constraints,
        )
    return VerdictResult(
        VERDICT_PASS,
        [f"improvement {improvement:.2f}% >= min {min_improvement_pct:.2f}%"],
        improvement_pct=round(improvement, 6),
        constraints=resolved_constraints,
    )
