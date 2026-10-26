"""Three-state verdict for the live tier (Sub-issue 3).

Ported from InferRSI's evaluation semantics, adapted to sr-bench evidence:

- INCONCLUSIVE: evidence is incomplete (an interrupted trial, missing
  metrics, an unfinished guard evaluation, a changed snapshot hash). This
  outranks FAIL - a broken run must never be recorded as "the recipe is
  worse".
- FAIL: evidence complete but the candidate does not beat the baseline by
  at least `min_improvement_pct`, OR a guard floor is violated (veto).
- PASS: evidence complete, guard floors hold, improvement reaches the bar.

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

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict,
            "reasons": list(self.reasons),
            "improvement_pct": self.improvement_pct,
        }


def _improvement_pct(baseline: float, candidate: float, direction: str) -> float:
    if direction == "minimize":
        if baseline == 0:
            return 0.0 if candidate == 0 else -100.0
        return (baseline - candidate) / baseline * 100.0
    if baseline == 0:
        return 0.0 if candidate == 0 else 100.0
    return (candidate - baseline) / baseline * 100.0


def evaluate_verdict(
    *,
    baseline_metric: float | None,
    candidate_metric: float | None,
    direction: str,
    min_improvement_pct: float,
    guards: list[GuardResult],
    incomplete_reasons: list[str],
) -> VerdictResult:
    """Decide PASS / FAIL / INCONCLUSIVE from live-tier evidence.

    `incomplete_reasons` lists everything that makes the evidence unreliable
    (timeout, missing metric, snapshot mismatch, ...). Guards that did not
    complete also count as incomplete evidence, not as failures.
    """
    reasons = list(incomplete_reasons)
    unevaluated = [g.name for g in guards if not g.evaluated]
    if unevaluated:
        reasons.append("guard evaluation incomplete: " + ", ".join(unevaluated))
    if reasons:
        return VerdictResult(VERDICT_INCONCLUSIVE, reasons)

    if baseline_metric is None:
        return VerdictResult(VERDICT_INCONCLUSIVE, ["baseline metric missing"])
    if candidate_metric is None:
        return VerdictResult(VERDICT_INCONCLUSIVE, ["candidate metric missing"])

    improvement = _improvement_pct(baseline_metric, candidate_metric, direction)
    failed_guards = [g for g in guards if not g.passed]
    if failed_guards:
        details = "; ".join(
            f"{g.name}: {g.value:.4f} < floor {g.floor:.4f}" for g in failed_guards
        )
        return VerdictResult(
            VERDICT_FAIL,
            [f"guard floor violated ({details})"],
            improvement_pct=round(improvement, 6),
        )
    if improvement < min_improvement_pct:
        return VerdictResult(
            VERDICT_FAIL,
            [f"improvement {improvement:.2f}% below min {min_improvement_pct:.2f}%"],
            improvement_pct=round(improvement, 6),
        )
    return VerdictResult(
        VERDICT_PASS,
        [f"improvement {improvement:.2f}% >= min {min_improvement_pct:.2f}%"],
        improvement_pct=round(improvement, 6),
    )
