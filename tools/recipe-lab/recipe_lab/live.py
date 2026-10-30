"""Live-tier orchestration (Sub-issue 3).

The live tier's evaluation engine is sr-bench; this module only wires the
evidence together:

1. the caller (or an upstream script) runs sr-bench for baseline and
   candidate and holds two run IDs;
2. `live_verdict_from_runs` compares them, reads the objective metric,
   applies the three-state verdict with the guard floors and the objective
   constraints, and returns the live-tier evidence block (bench_refs
   included - the ledger record is self-sufficient for Sub-issue 4's
   report replay).

Guards (A3): the contract is the only source of floors. `spec.gates`
declares `{name, metric, floor}`; the `--guard-measurements` file carries
measurements only (`name` / `candidate` / `baseline` values produced
offline by model_eval guard tooling). A `floor` inside the measurement
file is a contract bypass and is rejected. `floor: "baseline"` in the
contract resolves against the measured baseline value. A declared gate
without a usable measurement is not a failure: the gate is unevaluated
and the verdict becomes INCONCLUSIVE.

Verified cost fields (E): sr-bench compare exposes per-arm spend on the
`baseline_targets` / `candidate_targets` entries as `total_spend_usd`,
`cost_usd`, `cache_neutral_cost_usd` and `evaluation_cost_usd`; unknown
usage is serialized as `null` (with a `total_cost_comparison_reason`),
so a constraint on an unknown cost must be unevaluated (INCONCLUSIVE),
never zero.

Live `verify` orchestration (I): `verify_live` drives sr-bench end to end
for one candidate - a preview run (routing distribution into the ledger
notes) plus `repetitions` AB/BA pairs, where each pair re-runs the
baseline's frozen manifest AND submits a fresh candidate-plan, so drift
spreads across both arms. Per-pair compare feeds descriptive medians and
ranges (abba.paired_summary); the verdict reuses `evaluate_verdict`, and
any incomplete pair makes the whole verdict INCONCLUSIVE.
"""

from __future__ import annotations

import json
import statistics
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from . import abba
from .contract import ContractError, ExperimentSpec, Gate
from .srbench import SrbenchCli, extract_metric_values
from .verdict import GuardResult, evaluate_verdict

_MEASUREMENT_KEYS = ("name", "candidate", "baseline")


def measurements_from_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Validate a guard-measurements JSONL (measurements only, no floors).

    Each row is `{name, candidate, baseline}`; `candidate` and `baseline`
    are optional numbers (missing = not measured). A `floor` key means the
    caller is trying to bypass the contract's gates - ContractError. Keys
    outside the measurement schema are rejected the same way.
    """
    validated: list[dict[str, Any]] = []
    for index, row in enumerate(rows, 1):
        if not isinstance(row, dict):
            raise ContractError(f"guard measurement row {index} must be a JSON object")
        unknown = sorted(set(row) - set(_MEASUREMENT_KEYS))
        if unknown:
            raise ContractError(
                f"guard measurement row {index} has non-measurement keys {unknown}; "
                "floors come from the spec gates, the file carries measurements only"
            )
        name = row.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ContractError(
                f"guard measurement row {index}: name must be a non-empty string"
            )
        for key in ("candidate", "baseline"):
            value = row.get(key)
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ContractError(
                    f"guard measurement row {index}: {key} must be a number or null"
                )
        validated.append({key: row.get(key) for key in _MEASUREMENT_KEYS})
    return validated


def resolve_gates(
    gates: Sequence[Gate],
    measurements: list[dict[str, Any]],
) -> tuple[list[GuardResult], list[str]]:
    """Resolve the contract's gates against measured values (A3).

    Returns the GuardResult list (one per declared gate, in contract
    order) and the sorted names of measurements the contract never
    declared. `floor: "baseline"` resolves against the measured baseline
    value; a gate with no measurement, no baseline value for a baseline
    floor, or no candidate value stays unevaluated (value=None, floor=None)
    so the verdict becomes INCONCLUSIVE instead of guessing.
    """
    by_name: dict[str, dict[str, Any]] = {}
    for row in measurements:
        name = row["name"]
        if name in by_name:
            raise ContractError(f"duplicate guard measurement for {name!r}")
        by_name[name] = row
    undeclared = sorted(set(by_name) - {gate.name for gate in gates})

    guards: list[GuardResult] = []
    for gate in gates:
        row = by_name.get(gate.name)
        if row is None:
            guards.append(
                GuardResult(name=gate.name, metric=gate.metric, value=None, floor=None)
            )
            continue
        floor = gate.floor
        if floor == "baseline":
            base = row.get("baseline")
            if base is None:
                guards.append(
                    GuardResult(
                        name=gate.name, metric=gate.metric, value=None, floor=None
                    )
                )
                continue
            floor = float(base)
        candidate = row.get("candidate")
        guards.append(
            GuardResult(
                name=gate.name,
                metric=gate.metric,
                value=None if candidate is None else float(candidate),
                floor=floor,
            )
        )
    return guards, undeclared


def _arm_accuracy(targets: Any, metric_key: str, *, routing_arm: bool) -> float | None:
    """Accuracy of one run's comparison arm from a target list.

    A candidate run typically carries a reference single-target lane (the
    "no routing" arm) before the routing arm, so traversal order alone would
    read the reference instead of the router. The routing arm is the `mom`
    target when present, else the last target with the metric.
    """
    if not isinstance(targets, list):
        return None
    entries = [t for t in targets if isinstance(t, dict)]
    if routing_arm:
        for entry in entries:
            if entry.get("kind") == "mom":
                values = extract_metric_values(entry, metric_key)
                if values:
                    return values[0]
    for entry in reversed(entries):
        values = extract_metric_values(entry, metric_key)
        if values:
            return values[0]
    return None


def _pair_metrics(
    comparison: Any,
    metric_key: str,
    incomplete: list[str],
) -> tuple[float | None, float | None]:
    """Pair baseline/candidate metric values from a compare payload.

    Preferred shape: {"baseline_targets": [...], "candidate_targets": [...]}
    sections (sr-bench compare), where the candidate arm is the routing
    target, not its reference single lane. Flat payloads with exactly two
    numeric occurrences are read in traversal order; anything else is
    ambiguous evidence and becomes INCONCLUSIVE rather than a guess.
    """
    if isinstance(comparison, dict):
        base_section = comparison.get("baseline", comparison.get("baseline_targets"))
        cand_section = comparison.get("candidate", comparison.get("candidate_targets"))
        if isinstance(base_section, (dict, list)) and isinstance(
            cand_section, (dict, list)
        ):
            if isinstance(base_section, list):
                baseline_metric = _arm_accuracy(
                    base_section, metric_key, routing_arm=False
                )
            else:
                base_values = extract_metric_values(base_section, metric_key)
                baseline_metric = base_values[0] if base_values else None
            if isinstance(cand_section, list):
                candidate_metric = _arm_accuracy(
                    cand_section, metric_key, routing_arm=True
                )
            else:
                cand_values = extract_metric_values(cand_section, metric_key)
                candidate_metric = cand_values[0] if cand_values else None
            return baseline_metric, candidate_metric
    values = extract_metric_values(comparison, metric_key)
    if len(values) == 2:
        return values[0], values[1]
    incomplete.append(
        f"ambiguous compare payload for {metric_key!r}: {len(values)} numeric values found"
    )
    return None, None


def _constraint_evidence(
    comparison: Any,
    constraints: list[Constraint],
    incomplete: list[str],
) -> list[dict[str, Any]]:
    """Measure every declared constraint metric on both arms (E).

    The same pairing rules as the objective metric apply (see
    `_pair_metrics`). Unknown values stay None: sr-bench serializes
    unknown spend as null and an unevaluable constraint is INCONCLUSIVE,
    never zero.
    """
    evidence: list[dict[str, Any]] = []
    for constraint in constraints:
        baseline_value, candidate_value = _pair_metrics(
            comparison, constraint.metric, incomplete
        )
        evidence.append(
            {
                "name": constraint.name,
                "metric": constraint.metric,
                "op": constraint.op,
                "bound": constraint.bound,
                "tolerance_pct": constraint.tolerance_pct,
                "baseline": baseline_value,
                "candidate": candidate_value,
            }
        )
    return evidence


def live_verdict_from_runs(
    *,
    spec: ExperimentSpec,
    cli: SrbenchCli,
    baseline_run_id: str,
    candidate_run_id: str,
    measurements: list[dict[str, Any]],
    incomplete_reasons: list[str] | None = None,
) -> dict[str, Any]:
    """Compare two sr-bench runs and decide the live-tier verdict (A3).

    Guard floors come only from `spec.gates`; `measurements` carries the
    offline measured values. Undeclared measurement names are reported
    (in `undeclared_measurements` and the reasons) but never veto. An
    empty contract adds an explicit `gates_note`.
    """
    comparison = cli.compare(baseline_run_id, candidate_run_id)
    incomplete = list(incomplete_reasons or [])
    baseline_metric, candidate_metric = _pair_metrics(
        comparison, spec.objective.metric_key, incomplete
    )

    guards, undeclared = resolve_gates(spec.gates, measurements)
    constraint_evidence = _constraint_evidence(
        comparison, list(spec.objective.constraints), incomplete
    )
    verdict = evaluate_verdict(
        baseline_metric=baseline_metric,
        candidate_metric=candidate_metric,
        direction=spec.objective.direction,
        min_improvement_pct=spec.objective.min_improvement_pct,
        guards=guards,
        incomplete_reasons=incomplete,
        constraints=constraint_evidence,
    )
    reasons = list(verdict.reasons)
    if undeclared:
        reasons.append(
            "undeclared guard measurements ignored (not declared in spec.gates): "
            + ", ".join(undeclared)
        )
    result: dict[str, Any] = {
        "verdict": verdict.verdict,
        "reasons": reasons,
        "improvement_pct": verdict.improvement_pct,
        "metrics": {
            "baseline": baseline_metric,
            "candidate": candidate_metric,
        },
        "gates": {
            g.name: (
                "pass" if g.passed else ("incomplete" if not g.evaluated else "fail")
            )
            for g in guards
        },
        "constraints": verdict.constraints,
        "undeclared_measurements": undeclared,
        "bench_refs": [baseline_run_id, candidate_run_id],
    }
    if not spec.gates:
        result["gates_note"] = "no guard declared"
    return result


def extract_routing_distribution(report: Any) -> dict[str, int]:
    """Merge every `selected_models` counter in a preview report (I).

    The preview report carries per-target routing distributions as
    `selected_models` (model -> case count, verified in sr-bench's
    report builder). Depth-first merge keeps us decoupled from the exact
    nesting; a report without any counter (or a raw-text payload) yields
    an empty dict - missing evidence is reported, never guessed.
    """
    distribution: dict[str, int] = {}

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            selected = node.get("selected_models")
            if isinstance(selected, dict):
                for model, count in selected.items():
                    if isinstance(count, (int, float)) and not isinstance(count, bool):
                        distribution[str(model)] = distribution.get(
                            str(model), 0
                        ) + int(count)
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(report)
    return distribution


def verify_live(
    *,
    spec: ExperimentSpec,
    cli: SrbenchCli,
    baseline_run_id: str,
    targets: list[str],
    repetitions: int,
    measurements: list[dict[str, Any]],
    incomplete_reasons: list[str] | None = None,
    work_dir: str | Path,
) -> dict[str, Any]:
    """Drive sr-bench for one candidate and decide the live verdict (I).

    Per repetition, BOTH arms are measured live in AB/BA order: the
    baseline arm re-runs the baseline run's frozen manifest, the candidate
    arm is a fresh `candidate-plan` (inheriting the same frozen protocol
    and questions). Each completed pair is compared immediately; medians
    and ranges stay descriptive (abba.paired_summary). The final verdict
    reuses `evaluate_verdict` with the per-label medians - any incomplete
    pair, missing metric, or unmeasurable constraint lands in
    `incomplete_reasons` and turns the verdict INCONCLUSIVE.

    `targets` are the registered sr-bench MoM target ids the router is
    serving the candidate config on. The preview run is diagnostic only:
    a failed preview is recorded in the reasons but does not fail the
    verdict (it produces no quality numbers anyway).
    """
    if repetitions < 1:
        raise ContractError("verify needs at least one repetition")
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    baseline_doc = cli.show(baseline_run_id)
    if not isinstance(baseline_doc, dict):
        raise ContractError(
            f"baseline run {baseline_run_id} did not return a run document"
        )
    if baseline_doc.get("status") != "completed":
        raise ContractError(
            f"baseline run {baseline_run_id} is {baseline_doc.get('status')!r}, not completed; "
            "AB/BA pairs must inherit a terminal completed baseline protocol"
        )
    manifest = baseline_doc.get("manifest")
    if not isinstance(manifest, dict):
        raise ContractError(
            f"baseline run {baseline_run_id} does not expose its frozen manifest"
        )
    manifest_path = work_dir / "baseline-manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    incomplete = list(incomplete_reasons or [])
    diagnostics: list[str] = []
    bench_refs = [baseline_run_id]

    # preview first: routing distribution for the ledger notes (no quality).
    # Diagnostic only - a failed preview is recorded but never fails the
    # verdict, since the pairs below carry all quality evidence.
    preview_run = cli.candidate_plan(baseline_run_id, targets=targets, mode="preview")
    preview_id = preview_run.get("id")
    preview_info: dict[str, Any] = {
        "run_id": preview_id,
        "status": preview_run.get("status"),
    }
    bench_refs.append(preview_id)
    if preview_run.get("status") == "completed":
        preview_info["distribution"] = extract_routing_distribution(
            cli.report(preview_id, work_dir / "preview-report.json")
        )
    else:
        diagnostics.append(
            f"preview run {preview_id} ended {preview_run.get('status')!r}; no routing distribution"
        )

    # AB/BA pairs: both arms live, alternating execution order per repetition
    trials: dict[int, dict[str, Any]] = {}
    for label, rep in abba.schedule(repetitions):
        arm = trials.setdefault(rep, {})
        if label == abba.BASELINE:
            run = cli.run_manifest(manifest_path)
        else:
            run = cli.candidate_plan(baseline_run_id, targets=targets, mode="live")
        arm[label] = {"run_id": run.get("id"), "status": run.get("status")}
        bench_refs.append(run.get("id"))

    # per-pair compare + metric/constraint pairing
    metric_key = spec.objective.metric_key
    rows: list[dict[str, Any]] = []
    constraint_values: dict[str, list[tuple[float | None, float | None]]] = {
        constraint.name: [] for constraint in spec.objective.constraints
    }
    for rep in sorted(trials):
        base_run = trials[rep].get(abba.BASELINE, {})
        cand_run = trials[rep].get(abba.CANDIDATE, {})
        if (
            base_run.get("status") != "completed"
            or cand_run.get("status") != "completed"
        ):
            incomplete.append(
                f"pair {rep}: baseline ended {base_run.get('status')!r}, candidate ended {cand_run.get('status')!r}"
            )
            continue
        comparison = cli.compare(base_run["run_id"], cand_run["run_id"])
        baseline_value, candidate_value = _pair_metrics(
            comparison, metric_key, incomplete
        )
        if baseline_value is not None:
            rows.append(
                {"label": abba.BASELINE, "metrics": {metric_key: baseline_value}}
            )
        if candidate_value is not None:
            rows.append(
                {"label": abba.CANDIDATE, "metrics": {metric_key: candidate_value}}
            )
        for constraint in spec.objective.constraints:
            constraint_values[constraint.name].append(
                _pair_metrics(comparison, constraint.metric, incomplete)
            )

    summary = abba.paired_summary(rows, metric_key)
    baseline_metric = (
        summary[abba.BASELINE]["median"] if summary[abba.BASELINE] else None
    )
    candidate_metric = (
        summary[abba.CANDIDATE]["median"] if summary[abba.CANDIDATE] else None
    )

    # constraints: median over pairs; any unmeasured pair stays unevaluated
    constraint_evidence: list[dict[str, Any]] = []
    for constraint in spec.objective.constraints:
        pairs = constraint_values[constraint.name]
        measured = [(b, c) for b, c in pairs if b is not None and c is not None]
        if repetitions and len(measured) == repetitions:
            baseline_c = statistics.median(b for b, _ in measured)
            candidate_c = statistics.median(c for _, c in measured)
        else:
            incomplete.append(
                f"constraint {constraint.name}: measured in {len(measured)} of {repetitions} pairs"
            )
            baseline_c = candidate_c = None
        constraint_evidence.append(
            {
                "name": constraint.name,
                "metric": constraint.metric,
                "op": constraint.op,
                "bound": constraint.bound,
                "tolerance_pct": constraint.tolerance_pct,
                "baseline": baseline_c,
                "candidate": candidate_c,
            }
        )

    guards, undeclared = resolve_gates(spec.gates, measurements)
    verdict = evaluate_verdict(
        baseline_metric=baseline_metric,
        candidate_metric=candidate_metric,
        direction=spec.objective.direction,
        min_improvement_pct=spec.objective.min_improvement_pct,
        guards=guards,
        incomplete_reasons=incomplete,
        constraints=constraint_evidence,
    )
    reasons = list(verdict.reasons)
    reasons.extend(diagnostics)
    if undeclared:
        reasons.append(
            "undeclared guard measurements ignored (not declared in spec.gates): "
            + ", ".join(undeclared)
        )
    result: dict[str, Any] = {
        "verdict": verdict.verdict,
        "reasons": reasons,
        "improvement_pct": verdict.improvement_pct,
        "metrics": {
            "baseline": baseline_metric,
            "candidate": candidate_metric,
        },
        "gates": {
            g.name: (
                "pass" if g.passed else ("incomplete" if not g.evaluated else "fail")
            )
            for g in guards
        },
        "constraints": verdict.constraints,
        "undeclared_measurements": undeclared,
        "bench_refs": bench_refs,
        "trials": {str(rep): trials[rep] for rep in sorted(trials)},
        "abba_summary": summary,
        "preview": preview_info,
    }
    if not spec.gates:
        result["gates_note"] = "no guard declared"
    return result
