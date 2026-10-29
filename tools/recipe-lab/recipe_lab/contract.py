"""Experiment contract: ExperimentSpec / Candidate / Gate / oracle policy.

The contract is the source of truth for one improvement campaign. It is
loaded from YAML (api_version: recipe-lab/v1alpha1) and validated eagerly:
invalid contracts are rejected at load time with actionable errors, never
mid-experiment.

Layout follows the recipe-lab design §5.3:
  objective:    metric_key / direction / min_improvement_pct / replay_metric
  workload:     manifest / content_hash / profile
  attribution:  oracle_policy (label -> intended model)
  gates:        [{name, metric, floor}]  floor = "baseline" | number
  budget:       rounds / max_minutes_per_trial
  stack:        mode / endpoint / baseline_ref

A5: the `evaluation` section is retired - replay and live are separate
commands with CLI-fixed tiers, not contracted per experiment.

Dataset identity (A4): `content_hash = "sha256:" + sha256(<dataset JSONL raw
bytes>)` and `manifest` is the path of that JSONL relative to the spec file.
`ExperimentSpec.verify_workload()` re-hashes the manifest file and rejects a
mismatch before any experiment command touches the data, so a drifted dataset
can never silently join two rounds of results.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from . import API_VERSION

DIRECTIONS = {"minimize", "maximize"}
PROFILES = {"smoke", "quick", "standard"}
STACK_MODES = {"external", "serve"}
# A1: the replay tier ranks candidates with exactly one contracted metric.
# `metric_key` stays the live sr-bench metric; `replay_metric` is what the
# replay ledger records and `ledger top --spec` ranks by.
REPLAY_METRICS = {"routed_accuracy"}
TIERS = {"replay", "preview", "live"}
CANDIDATE_KINDS = {"config", "artifact"}
VERDICTS = {"PASS", "FAIL", "INCONCLUSIVE", "RANKED"}
LIVE_VERDICTS = {"PASS", "FAIL", "INCONCLUSIVE"}
REPLAY_VERDICTS = {"RANKED"}


class ContractError(ValueError):
    """Raised when a contract file or candidate violates the v1alpha1 schema."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractError(message)


def _require_mapping(value: Any, where: str) -> dict:
    _require(isinstance(value, dict), f"{where} must be a mapping")
    return value


def _require_number(value: Any, where: str) -> float:
    _require(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value),
        f"{where} must be a finite number",
    )
    return float(value)


def dataset_content_hash(path: str | Path) -> str:
    """The frozen dataset identity: 'sha256:' + sha256 of the file's raw bytes."""
    return "sha256:" + hashlib.sha256(Path(path).read_bytes()).hexdigest()


@dataclass(frozen=True)
class Objective:
    metric_key: str
    direction: str
    min_improvement_pct: float
    replay_metric: str = "routed_accuracy"

    @classmethod
    def from_dict(cls, raw: dict) -> Objective:
        metric_key = raw.get("metric_key")
        _require(
            isinstance(metric_key, str) and metric_key.strip(),
            "objective.metric_key must be a non-empty string",
        )
        _require(
            all(part.strip() for part in metric_key.split(".")),
            "objective.metric_key must be a non-empty dotted JSON key",
        )
        direction = raw.get("direction")
        _require(
            direction in DIRECTIONS,
            f"objective.direction must be one of {sorted(DIRECTIONS)}",
        )
        pct = _require_number(
            raw.get("min_improvement_pct", 0.0), "objective.min_improvement_pct"
        )
        _require(pct >= 0, "objective.min_improvement_pct must be >= 0")
        replay_metric = raw.get("replay_metric", "routed_accuracy")
        _require(
            replay_metric in REPLAY_METRICS,
            f"objective.replay_metric must be one of {sorted(REPLAY_METRICS)} (got {replay_metric!r})",
        )
        return cls(
            metric_key=metric_key,
            direction=direction,
            min_improvement_pct=pct,
            replay_metric=replay_metric,
        )

    def to_dict(self) -> dict:
        return {
            "metric_key": self.metric_key,
            "direction": self.direction,
            "min_improvement_pct": self.min_improvement_pct,
            "replay_metric": self.replay_metric,
        }


@dataclass(frozen=True)
class Workload:
    manifest: str
    content_hash: str
    profile: str

    @classmethod
    def from_dict(cls, raw: dict) -> Workload:
        manifest = raw.get("manifest")
        _require(
            isinstance(manifest, str) and manifest.strip(),
            "workload.manifest must be a non-empty string",
        )
        content_hash = raw.get("content_hash")
        _require(
            isinstance(content_hash, str) and content_hash.strip(),
            "workload.content_hash is required (frozen dataset hash; cross-round comparability depends on it)",
        )
        profile = raw.get("profile", "quick")
        _require(
            profile in PROFILES, f"workload.profile must be one of {sorted(PROFILES)}"
        )
        return cls(manifest=manifest, content_hash=content_hash, profile=profile)

    def to_dict(self) -> dict:
        return {
            "manifest": self.manifest,
            "content_hash": self.content_hash,
            "profile": self.profile,
        }


@dataclass(frozen=True)
class OraclePolicy:
    """Label -> intended model. Declared, never recomputed: E2 attribution
    depends on this mapping, not on re-implementing router logic."""

    mapping: dict[str, str]

    @classmethod
    def from_dict(cls, raw: Any) -> OraclePolicy:
        _require(
            isinstance(raw, dict) and raw,
            "attribution.oracle_policy must be a non-empty mapping of label -> model",
        )
        for label, model in raw.items():
            _require(
                isinstance(label, str) and label.strip(),
                f"oracle policy label must be a non-empty string: {label!r}",
            )
            _require(
                isinstance(model, str) and model.strip(),
                f"oracle policy model for label {label!r} must be a non-empty string",
            )
        return cls(mapping=dict(raw))

    def expected(self, label: str) -> str | None:
        return self.mapping.get(label)

    def covers(self, labels: list[str]) -> list[str]:
        return [label for label in labels if label not in self.mapping]

    def to_dict(self) -> dict:
        return dict(self.mapping)


@dataclass(frozen=True)
class Gate:
    name: str
    metric: str
    floor: str | float  # "baseline" or a numeric floor

    @classmethod
    def from_dict(cls, raw: dict, index: int) -> Gate:
        name = raw.get("name")
        _require(
            isinstance(name, str) and name.strip(),
            f"gates[{index}].name must be a non-empty string",
        )
        metric = raw.get("metric")
        _require(
            isinstance(metric, str) and all(part.strip() for part in metric.split(".")),
            f"gates[{index}].metric must be a non-empty dotted JSON key",
        )
        floor = raw.get("floor")
        if floor != "baseline":
            floor = _require_number(floor, f"gates[{index}].floor")
        return cls(name=name, metric=metric, floor=floor)

    def to_dict(self) -> dict:
        return {"name": self.name, "metric": self.metric, "floor": self.floor}


@dataclass(frozen=True)
class Budget:
    rounds: int
    max_minutes_per_trial: float

    @classmethod
    def from_dict(cls, raw: dict) -> Budget:
        rounds = raw.get("rounds", 1)
        _require(
            isinstance(rounds, int) and not isinstance(rounds, bool) and rounds >= 1,
            "budget.rounds must be an integer >= 1",
        )
        minutes = _require_number(
            raw.get("max_minutes_per_trial", 120.0), "budget.max_minutes_per_trial"
        )
        _require(minutes > 0, "budget.max_minutes_per_trial must be > 0")
        return cls(rounds=rounds, max_minutes_per_trial=minutes)

    def to_dict(self) -> dict:
        return {
            "rounds": self.rounds,
            "max_minutes_per_trial": self.max_minutes_per_trial,
        }


@dataclass(frozen=True)
class Stack:
    mode: str
    endpoint: str
    baseline_ref: str

    @classmethod
    def from_dict(cls, raw: dict) -> Stack:
        mode = raw.get("mode", "external")
        _require(
            mode in STACK_MODES, f"stack.mode must be one of {sorted(STACK_MODES)}"
        )
        endpoint = raw.get("endpoint")
        _require(
            isinstance(endpoint, str) and endpoint.strip(),
            "stack.endpoint must be a non-empty string",
        )
        baseline_ref = raw.get("baseline_ref")
        _require(
            isinstance(baseline_ref, str) and baseline_ref.strip(),
            "stack.baseline_ref must be a non-empty string",
        )
        return cls(mode=mode, endpoint=endpoint, baseline_ref=baseline_ref)

    def to_dict(self) -> dict:
        return {
            "mode": self.mode,
            "endpoint": self.endpoint,
            "baseline_ref": self.baseline_ref,
        }


@dataclass(frozen=True)
class ExperimentSpec:
    api_version: str
    objective: Objective
    workload: Workload
    oracle_policy: OraclePolicy
    gates: tuple[Gate, ...] = field(default=())
    budget: Budget = field(
        default_factory=lambda: Budget(rounds=1, max_minutes_per_trial=120.0)
    )
    stack: Stack | None = None
    # directory of the YAML file this spec was loaded from (from_yaml only);
    # workload.manifest resolves against it. Not part of the contract.
    spec_dir: str | None = None

    @classmethod
    def from_dict(cls, raw: Any) -> ExperimentSpec:
        raw = _require_mapping(raw, "spec")
        api_version = raw.get("api_version")
        _require(
            api_version == API_VERSION,
            f"api_version must be {API_VERSION!r} (got {api_version!r})",
        )
        objective = Objective.from_dict(
            _require_mapping(raw.get("objective"), "objective")
        )
        workload = Workload.from_dict(_require_mapping(raw.get("workload"), "workload"))
        attribution = _require_mapping(raw.get("attribution", {}), "attribution")
        oracle_policy = OraclePolicy.from_dict(attribution.get("oracle_policy"))
        gates_raw = raw.get("gates", [])
        _require(isinstance(gates_raw, list), "gates must be a list")
        gates = tuple(Gate.from_dict(g, i) for i, g in enumerate(gates_raw))
        budget = Budget.from_dict(_require_mapping(raw.get("budget", {}), "budget"))
        stack_raw = raw.get("stack")
        stack = (
            Stack.from_dict(_require_mapping(stack_raw, "stack"))
            if stack_raw is not None
            else None
        )
        # A5: `evaluation` is retired - replay and live are separate commands,
        # tiers are fixed by the CLI, not contracted per experiment.
        _require(
            "evaluation" not in raw,
            "spec key 'evaluation' is retired: remove it (replay and live are separate commands now)",
        )
        return cls(
            api_version=api_version,
            objective=objective,
            workload=workload,
            oracle_policy=oracle_policy,
            gates=gates,
            budget=budget,
            stack=stack,
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> ExperimentSpec:
        path = Path(path)
        _require(path.is_file(), f"spec file not found: {path}")
        with path.open("r", encoding="utf-8") as fh:
            try:
                raw = yaml.safe_load(fh)
            except yaml.YAMLError as exc:
                raise ContractError(
                    f"spec file is not valid YAML: {path} ({exc})"
                ) from exc
        spec = cls.from_dict(raw)
        return cls(**{**spec.__dict__, "spec_dir": str(path.parent)})

    def verify_workload(self, base_dir: str | Path | None = None) -> str:
        """Re-hash the workload manifest and compare with the declared hash.

        Returns the verified content hash. `workload.manifest` resolves against
        `base_dir`, defaulting to the spec file's directory (from_yaml). This
        is the A4 contract check: attribute / run / propose-* call it before
        touching any data, so a drifted dataset can never silently join two
        rounds of results.
        """
        base = (
            Path(base_dir)
            if base_dir is not None
            else (Path(self.spec_dir) if self.spec_dir else None)
        )
        if base is None:
            raise ContractError(
                "cannot locate the workload manifest: the spec was not loaded from a file; pass base_dir"
            )
        manifest = base / self.workload.manifest
        if not manifest.is_file():
            raise ContractError(
                f"dataset manifest not found: {manifest} (workload.manifest is relative to the spec file)"
            )
        actual = dataset_content_hash(manifest)
        if actual != self.workload.content_hash:
            raise ContractError(
                f"dataset hash mismatch: spec={self.workload.content_hash} file={actual}"
            )
        return actual

    def require_dataset_file(self, path: str | Path, what: str) -> str:
        """Assert a CLI-provided data file (--cases) is byte-identical to the
        frozen workload. Raises ContractError (exit 2 via main) on drift or a
        missing file."""
        try:
            actual = dataset_content_hash(path)
        except OSError as exc:
            raise ContractError(f"cannot read {what}: {exc}") from exc
        if actual != self.workload.content_hash:
            raise ContractError(
                f"{what} hash mismatch: spec={self.workload.content_hash} file={actual}"
            )
        return actual

    def validate_policy_covers(self, labels: list[str]) -> list[str]:
        """Return oracle-policy coverage gaps for a dataset's label set."""
        return self.oracle_policy.covers(labels)

    def to_dict(self) -> dict:
        out: dict[str, Any] = {
            "api_version": self.api_version,
            "objective": self.objective.to_dict(),
            "workload": self.workload.to_dict(),
            "attribution": {"oracle_policy": self.oracle_policy.to_dict()},
            "gates": [g.to_dict() for g in self.gates],
            "budget": self.budget.to_dict(),
        }
        if self.stack is not None:
            out["stack"] = self.stack.to_dict()
        return out


@dataclass(frozen=True)
class Candidate:
    kind: str
    ref: str
    change_summary: str = ""
    proposer: str = "manual"
    insight_refs: tuple[str, ...] = field(default=())

    @classmethod
    def from_dict(cls, raw: dict) -> Candidate:
        kind = raw.get("kind")
        _require(
            kind in CANDIDATE_KINDS,
            f"candidate.kind must be one of {sorted(CANDIDATE_KINDS)}",
        )
        ref = raw.get("ref")
        _require(
            isinstance(ref, str) and ref.strip(),
            "candidate.ref must be a non-empty string",
        )
        provenance = raw.get("provenance", {})
        provenance = _require_mapping(provenance, "candidate.provenance")
        insight_refs = provenance.get("insight_refs", [])
        _require(
            isinstance(insight_refs, list),
            "candidate.provenance.insight_refs must be a list",
        )
        return cls(
            kind=kind,
            ref=ref,
            change_summary=str(raw.get("change_summary", "")),
            proposer=str(provenance.get("proposer", "manual")),
            insight_refs=tuple(str(r) for r in insight_refs),
        )

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "ref": self.ref,
            "change_summary": self.change_summary,
            "provenance": {
                "proposer": self.proposer,
                "insight_refs": list(self.insight_refs),
            },
        }
