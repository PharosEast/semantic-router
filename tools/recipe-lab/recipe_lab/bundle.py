"""C9: acceptance bundle packaging (final live PASS + transfer guards).

`recipe-lab bundle` packages the acceptance evidence for exactly one ledger
record into a reviewable directory. It is deliberately hard to run: only a
live-tier PASS that cites a final-verify workload (split ``frozen_test`` or a
``role: transfer`` workload) with every declared guard passing is eligible -
anything else is a contract error (exit 2), never a warning.

Artifacts (a byte-stable pure function of the inputs; no wall-clock content):
config.yaml / validate.txt / config.diff / report.md / profile.json /
ledger.jsonl (campaign excerpt) / provenance.json / PROMOTION.md / REVIEW.md.

Honesty rules the bundle enforces:
- provenance cites hashes and basenames only - no absolute personal paths;
- every artifact is scanned for private content (personal user directories,
  non-local literal-IP URLs) plus any --private-pattern regexes before
  anything is written; a match fails the bundle;
- missing evidence (router commit, model revisions, model_eval script) is
  recorded as null, never guessed.
"""

from __future__ import annotations

import difflib
import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any

from .contract import ContractError, ExperimentSpec
from .ledger import Ledger
from .report import render_promotion_draft

_LIMITATIONS_TEXT = (
    "Replayed from the recipe-lab ledger only. Metrics are descriptive; "
    "statistical significance is not established. A PASS is a recommendation, "
    "not a promotion: promotion remains a human decision (see PROMOTION.md "
    "and REVIEW.md)."
)

_CACHE_TOOLS_TEXT = (
    "The acceptance protocol evaluates the router with response caching and "
    "tool-call features disabled on the serving stack; the packaged metrics "
    "are only valid under that configuration."
)

_REVIEW_TEMPLATE = """# Recipe bundle review (V5)

- record: {record_id}
- candidate ref: {ref} ({kind})
- evidence draft: PROMOTION.md
- hashes: provenance.json

## Reviewer checklist (fill in during V5)

- [ ] PROMOTION.md human checklist complete
- [ ] provenance.json hashes verified against the archived inputs
- [ ] transfer guards and constraints reviewed in report.md
- [ ] decision (promote / hold / rework):
"""

# C9 private-content scan: personal user directories and non-local literal-IP
# URLs. Custom --private-pattern regexes extend these; a match fails the
# bundle before anything is written.
_BASE_PRIVATE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("windows-user-path", re.compile(r'[A-Za-z]:\\Users\\[^\s/:*?"<>|\\]+')),
    ("unix-user-path", re.compile(r"/(?:home|Users)/[A-Za-z0-9._-]+")),
)
_NON_LOCAL_URL = re.compile(r"https?://(\d{1,3}(?:\.\d{1,3}){3})(?=[/:\"\s']|$)")
_LOCAL_IPS = {"127.0.0.1", "0.0.0.0"}


def _content_hash(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _record_is_final(spec: ExperimentSpec, record: dict[str, Any]) -> bool:
    """A record is a final-verify record only if the contract forced --final.

    The frozen_test quarantine and the transfer-workload quarantine are the
    two contract paths that refuse a non-final verdict, so a record citing
    either could only have been produced by a --final run. Anything else
    (a tune workload, a dev split, a legacy record) is not promotion
    evidence no matter how it was produced.
    """
    if record.get("split") == "frozen_test":
        return True
    workload_name = record.get("workload")
    if not workload_name:
        return False
    workload = next((w for w in spec.workloads if w.name == workload_name), None)
    if workload is None:
        raise ContractError(
            f"record cites workload {workload_name!r} which the spec does not declare"
        )
    return workload.role == "transfer"


def _run_config_validate(config_path: Path, validate_bin: str) -> str:
    """Run `<validate-bin> config validate` and return its output text.

    A failing validation is a bundle rejection, not a packaged artifact.
    """
    try:
        proc = subprocess.run(
            [validate_bin, "config", "validate", "--config", str(config_path)],
            capture_output=True,
            text=True,
            timeout=300.0,
            check=False,
        )
    except FileNotFoundError as exc:
        raise ContractError(f"validator binary not found: {validate_bin}") from exc
    except subprocess.TimeoutExpired as exc:
        raise ContractError("config validate timed out") from exc
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip().splitlines()
        tail = detail[-1] if detail else "no output"
        raise ContractError(
            f"bundle rejected: config validate exit {proc.returncode}: {tail}"
        )
    return proc.stdout + proc.stderr


def scan_private_content(
    artifacts: dict[str, str], extra_patterns: list[str] | None = None
) -> None:
    """Fail the bundle when any artifact carries private content.

    The error lists offending artifact:pattern pairs; the matched content is
    never echoed (it must not leak into logs either).
    """
    compiled = list(_BASE_PRIVATE_PATTERNS)
    for index, raw in enumerate(extra_patterns or []):
        try:
            compiled.append((f"custom[{index}]", re.compile(raw)))
        except re.error as exc:
            raise ContractError(f"invalid --private-pattern {raw!r}: {exc}") from exc
    hits: list[str] = []
    for name in sorted(artifacts):
        text = artifacts[name]
        for pattern_name, pattern in compiled:
            if pattern.search(text):
                hits.append(f"{name}:{pattern_name}")
        for match in _NON_LOCAL_URL.finditer(text):
            if match.group(1) not in _LOCAL_IPS:
                hits.append(f"{name}:non-local-ip-url")
                break
    if hits:
        raise ContractError(
            "bundle artifacts contain private content "
            "(artifact:pattern; content not echoed): " + ", ".join(sorted(set(hits)))
        )


def _metrics_json(metrics: dict[str, Any] | None) -> str:
    if not metrics:
        return "n/a"
    return json.dumps(metrics, sort_keys=True, ensure_ascii=False)


def _routing_distribution(notes: str) -> dict[str, Any] | None:
    marker = "routing_distribution="
    if not notes or marker not in notes:
        return None
    tail = notes.rsplit(marker, 1)[1]
    try:
        parsed = json.loads(tail)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _repetitions_from_notes(notes: str) -> int | None:
    match = re.search(r"repetitions=(\d+)", notes or "")
    return int(match.group(1)) if match else None


def render_bundle_report(
    spec: ExperimentSpec,
    records: list[dict[str, Any]],
    record: dict[str, Any],
) -> str:
    """Deterministic acceptance report rendered from the ledger + spec only."""
    candidate = record.get("candidate") or {}
    paired = record.get("paired") or {}
    gates = record.get("gates") or {}
    lines = [
        "# recipe-lab acceptance bundle report",
        "",
        f"- record: `{record.get('record_id', 'unrecorded')}` (final verify, tier live)",
        f"- verdict: **{record.get('verdict')}**",
        f"- workload: {record.get('workload') or '-'} / split: {record.get('split') or '-'}",
        f"- candidate ref: `{candidate.get('ref')}` ({candidate.get('kind')})",
        f"- dataset: {record.get('dataset_hash')}",
        f"- baseline ref: {record.get('baseline_ref') or '-'}",
        "",
        "## Verdict detail",
        "",
        f"- metrics: {_metrics_json(record.get('metrics'))}",
        f"- gates: {', '.join(f'{k}={v}' for k, v in gates.items()) or 'none declared'}",
        f"- bench evidence: {', '.join(record.get('bench_refs') or []) or '-'}",
    ]
    if paired:
        lines += [
            f"- paired cases: {paired.get('paired_cases')}",
            f"- quality delta: {paired.get('quality_delta')}",
            f"- quality delta CI95: {paired.get('quality_delta_ci95')}",
        ]
        if paired.get("quality_delta_bootstrap_ci95") is not None:
            lines.append(
                f"- quality delta bootstrap CI95: {paired.get('quality_delta_bootstrap_ci95')}"
            )
    constraints = record.get("constraints") or {}
    if constraints:
        lines += ["", "### Constraints", ""]
        for name, evidence in constraints.items():
            lines.append(
                f"- {name}: {evidence.get('status', '?')} "
                f"(candidate={evidence.get('candidate')}, bound={evidence.get('bound_value')})"
            )
    reasons = record.get("reasons") or []
    if reasons:
        lines += ["", "### Verdict reasons", ""]
        lines += [f"- {reason}" for reason in reasons]
    distribution = _routing_distribution(record.get("notes") or "")
    if distribution:
        lines += ["", "### Observed routing distribution (final verify)", ""]
        lines += [f"- {model}: {count}" for model, count in sorted(distribution.items())]
    lines += [
        "",
        "## Campaign evidence (ledger replay)",
        "",
        "| record | round | tier | workload | verdict | metrics | gates |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in records:
        row_gates = row.get("gates") or {}
        gates_text = ", ".join(f"{k}={v}" for k, v in row_gates.items()) or "-"
        lines.append(
            f"| {row.get('record_id', '-')} | {row.get('round')} | {row.get('tier')} "
            f"| {row.get('workload') or '-'} | {row.get('verdict')} "
            f"| {_metrics_json(row.get('metrics'))} | {gates_text} |"
        )
    counts: dict[str, dict[str, int]] = {}
    for row in records:
        name = row.get("workload") or "default"
        verdict = str(row.get("verdict"))
        counts.setdefault(name, {})
        counts[name][verdict] = counts[name].get(verdict, 0) + 1
    lines += ["", "### Per-workload verdict counts", ""]
    for name in sorted(counts):
        detail = ", ".join(f"{count}x {verdict}" for verdict, count in sorted(counts[name].items()))
        lines.append(f"- {name}: {detail}")
    lines += ["", "## Evidence quality", ""]
    repetitions = _repetitions_from_notes(record.get("notes") or "")
    lines.append(
        f"- verify repetitions: {repetitions if repetitions is not None else 'unknown'}"
    )
    if paired:
        paired_cases = paired.get("paired_cases")
        min_cases = spec.objective.min_cases
        verdict_text = (
            "meets"
            if isinstance(paired_cases, int) and paired_cases >= min_cases
            else "below or unknown"
        )
        lines.append(
            f"- paired cases {paired_cases} vs objective.min_cases {min_cases}: {verdict_text}"
        )
    warnings = [
        f"{row.get('record_id')}: coverage_warning"
        for row in records
        if row.get("coverage_warning")
    ]
    warnings += [
        f"{row.get('record_id')}: constraint_warning"
        for row in records
        if row.get("constraint_warning")
    ]
    lines.append("- warnings: " + ("; ".join(warnings) if warnings else "none"))
    lines += ["", "## Assumptions", ""]
    if spec.cost_basis is not None:
        text = f"- cost basis: {spec.cost_basis.source}"
        if spec.cost_basis.assumed_pricing:
            text += " (assumed pricing, not metered bills)"
        lines.append(text)
    if spec.references is not None:
        lines.append(
            f"- reference arms: strong={spec.references.strong} weak={spec.references.weak}"
        )
    lines.append(
        "- guard floors come from the spec's gates section; the ledger record "
        "carries the measured outcomes"
    )
    lines += [
        "",
        "## Limitations",
        "",
        _LIMITATIONS_TEXT,
        "",
        "## Cache and tools",
        "",
        _CACHE_TOOLS_TEXT,
        "",
    ]
    return "\n".join(lines)


def build_bundle(
    *,
    spec: ExperimentSpec,
    ledger_path: str,
    record_id: str,
    out_dir: str | Path,
    config_path: str | Path,
    skeleton_path: str | Path,
    profile_path: str | Path,
    validate_bin: str,
    router_commit: str | None = None,
    model_revisions: dict[str, str] | None = None,
    model_eval_script: str | None = None,
    extra_private_patterns: list[str] | None = None,
) -> dict[str, Any]:
    """Assemble the C9 bundle for one final live PASS record.

    Every rejection is a ContractError (exit 2 through the CLI): unknown
    record, non-live tier, non-PASS verdict, non-final verify, a transfer
    guard not passing, a failing config validate, or private content in any
    artifact. Nothing is written unless every check passes.
    """
    records = Ledger(ledger_path).read()
    matches = [row for row in records if row.get("record_id") == record_id]
    if not matches:
        raise ContractError(f"record {record_id} not found in ledger {ledger_path}")
    record = matches[0]
    if record.get("tier") != "live":
        raise ContractError(f"record {record_id} is not a live-tier record")
    if record.get("verdict") != "PASS":
        raise ContractError(
            f"record {record_id} verdict is {record.get('verdict')!r}; "
            "bundle packages only PASS records"
        )
    if not _record_is_final(spec, record):
        raise ContractError(
            f"record {record_id} is not a final-verify record (needs split "
            "'frozen_test' or a role 'transfer' workload); bundle packages "
            "only --final promotion evidence"
        )
    not_passing = sorted(
        f"{name}={value}" for name, value in (record.get("gates") or {}).items()
        if value != "pass"
    )
    if not_passing:
        raise ContractError(
            f"record {record_id} has transfer guards not passing: "
            + ", ".join(not_passing)
        )

    config_file = Path(config_path)
    skeleton_file = Path(skeleton_path)
    profile_file = Path(profile_path)
    validate_text = _run_config_validate(config_file, validate_bin)

    config_bytes = config_file.read_bytes()
    skeleton_text = skeleton_file.read_text(encoding="utf-8")
    profile_bytes = profile_file.read_bytes()
    diff_text = "".join(
        difflib.unified_diff(
            skeleton_text.splitlines(keepends=True),
            config_bytes.decode("utf-8").splitlines(keepends=True),
            fromfile="skeleton",
            tofile="config.yaml",
        )
    )
    campaign_hash = record.get("dataset_hash")
    excerpt = [row for row in records if row.get("dataset_hash") == campaign_hash]
    ledger_text = "".join(
        json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in excerpt
    )
    revisions = dict(model_revisions or {})
    models_revision: dict[str, str | None] = {
        model.name: revisions.pop(model.name, None) for model in spec.models
    }
    models_revision.update(revisions)
    provenance = {
        "schema_version": "v1",
        "kind": "recipe-lab-bundle",
        "id": f"recipe-lab-bundle:{record_id}",
        "record_id": record_id,
        "router_commit": router_commit,
        "models": models_revision,
        "datasets": {
            "record_dataset_hash": campaign_hash,
            "workloads": {
                workload.name: {
                    split: split_spec.content_hash
                    for split, split_spec in workload.splits.items()
                }
                for workload in spec.workloads
            },
        },
        "artifacts": {
            "skeleton": {
                "name": skeleton_file.name,
                "sha256": _content_hash(skeleton_file),
            },
            "config": {"name": "config.yaml", "sha256": _content_hash(config_file)},
            "profile": {
                "name": "profile.json",
                "sha256": _content_hash(profile_file),
            },
            "ledger": {
                "name": Path(ledger_path).name,
                "sha256": _content_hash(Path(ledger_path)),
            },
        },
        "run_ids": {
            "baseline_ref": record.get("baseline_ref"),
            "bench_refs": list(record.get("bench_refs") or []),
        },
        "model_eval_script": (
            {
                "name": Path(model_eval_script).name,
                "sha256": _content_hash(Path(model_eval_script)),
            }
            if model_eval_script
            else None
        ),
    }

    artifacts: dict[str, str] = {
        "validate.txt": validate_text,
        "config.diff": diff_text,
        "report.md": render_bundle_report(spec, records, record),
        "ledger.jsonl": ledger_text,
        "provenance.json": json.dumps(
            provenance, sort_keys=True, ensure_ascii=False, indent=1
        )
        + "\n",
        "PROMOTION.md": render_promotion_draft(record),
        "REVIEW.md": _REVIEW_TEMPLATE.format(
            record_id=record_id,
            ref=(record.get("candidate") or {}).get("ref"),
            kind=(record.get("candidate") or {}).get("kind"),
        ),
        # byte copies are scanned through their decoded text as well
        "config.yaml": "",
        "profile.json": "",
    }
    artifacts["config.yaml"] = config_bytes.decode("utf-8")
    artifacts["profile.json"] = profile_bytes.decode("utf-8")
    scan_private_content(artifacts, extra_private_patterns)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.yaml").write_bytes(config_bytes)
    (out / "profile.json").write_bytes(profile_bytes)
    for name in ("validate.txt", "config.diff", "report.md", "ledger.jsonl",
                 "provenance.json", "PROMOTION.md", "REVIEW.md"):
        (out / name).write_text(artifacts[name], encoding="utf-8")
    return {
        "bundle": str(out),
        "record": record_id,
        "artifacts": sorted(artifacts),
    }
