"""Agentic cold start: profile aggregation and lane assignment (AG5).

The cold-start path for agentic workloads mirrors the MCQ one (C5): a
per-model profile feeds a deterministic lane assignment over the agent
recipe template, and every lane's choice carries its rule and the profile
evidence behind it in a sidecar.

- The profile (`recipe-lab-agentic-profile-v1`) aggregates one single-model
  sr-bench run per model (the AG1 protocol): task success rate with its
  interval, per-task actual and cache-neutral cost, calls per task,
  latency and the truncation rate. Unknown fields stay None - missing
  evidence is never counted as zero.
- Lane assignment over the template's decisions follows §3.5.6: the
  complex coding and complex-general lanes take the strongest model (best
  success rate, ties broken cheaper); the medium lanes take the cheapest
  model within `medium_tolerance_pp` of the best (models without a cost
  cannot be price-compared and stay out); the simple lane and the fallback
  take the cheapest model; the privacy and security lanes take the
  contract's local models with `adaptations.mode: bypass`; the
  legal-health and STEM lanes use the MCQ picture when one exists, else
  the medium lane's model.
- Session protection is written per client: claude-code protects the
  whole declared agent session keyed by the verified client headers
  (x-claude-code-session-id, agent: x-claude-code-agent-id, from AG0-W2);
  every other client protects one conversation keyed by the neutral
  headers. Adaptation is disabled, the progress gate starts in observe
  mode, candidate requirements demand declared capabilities (tool use is
  a model-card capability), and the decision model stays pinned to the
  template's deployment.

Pure functions, no I/O, deterministic: same inputs -> same config bytes.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from typing import Any

PROFILE_KIND = "recipe-lab-agentic-profile-v1"
BOOTSTRAP_KIND = "recipe-lab-agentic-bootstrap-v1"
DEFAULT_MEDIUM_TOLERANCE_PP = 5.0

CLIENTS = ("claude-code", "openai-chat", "generic")
# AG0-W2: Claude Code natively sends x-claude-code-session-id and (on
# delegated subagents) x-claude-code-agent-id; it never sends the neutral
# headers. Every other client uses the neutral pair.
CLIENT_IDENTITY_HEADERS = {
    "claude-code": {
        "scope": "session",
        "session": "x-claude-code-session-id",
        "agent": "x-claude-code-agent-id",
    },
    "openai-chat": {
        "scope": "conversation",
        "session": "x-session-id",
        "conversation": "x-conversation-id",
    },
    "generic": {
        "scope": "conversation",
        "session": "x-session-id",
        "conversation": "x-conversation-id",
    },
}

# template decision -> lane rule. "keep" decisions are rewritten by nobody
# (the omni visual-language lane is a hard modality route, not a profile
# choice).
LANE_DECISIONS: dict[str, str] = {
    "domain_code_complex": "complex",
    "complex_general": "complex",
    "domain_code": "medium",
    "domain_business": "medium",
    "medium_general": "medium",
    "domain_stem_research": "stem",
    "domain_legal_health": "stem",
    "simple_math_fast_path": "cheap",
    "simple_general": "cheap",
    "local_security_containment": "local",
    "local_privacy_policy": "local",
}

_LANE_RULE_TEXT = {
    "complex": "highest task success rate",
    "medium": ("cheapest within medium_tolerance_pp of the best success rate"),
    "cheap": "cheapest model",
    "local": "contract-declared local model (privacy/security bypass)",
    "stem": "MCQ picture order when available, else the medium lane",
}


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def profile_from_reports(reports: Sequence[Any]) -> dict[str, Any]:
    """Aggregate per-model single-target sr-bench run reports (AG1's
    protocol) into the agentic profile.

    Each report contributes the arm a single-model run serves: the `mom`
    target when present, else the last target with an accuracy (the same
    pinning the live tier uses). Fields a report does not carry stay None
    and are flagged in the model entry's `missing` list - the lane
    assignment treats missing cost as "cannot be price-compared", never as
    free.
    """
    models: list[dict[str, Any]] = []
    for index, report in enumerate(reports, 1):
        summary = report.get("summary") if isinstance(report, dict) else None
        targets = summary.get("targets") if isinstance(summary, dict) else None
        entries = [t for t in targets or [] if isinstance(t, dict)]
        target = next((t for t in entries if t.get("kind") == "mom"), None)
        if target is None:
            for candidate in reversed(entries):
                if _number(candidate.get("accuracy")) is not None:
                    target = candidate
                    break
        if target is None:
            raise ValueError(f"report {index} carries no target with an accuracy")
        total = target.get("total")
        total = total if isinstance(total, int) and total > 0 else None
        accuracy = _number(target.get("accuracy"))
        diagnostics = target.get("output_diagnostics")
        planned = (
            diagnostics.get("planned_cases") if isinstance(diagnostics, dict) else None
        )
        output_limit = (
            diagnostics.get("output_limit_cases")
            if isinstance(diagnostics, dict)
            else None
        )
        calls = (
            diagnostics.get("subject_calls") if isinstance(diagnostics, dict) else None
        )
        call_total = calls.get("total") if isinstance(calls, dict) else None
        spend = _number(target.get("total_spend_usd"))
        neutral = _number(target.get("cache_neutral_cost_usd"))
        missing: list[str] = []
        if accuracy is None:
            missing.append("success_rate")
        cost_per_task = spend / total if spend is not None and total else None
        if cost_per_task is None:
            missing.append("cost_per_task_usd")
        neutral_per_task = neutral / total if neutral is not None and total else None
        if neutral_per_task is None:
            missing.append("cache_neutral_cost_per_task_usd")
        mean_calls = (
            call_total / total if isinstance(call_total, int) and total else None
        )
        if mean_calls is None:
            missing.append("mean_calls_per_task")
        truncation = (
            output_limit / planned
            if isinstance(output_limit, int)
            and isinstance(planned, int)
            and planned > 0
            else None
        )
        if truncation is None:
            missing.append("truncation_rate")
        ci = target.get("accuracy_ci95")
        models.append(
            {
                "model": str(target.get("id") or f"report-{index}"),
                "tasks": total,
                "success_rate": accuracy,
                "ci95": [c for c in (ci or []) if _number(c) is not None] or None,
                "cost_per_task_usd": cost_per_task,
                "cache_neutral_cost_per_task_usd": neutral_per_task,
                "mean_calls_per_task": mean_calls,
                "mean_latency_p50_s": _number(target.get("latency_p50_s")),
                "truncation_rate": truncation,
                "missing": sorted(missing),
            }
        )
    return {"kind": PROFILE_KIND, "models": models}


def validate_profile(profile: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Kind- and shape-check a profile; returns its model entries."""
    if not isinstance(profile, Mapping) or profile.get("kind") != PROFILE_KIND:
        raise ValueError(
            f"profile must be a {PROFILE_KIND} artifact, got {profile.get('kind')!r}"
            if isinstance(profile, Mapping)
            else f"profile must be a {PROFILE_KIND} artifact"
        )
    models = profile.get("models")
    if not isinstance(models, list) or not models:
        raise ValueError("profile carries no models")
    for entry in models:
        if not isinstance(entry, Mapping) or not str(entry.get("model") or "").strip():
            raise ValueError("profile model entries need a model name")
        rate = _number(entry.get("success_rate"))
        if rate is None or not 0.0 <= rate <= 1.0:
            raise ValueError(
                f"profile model {entry.get('model')!r} needs a success_rate in [0, 1]"
            )
    return list(models)


def select_models(
    profile: Mapping[str, Any],
    *,
    medium_tolerance_pp: float = DEFAULT_MEDIUM_TOLERANCE_PP,
    local_models: Sequence[str] = (),
    mcq_order: Sequence[str] = (),
) -> dict[str, Any]:
    """AG5 lane assignment: which model serves each lane rule.

    Returns {"models": {lane -> model}, "reasons": {lane -> {...}}} with
    the rule text and the evidence the choice read. Deterministic: ties
    break toward the cheaper model, then the name.
    """
    entries = validate_profile(profile)
    priced = sorted(
        (m for m in entries if _number(m.get("cost_per_task_usd")) is not None),
        key=lambda m: (
            _number(m.get("cost_per_task_usd")),
            -(_number(m.get("success_rate")) or 0.0),
            str(m.get("model")),
        ),
    )
    ranked = sorted(
        entries,
        key=lambda m: (
            -(_number(m.get("success_rate")) or 0.0),
            (
                _number(m.get("cost_per_task_usd"))
                if _number(m.get("cost_per_task_usd")) is not None
                else float("inf")
            ),
            str(m.get("model")),
        ),
    )
    best = ranked[0]
    best_rate = _number(best.get("success_rate")) or 0.0
    complex_model = str(best["model"])

    eligible = [
        m
        for m in ranked
        if best_rate - (_number(m.get("success_rate")) or 0.0)
        <= medium_tolerance_pp / 100.0
    ]
    priced_eligible = sorted(
        (m for m in eligible if _number(m.get("cost_per_task_usd")) is not None),
        key=lambda m: (
            _number(m.get("cost_per_task_usd")),
            -(_number(m.get("success_rate")) or 0.0),
            str(m.get("model")),
        ),
    )
    if priced_eligible:
        medium_model = str(priced_eligible[0]["model"])
        medium_note = None
    else:
        medium_model = complex_model
        medium_note = (
            "no model within the tolerance carries a cost; falling back to "
            "the strongest model (unknown cost is never treated as cheap)"
        )

    if priced:
        cheap_model = str(priced[0]["model"])
        cheap_note = None
    else:
        cheap_model = complex_model
        cheap_note = "no model carries a cost; falling back to the strongest model"

    local_models = [model for model in local_models if str(model).strip()]
    local_model = str(local_models[0]) if local_models else None

    stem_model: str | None = None
    stem_note = "no MCQ picture declared; using the medium lane's model"
    if mcq_order:
        for model in mcq_order:
            if any(str(m.get("model")) == str(model) for m in entries):
                stem_model = str(model)
                stem_note = "ordered by the declared MCQ picture"
                break
        if stem_model is None:
            stem_note = (
                "the MCQ picture names no profiled model; using the medium "
                "lane's model"
            )
    if stem_model is None:
        stem_model = medium_model

    models = {
        "complex": complex_model,
        "medium": medium_model,
        "cheap": cheap_model,
        "stem": stem_model,
        **({"local": local_model} if local_model else {}),
    }
    reasons: dict[str, Any] = {}
    for lane, model in models.items():
        reason: dict[str, Any] = {
            "model": model,
            "rule": _LANE_RULE_TEXT[lane],
        }
        if lane == "medium" and medium_note:
            reason["note"] = medium_note
        if lane == "cheap" and cheap_note:
            reason["note"] = cheap_note
        if lane == "stem":
            reason["note"] = stem_note
        reasons[lane] = reason
    return {"models": models, "reasons": reasons}


def _set_lane_models(decision: dict[str, Any], model: str) -> None:
    decision["modelRefs"] = [{"model": model, "use_reasoning": False}]


def bootstrap_agentic_config(
    template: Mapping[str, Any],
    profile: Mapping[str, Any],
    *,
    client: str,
    medium_tolerance_pp: float = DEFAULT_MEDIUM_TOLERANCE_PP,
    local_models: Sequence[str] = (),
    mcq_order: Sequence[str] = (),
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Rewrite the agent recipe template from the profile (pure function).

    Returns (config, sidecar). The sidecar records every lane's choice
    with its rule and evidence, the client identity contract and the
    protection settings, so the candidate's provenance can answer "why is
    this model on this lane" without re-deriving anything.
    """
    if client not in CLIENTS:
        raise ValueError(f"client must be one of {list(CLIENTS)}, got {client!r}")
    assignment = select_models(
        profile,
        medium_tolerance_pp=medium_tolerance_pp,
        local_models=local_models,
        mcq_order=mcq_order,
    )
    config = copy.deepcopy(dict(template))
    decisions = config.get("decisions")
    if not isinstance(decisions, list):
        raise ValueError("template carries no decisions list")
    reasons: dict[str, Any] = {}
    rewritten = []
    for decision in decisions:
        if not isinstance(decision, dict):
            continue
        name = str(decision.get("name") or "")
        lane = LANE_DECISIONS.get(name)
        if lane is None:
            rewritten.append(decision)
            continue
        model = assignment["models"].get(lane)
        if model is None:
            raise ValueError(
                f"lane {lane!r} (decision {name!r}) has no model: declare "
                "local_models for the privacy/security lanes"
            )
        _set_lane_models(decision, model)
        if lane == "local":
            adaptations = decision.get("adaptations")
            if not isinstance(adaptations, dict):
                adaptations = {}
            adaptations["mode"] = "bypass"
            decision["adaptations"] = adaptations
        rewritten.append(decision)
        reasons[name] = {"lane": lane, **assignment["reasons"][lane]}
    config["decisions"] = rewritten

    identity = CLIENT_IDENTITY_HEADERS[client]
    headers: dict[str, str] = {"session": identity["session"]}
    if "conversation" in identity:
        headers["conversation"] = identity["conversation"]
    if "agent" in identity:
        headers["agent"] = identity["agent"]

    global_section = config.get("global")
    if not isinstance(global_section, dict):
        raise ValueError("template carries no global section")
    router = global_section.get("router")
    if not isinstance(router, dict):
        raise ValueError("template carries no global.router section")
    learning = router.get("learning")
    if not isinstance(learning, dict):
        raise ValueError("template carries no global.router.learning section")
    adaptation = learning.get("adaptation")
    if not isinstance(adaptation, dict):
        adaptation = {}
    adaptation["enabled"] = False
    learning["adaptation"] = adaptation
    protection = learning.get("protection")
    if not isinstance(protection, dict):
        protection = {}
    protection["enabled"] = True
    protection["scope"] = identity["scope"]
    identity_section = protection.get("identity")
    if not isinstance(identity_section, dict):
        identity_section = {}
    identity_section["headers"] = headers
    protection["identity"] = identity_section
    tuning = protection.get("tuning")
    if not isinstance(tuning, dict):
        tuning = {}
    tuning["progress_gate"] = {"mode": "observe"}
    protection["tuning"] = tuning
    learning["protection"] = protection

    routing = config.get("routing")
    if not isinstance(routing, dict):
        raise ValueError("template carries no routing section")
    routing["candidate_requirements"] = {"capabilities": "declared"}

    sidecar = {
        "kind": BOOTSTRAP_KIND,
        "client": client,
        "scope": identity["scope"],
        "identity_headers": headers,
        "lanes": [
            {"decision": name, **reason} for name, reason in sorted(reasons.items())
        ],
        "medium_tolerance_pp": medium_tolerance_pp,
        "local_models": [str(m) for m in local_models],
        "adaptation_enabled": False,
        "progress_gate": {"mode": "observe"},
        "candidate_requirements": {"capabilities": "declared"},
        "decision_model": (
            (config.get("model_catalog") or {}).get("system", {}).get("decision_model")
            if isinstance(config.get("model_catalog"), dict)
            else None
        ),
    }
    return config, sidecar
