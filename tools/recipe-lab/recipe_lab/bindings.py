"""Cold-start C4+C5 (template path): per-lane bindings from the C3 profile.

Replaces the legacy ``result_to_config.py`` scaffold as the main generation
path (sub-issue-8 §2.2 steps 5-7). For every ``--lambdas`` value this module:

1. **Checks the routing space** (step 5). Either condition ends the flow
   with a "no benefit to routing" report and no candidates:
   - on every measured lane the best model's advantage over the cheapest
     model is below ``bootstrap.min_advantage_pp``;
   - every measured lane has the same first-place model and the cheapest
     model is never within the advantage threshold of the best on any
     measured lane.
2. **Plans the bindings** (step 6). Per lane, in the template's order:
   capability filter (only models whose user-declared capabilities satisfy
   the lane requirement, sub-issue-8 §2.1 - never guessed) -> rank by
   ``accuracy - lambda * normalized_cost`` (cost normalized per lane by the
   max across models, the ``cost_aware_calib.py`` convention) -> take the
   top ``minimum_candidates`` (at least one; frontier-style lanes take
   their declared minimum). Lanes observed on fewer than
   ``bootstrap.min_lane_cases`` cases are **unmeasured**: their candidates
   are ranked by the lambda-adjusted overall score and marked "assigned by
   the default rule, untested". A lane with no model satisfying its
   capability requirement must be explicitly excluded via
   ``--exclude-decision`` (which makes ``recipe builtin init`` drop the
   lane) - the report names every exclusion.
3. **Generates the config** (step 7). ``vllm-sr recipe builtin init`` binds
   the template to the planned refs over the spec's ``providers_config``
   (endpoints and credentials come only from there), then
   ``vllm-sr config validate`` must accept the output before it is emitted.
   The derived eval copy of the provider config disables the response cache
   (a warm cache would poison replay/live evidence) and records the reason.

Anti-leak: only the calib split feeds the ranking; ``dev`` belongs to the
answer bank and ``frozen_test`` is quarantined (enforced by
``bootstrap.load_profile``).
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import yaml

from .bootstrap import load_profile, parse_lambdas, validate_config
from .contract import ContractError, ExperimentSpec, dataset_content_hash, lambda_tag
from .observe_lanes import (
    OBSERVE_MODEL_NAME,
    capability_satisfied,
    lane_requirements,
    template_info,
)
from .observe_lanes import _run_json as _default_run_json

BINDINGS_SCHEMA = "recipe-lab-bindings-v1"
CACHE_DISABLED_REASON = (
    "disabled for evaluation: a warm cache would let repeated cases hit the "
    "cache instead of the routed model and poison replay/live evidence. "
    "Re-enable for production if wanted."
)


def resolve_lambdas(spec: ExperimentSpec, raw: str | None) -> list[float]:
    """The lambda list: CLI override, else the contract's bootstrap.lambdas.

    A raw string is the legacy comma-separated form (parse_lambdas
    semantics: deduped, order kept); None falls back to the contract, and a
    contract without lambdas defaults to ``[0.0]`` (pure accuracy).
    """
    if raw is not None and raw.strip():
        return parse_lambdas(raw)
    contract = spec.bootstrap_config.lambdas
    if contract:
        return [float(lam) for lam in contract]
    return [0.0]


def _calib_stats(profile: dict[str, Any]) -> dict[str, Any]:
    stats = (profile.get("model_split_stats") or {}).get("calib") or {}
    if not stats:
        raise ContractError(
            "profile has no calib stats: bindings consumes the C3 profile's "
            "calib split only (rerun profile with --splits calib)"
        )
    return stats


def lane_case_counts(profile: dict[str, Any]) -> dict[str, int]:
    """calib lane -> case count (union over models; all models share tasks)."""
    counts: dict[str, int] = {}
    for alias, block in _calib_stats(profile).items():
        labels = (block or {}).get("labels") or {}
        if not isinstance(labels, dict):
            raise ContractError(
                f"profile calib stats for {alias!r} have no labels block"
            )
        for lane, entry in labels.items():
            n = entry.get("n") if isinstance(entry, dict) else None
            if not isinstance(n, int) or n < 0:
                raise ContractError(
                    f"profile calib stats for {alias!r} lane {lane!r} has no case count"
                )
            counts[lane] = max(counts.get(lane, 0), n)
    if not counts:
        raise ContractError("profile has no calib lanes: nothing to bind")
    return counts


def _lane_stat(
    profile: dict[str, Any], alias: str, lane: str, key: str
) -> float | None:
    block = ((_calib_stats(profile).get(alias) or {}).get("labels") or {}).get(lane)
    if not isinstance(block, dict):
        return None
    value = block.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _overall_stat(profile: dict[str, Any], alias: str, key: str) -> float | None:
    block = (_calib_stats(profile).get(alias) or {}).get("total")
    if not isinstance(block, dict):
        return None
    value = block.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _normalize_costs(
    costs: dict[str, float | None],
) -> dict[str, float] | None:
    """Per-lane cost normalization (``cost_aware_calib`` convention: divide
    by the max across models). None when no model has a cost."""
    values = {alias: cost for alias, cost in costs.items() if cost is not None}
    if not values:
        return None
    top = max(values.values())
    if top <= 0:
        return dict.fromkeys(costs, 0.0)
    return {
        alias: (cost / top if cost is not None else None)
        for alias, cost in costs.items()
    }


def check_routing_space(
    *,
    measured_lanes: list[str],
    accuracies: dict[str, dict[str, float | None]],
    costs: dict[str, dict[str, float | None]],
    min_advantage_pp: float,
) -> dict[str, Any]:
    """Sub-issue-8 §2.2 step 5: is there anything to route?

    ``accuracies[lane][alias]`` / ``costs[lane][alias]`` come from the calib
    profile; the "cheapest" model is the one with the lowest normalized
    per-question cost (ties broken alphabetically; models without a cost
    are never the cheapest - cost_basis is mandatory for the comparison to
    be economic).
    """
    # The cheapest model comes from the per-lane cost union; models without
    # a cost are never the cheapest.
    cost_union: dict[str, float] = {}
    for lane in measured_lanes:
        for alias, cost in (costs.get(lane) or {}).items():
            if cost is not None:
                cost_union[alias] = min(cost_union.get(alias, cost), cost)
    if cost_union:
        floor = min(cost_union.values())
        cheapest = sorted(alias for alias, cost in cost_union.items() if cost == floor)[
            0
        ]

    if cheapest is None or not measured_lanes:
        return {
            "verdict": "no_routing_space",
            "reason": (
                "no measured lane carries cost data: the advantage comparison "
                "needs a cheapest model (declare cost_basis assumed_pricing)"
            ),
            "cheapest_model": cheapest,
        }

    lane_advantages: dict[str, float] = {}
    first_places: set[str] = set()
    for lane in measured_lanes:
        accs = {
            alias: acc
            for alias, acc in (accuracies.get(lane) or {}).items()
            if acc is not None
        }
        if not accs:
            continue
        best = max(accs.values())
        first = sorted(a for a, v in accs.items() if v == best)[0]
        first_places.add(first)
        lane_advantages[lane] = best - accs.get(cheapest, 0.0)

    if lane_advantages and all(
        advantage * 100 < min_advantage_pp for advantage in lane_advantages.values()
    ):
        worst = min(lane_advantages.values())
        return {
            "verdict": "no_routing_space",
            "reason": (
                f"no measured lane shows a {min_advantage_pp:g}pp advantage of "
                f"the best model over the cheapest ({cheapest}); the largest "
                f"advantage is {worst * 100:.1f}pp"
            ),
            "cheapest_model": cheapest,
            "lane_advantages": {
                lane: round(advantage, 6)
                for lane, advantage in sorted(lane_advantages.items())
            },
        }
    if len(first_places) == 1:
        # The cheapest model is "acceptable" on a lane when it sits within
        # the advantage threshold of the best; condition B requires it to be
        # unacceptable on EVERY measured lane (advantage >= threshold
        # everywhere), so routing degenerates to the single winner.
        cheapest_never_acceptable = all(
            advantage * 100 >= min_advantage_pp
            for advantage in lane_advantages.values()
        )
        if cheapest_never_acceptable:
            winner = sorted(first_places)[0]
            return {
                "verdict": "no_routing_space",
                "reason": (
                    f"every measured lane prefers {winner} and the cheapest "
                    f"model ({cheapest}) is never within {min_advantage_pp:g}pp "
                    "of the best: routing degenerates to one model"
                ),
                "cheapest_model": cheapest,
                "first_places": sorted(first_places),
                "lane_advantages": {
                    lane: round(advantage, 6)
                    for lane, advantage in sorted(lane_advantages.items())
                },
            }
    return {"verdict": "route", "cheapest_model": cheapest}


def plan_lane(
    *,
    lane: str,
    template_lane: dict[str, Any],
    models: Any,
    profile: dict[str, Any],
    lam: float,
    min_lane_cases: int,
    lane_counts: dict[str, int],
) -> dict[str, Any]:
    """One lane's binding plan (sub-issue-8 §2.2 step 6).

    Returns the plan dict with ``selected`` refs, the ``measured`` flag, the
    selection basis and the per-model scores the sidecar records.
    """
    requirement = lane_requirements(lane)
    eligible = [
        model
        for model in models
        if capability_satisfied(requirement, model.capabilities)
    ]
    basis = "accuracy - lambda * normalized_cost"
    if template_lane["immediate"]:
        return {
            "lane": lane,
            "immediate": True,
            "measured": False,
            "n": lane_counts.get(lane, 0),
            "selected": [],
            "basis": "immediate response (no backend candidates)",
            "scores": {},
        }
    minimum = template_lane["minimum_candidates"]
    if not eligible:
        raise ContractError(
            f"lane {lane!r} requires capabilities "
            f"{requirement.get('capabilities') or ()} / context window "
            f"{requirement.get('min_context_window')} that none of the declared "
            "models satisfy; pass --exclude-decision "
            f"{lane} explicitly or declare the capability"
        )
    n = lane_counts.get(lane, 0)
    measured = n >= min_lane_cases
    if measured:
        costs = {
            model.name: _lane_stat(profile, model.name, lane, "cost_per_question")
            for model in eligible
        }
        normalized = _normalize_costs(costs)
        if lam > 0 and (normalized is None or any(c is None for c in costs.values())):
            raise ContractError(
                f"lane {lane!r} has incomplete cost data: a lambda > 0 would "
                "silently rank cost-less models as free; declare cost_basis "
                "assumed_pricing (or rerun profile) or drop the lambda"
            )

        def score(model) -> tuple[float, str]:
            acc = _lane_stat(profile, model.name, lane, "accuracy")
            acc = acc if acc is not None else 0.0
            norm = (normalized or {}).get(model.name)
            return acc - lam * (norm if norm is not None else 0.0), model.name

        ranked = sorted(eligible, key=lambda model: (-score(model)[0], score(model)[1]))
        basis_scores = {
            model.name: {
                "accuracy": _lane_stat(profile, model.name, lane, "accuracy"),
                "normalized_cost": (normalized or {}).get(model.name),
                "score": score(model)[0],
            }
            for model in eligible
        }
    else:
        # Unmeasured lane: default rule on the lambda-adjusted overall score,
        # marked untested (sub-issue-8 §2.2 step 6).
        basis = "default rule (unmeasured lane): overall accuracy - lambda * normalized overall cost"
        costs = {
            model.name: _overall_stat(profile, model.name, "cost_per_question")
            for model in eligible
        }
        normalized = _normalize_costs(costs)
        if lam > 0 and (normalized is None or any(c is None for c in costs.values())):
            raise ContractError(
                f"lane {lane!r} (unmeasured) has incomplete overall cost data: "
                "a lambda > 0 would silently rank cost-less models as free; "
                "declare cost_basis assumed_pricing or drop the lambda"
            )

        def oscore(model) -> tuple[float, str]:
            acc = _overall_stat(profile, model.name, "accuracy")
            acc = acc if acc is not None else 0.0
            norm = (normalized or {}).get(model.name)
            return acc - lam * (norm if norm is not None else 0.0), model.name

        ranked = sorted(
            eligible, key=lambda model: (-oscore(model)[0], oscore(model)[1])
        )
        basis_scores = {
            model.name: {
                "overall_accuracy": _overall_stat(profile, model.name, "accuracy"),
                "normalized_overall_cost": (normalized or {}).get(model.name),
                "score": oscore(model)[0],
            }
            for model in eligible
        }
    if len(ranked) < minimum:
        raise ContractError(
            f"lane {lane!r} requires {minimum} distinct candidates but only "
            f"{len(ranked)} model(s) satisfy its capability requirement; pass "
            f"--exclude-decision {lane} explicitly or declare more capable models"
        )
    selected = [model.name for model in ranked[:minimum]]
    return {
        "lane": lane,
        "immediate": False,
        "measured": measured,
        "n": n,
        "minimum_candidates": minimum,
        "selected": selected,
        "basis": basis,
        "scores": basis_scores,
    }


def plan_bindings(
    *,
    spec: ExperimentSpec,
    profile: dict[str, Any],
    template: dict[str, Any],
    lam: float,
    exclude: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Every lane's plan for one lambda, plus the routing-space verdict."""
    lanes = template["lanes"]
    unknown = sorted(set(exclude) - set(lanes))
    if unknown:
        raise ContractError(
            "--exclude-decision names lanes the template does not declare: "
            + ", ".join(unknown)
        )
    counts = lane_case_counts(profile)
    min_lane_cases = spec.bootstrap_config.min_lane_cases
    measured_lanes = [lane for lane, n in sorted(counts.items()) if n >= min_lane_cases]
    accuracies = {
        lane: {
            model.name: _lane_stat(profile, model.name, lane, "accuracy")
            for model in spec.models
        }
        for lane in measured_lanes
    }
    costs = {
        lane: {
            model.name: _lane_stat(profile, model.name, lane, "cost_per_question")
            for model in spec.models
        }
        for lane in measured_lanes
    }
    space = check_routing_space(
        measured_lanes=measured_lanes,
        accuracies=accuracies,
        costs=costs,
        min_advantage_pp=spec.bootstrap_config.min_advantage_pp,
    )
    plans: dict[str, dict[str, Any]] = {}
    for lane in sorted(lanes):
        if lane in exclude:
            plans[lane] = {
                "lane": lane,
                "excluded": True,
                "measured": counts.get(lane, 0) >= min_lane_cases,
                "n": counts.get(lane, 0),
                "selected": [],
                "basis": "explicitly excluded (--exclude-decision)",
                "scores": {},
            }
            continue
        plans[lane] = plan_lane(
            lane=lane,
            template_lane=lanes[lane],
            models=spec.models,
            profile=profile,
            lam=lam,
            min_lane_cases=min_lane_cases,
            lane_counts=counts,
        )
    bindings = {
        lane: [{"model": name} for name in plan["selected"]]
        for lane, plan in sorted(plans.items())
        if not plan.get("excluded")
    }
    return {
        "plans": plans,
        "bindings": bindings,
        "routing_space": space,
        "measured_lanes": measured_lanes,
        "unmeasured_lanes": sorted(
            lane
            for lane, plan in plans.items()
            if not plan.get("immediate") and not plan["measured"]
        ),
        "excluded_lanes": sorted(exclude),
    }


def _derived_eval_providers(
    providers: dict[str, Any],
) -> tuple[dict[str, Any], bool]:
    """The eval copy of the provider config: response cache off.

    Returns ``(config, cache_disabled)``; everything else is preserved
    verbatim (endpoints and credentials come only from the user's file).
    """
    config = copy.deepcopy(providers)
    global_section = config.setdefault("global", {})
    stores = global_section.setdefault("stores", {})
    cache = stores.get("response_cache")
    disabled = False
    if isinstance(cache, dict) and cache.get("enabled"):
        cache["enabled"] = False
        disabled = True
    elif not isinstance(cache, dict):
        stores["response_cache"] = {"enabled": False, "backend_type": "memory"}
        disabled = True
    return config, disabled


def run_bindings(
    *,
    spec: ExperimentSpec,
    profile_dir: str | Path,
    out_dir: str | Path,
    lambdas_raw: str | None,
    exclude: tuple[str, ...] = (),
    vllm_sr_bin: str = "vllm-sr",
    catalog_version: str = "latest",
    model_name: str = OBSERVE_MODEL_NAME,
    timeout_s: float = 300.0,
    run_json: Any = None,
) -> dict[str, Any]:
    """Run the bindings chain for every lambda (sub-issue-8 §2.2 steps 5-7).

    Per lambda: plan -> write ``bindings-lambda-<tag>.yaml`` -> derive the
    eval copy of ``providers_config`` -> ``recipe builtin init`` ->
    ``config validate`` -> emit ``candidate-lambda-<tag>.yaml`` plus the
    provenance sidecar. The routing-space check runs once (it does not
    depend on lambda) and a ``no_routing_space`` verdict ends the command
    with the report and no candidates.
    """
    runner = run_json or _default_run_json
    if spec.template is None:
        raise ContractError(
            "spec declares no template section: the bindings path generates "
            "configs from a built-in recipe template (sub-issue-8 §2.1)"
        )
    if not spec.providers_config:
        raise ContractError(
            "spec declares no providers_config: recipe builtin init consumes "
            "it (endpoints and credentials come only from there)"
        )
    providers_path = Path(spec.providers_config)
    if (
        not providers_path.is_file()
        and spec.spec_dir is not None
        and (Path(spec.spec_dir) / providers_path).is_file()
    ):
        providers_path = Path(spec.spec_dir) / providers_path
    if not providers_path.is_file():
        raise ContractError(f"providers_config not found: {providers_path}")

    profile_path = Path(profile_dir) / "profile.json"
    profile = load_profile(profile_path)

    template = template_info(
        bundle=spec.template.bundle,
        recipe=spec.template.recipe,
        vllm_sr_bin=vllm_sr_bin,
        catalog_version=catalog_version,
        timeout_s=timeout_s,
        run_json=runner,
    )
    lam_values = resolve_lambdas(spec, lambdas_raw)

    space_probe = plan_bindings(
        spec=spec,
        profile=profile,
        template=template,
        lam=lam_values[0],
        exclude=exclude,
    )["routing_space"]
    if space_probe["verdict"] != "route":
        return {
            "schema": BINDINGS_SCHEMA,
            "template": {
                "bundle": template["bundle"],
                "recipe": template["recipe"],
                "sha256": template["sha256"],
                "catalog_version": template["catalog_version"],
            },
            "profile": str(profile_path),
            "routing_space": space_probe,
            "candidates": [],
            "report": ("No benefit to routing: " + space_probe["reason"]),
        }

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    providers = yaml.safe_load(providers_path.read_text(encoding="utf-8"))
    if not isinstance(providers, dict):
        raise ContractError(f"{providers_path} is not a YAML mapping")
    eval_providers, cache_disabled = _derived_eval_providers(providers)
    eval_providers_path = out / "providers-eval.yaml"
    eval_providers_path.write_text(
        yaml.safe_dump(eval_providers, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )

    candidates: list[dict[str, Any]] = []
    for lam in lam_values:
        tag = lambda_tag(lam)
        plan = plan_bindings(
            spec=spec,
            profile=profile,
            template=template,
            lam=lam,
            exclude=exclude,
        )
        bindings_path = out / f"bindings-lambda-{tag}.yaml"
        bindings_path.write_text(
            yaml.safe_dump(plan["bindings"], sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
        candidate_path = out / f"candidate-lambda-{tag}.yaml"
        init_argv = [
            str(vllm_sr_bin),
            "recipe",
            "builtin",
            "init",
            str(template["recipe"]),
            "--bundle",
            str(template["bundle"]),
            "--config",
            str(eval_providers_path),
            "--bindings",
            str(bindings_path),
            "--model-name",
            str(model_name),
            "--output",
            str(candidate_path),
        ]
        for lane in plan["excluded_lanes"]:
            init_argv += ["--exclude-decision", lane]
        runner(init_argv, timeout_s=timeout_s, what="recipe builtin init")
        if not candidate_path.is_file():
            raise ContractError(
                "recipe builtin init reported success but wrote no config: "
                f"{candidate_path}"
            )
        reason = validate_config(candidate_path, vllm_sr_bin)
        if reason is not None:
            raise ContractError(
                f"candidate for lambda {tag} was rejected by config validate: {reason}"
            )

        sidecar = {
            "schema": BINDINGS_SCHEMA,
            "proposer": "bindings",
            "lambda": lam,
            "template": {
                "bundle": template["bundle"],
                "recipe": template["recipe"],
                "sha256": template["sha256"],
                "catalog_version": template["catalog_version"],
            },
            "profile": {
                "path": str(profile_path),
                "sha256": dataset_content_hash(profile_path),
            },
            "providers": {
                "input": str(providers_path),
                "input_sha256": dataset_content_hash(providers_path),
                "eval_copy": str(eval_providers_path),
                "eval_copy_sha256": dataset_content_hash(eval_providers_path),
                "cache_disabled": cache_disabled,
                "cache_disabled_reason": (
                    CACHE_DISABLED_REASON if cache_disabled else None
                ),
            },
            "bindings": {
                "path": str(bindings_path),
                "sha256": dataset_content_hash(bindings_path),
            },
            "candidate": {
                "path": str(candidate_path),
                "sha256": dataset_content_hash(candidate_path),
            },
            "init_command": init_argv,
            "routing_space": plan["routing_space"],
            "measured_lanes": plan["measured_lanes"],
            "unmeasured_lanes": plan["unmeasured_lanes"],
            "excluded_lanes": plan["excluded_lanes"],
            "lanes": plan["plans"],
        }
        sidecar_path = out / f"bindings-lambda-{tag}.provenance.json"
        sidecar_path.write_text(
            json.dumps(sidecar, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        candidates.append(
            {
                "lambda": lam,
                "tag": tag,
                "bindings": str(bindings_path),
                "candidate": str(candidate_path),
                "provenance": str(sidecar_path),
                "measured_lanes": plan["measured_lanes"],
                "unmeasured_lanes": plan["unmeasured_lanes"],
                "excluded_lanes": plan["excluded_lanes"],
            }
        )

    return {
        "schema": BINDINGS_SCHEMA,
        "template": {
            "bundle": template["bundle"],
            "recipe": template["recipe"],
            "sha256": template["sha256"],
            "catalog_version": template["catalog_version"],
        },
        "profile": str(profile_path),
        "providers_eval_copy": str(eval_providers_path),
        "routing_space": space_probe,
        "candidates": candidates,
    }
