"""Bindings (C4+C5 template path) unit tests - sub-issue-8 §2.7 golden scenarios.

All subprocess boundaries are faked (recipe builtin list/init via run_json,
config validate via monkeypatched validate_config), so the suite is
deterministic and platform-independent.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from recipe_lab import bindings as bindings_mod
from recipe_lab.contract import ContractError
from recipe_lab.observe_lanes import (
    capability_satisfied,
    lane_requirements,
)

API_VERSION = "recipe-lab/v1alpha1"

# A small template with the shapes the golden scenarios need: three
# single-candidate measured lanes, one unmeasured lane, one two-candidate
# lane (frontier semantics), and an immediate lane.
TEMPLATE_LANES = {
    "code": {"algorithm": "static", "minimum_candidates": 1, "immediate": False},
    "hard": {"algorithm": "multi_factor", "minimum_candidates": 1, "immediate": False},
    "facts": {"algorithm": "static", "minimum_candidates": 1, "immediate": False},
    "fast": {"algorithm": "static", "minimum_candidates": 1, "immediate": False},
    "frontier": {"algorithm": "decision", "minimum_candidates": 2, "immediate": False},
    "vision": {"algorithm": "static", "minimum_candidates": 1, "immediate": False},
    "agentic": {"algorithm": "static", "minimum_candidates": 1, "immediate": False},
    "standard": {"algorithm": "multi_factor", "minimum_candidates": 1, "immediate": False},
    "long_context": {"algorithm": "static", "minimum_candidates": 1, "immediate": False},
    "recovery": {"algorithm": "static", "minimum_candidates": 1, "immediate": False},
}


def fake_template():
    return {
        "bundle": "mom-v1",
        "recipe": "balance",
        "sha256": "sha256:template",
        "catalog_version": "latest",
        "lanes": dict(TEMPLATE_LANES),
    }


def write_spec(
    tmp_path: Path,
    *,
    capabilities: dict[str, dict] | None = None,
    lambdas: list[float] | None = None,
    min_lane_cases: int = 20,
    with_pricing: bool = True,
) -> Path:
    """A minimal valid spec declaring the template path.

    Without an explicit ``capabilities`` mapping both models declare the
    default set (tools + a 256K window; A also vision), so every template
    lane is satisfiable - the capability-filter tests pass their own
    restricted mapping.
    """
    default_caps = {
        "local/model-a": {"vision": True, "tools": True, "context_window": 262144},
        "local/model-b": {"tools": True, "context_window": 262144},
    }
    models = []
    for alias, endpoint in (
        ("local/model-a", "http://127.0.0.1:9001/v1"),
        ("local/model-b", "http://127.0.0.1:9002/v1"),
    ):
        entry = {"name": alias, "endpoint": endpoint, "served_name": alias}
        if capabilities is None:
            entry["capabilities"] = default_caps[alias]
        elif alias in capabilities:
            entry["capabilities"] = capabilities[alias]
        # else: the capability key is omitted entirely (unsatisfiable lanes)
        models.append(entry)
    spec: dict = {
        "api_version": API_VERSION,
        "objective": {
            "metric_key": "capability.score",
            "direction": "maximize",
            "min_improvement_pct": 1.0,
            "replay_metric": "routed_accuracy",
        },
        "workload": {
            "manifest": "dev.jsonl",
            "content_hash": "sha256:" + "0" * 64,
            "profile": "quick",
        },
        "attribution": {"oracle_policy": {"math": "local/model-a"}},
        "budget": {"rounds": 1, "max_minutes_per_trial": 30},
        "models": models,
        "template": {"bundle": "mom-v1", "recipe": "balance"},
        "providers_config": "providers.yaml",
        "bootstrap": {"min_lane_cases": min_lane_cases},
    }
    if lambdas is not None:
        spec["bootstrap"]["lambdas"] = lambdas
    if with_pricing:
        spec["cost_basis"] = {
            "source": "provider_pricing",
            "assumed_pricing": {
                "local/model-a": {"prompt_per_1m": 0.1, "completion_per_1m": 0.4},
                "local/model-b": {"prompt_per_1m": 0.4, "completion_per_1m": 1.6},
            },
        }
    (tmp_path / "providers.yaml").write_text(
        yaml.safe_dump(
            {
                "version": "v0.3",
                "listeners": [
                    {"name": "gw", "address": "0.0.0.0", "port": 8899, "timeout": "300s"}
                ],
                "providers": {
                    "defaults": {"model": "local/model-a"},
                    "models": [
                        {
                            "name": alias,
                            "provider_model_id": alias,
                            "api_format": "openai",
                            "backend_refs": [
                                {
                                    "name": alias + "-backend",
                                    "endpoint": "127.0.0.1:9001",
                                    "protocol": "http",
                                    "provider": "vllm",
                                    "weight": 1,
                                }
                            ],
                        }
                        for alias in ("local/model-a", "local/model-b")
                    ],
                },
                "routing": {
                    "modelCards": [
                        {
                            "name": alias,
                            "context_window_size": 131072,
                            "capabilities": ["chat", "text"],
                            "modality": "text",
                        }
                        for alias in ("local/model-a", "local/model-b")
                    ]
                },
                "global": {
                    "stores": {
                        "response_cache": {"enabled": True, "backend_type": "memory"}
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "dev.jsonl").write_text("", encoding="utf-8")
    spec_path = tmp_path / "spec.yaml"
    spec_path.write_text(yaml.safe_dump(spec), encoding="utf-8")
    return spec_path


def write_profile(
    tmp_path: Path,
    *,
    lane_stats: dict[str, dict[str, dict[str, float]]],
    overall: dict[str, dict[str, float]],
) -> Path:
    """A recipe-lab-profile-v1 document (lane mode) the bindings consume.

    ``lane_stats``: {lane: {alias: {"n": int, "accuracy": float,
    "cost_per_question": float}}}; every lane's n is shared by all models
    (all models answer the same tasks).
    """
    stats: dict[str, dict] = {}
    for alias in ("local/model-a", "local/model-b"):
        labels = {
            lane: {
                "n": block["n"],
                "accuracy": block[alias]["accuracy"],
                "cost_per_question": block[alias]["cost_per_question"],
                "accuracy_ci95": [0.0, 1.0],
            }
            for lane, block in lane_stats.items()
        }
        stats[alias] = {
            "labels": labels,
            "total": {
                "n": 100,
                "accuracy": overall[alias]["accuracy"],
                "cost_per_question": overall[alias]["cost_per_question"],
            },
        }
    profile = {
        "schema": "recipe-lab-profile-v1",
        "grouping": "lanes",
        "splits": ["calib"],
        "models": ["local/model-a", "local/model-b"],
        "label_merges": {},
        "uncovered_cases": {},
        "unmeasured_lanes": {},
        "quality": {},
        "model_split_stats": {"calib": stats},
        "inputs": {},
    }
    profile_dir = tmp_path / "profile"
    profile_dir.mkdir(parents=True, exist_ok=True)
    (profile_dir / "profile.json").write_text(
        json.dumps(profile), encoding="utf-8"
    )
    return profile_dir


def fake_run_json_factory(outputs: dict[str, str]):
    """A run_json stand-in: any `init` argv writes the canned output file."""

    def run_json(argv, *, timeout_s, what):
        if "init" in argv:
            out = argv[argv.index("--output") + 1]
            Path(out).write_text(outputs.get("init", "version: v0.3\n"), encoding="utf-8")
            return {"valid": True}
        return {"bundles": []}

    return run_json


def plan(tmp_path, *, profile_dir, lam, exclude=(), spec_path=None):
    spec_path = spec_path or write_spec(tmp_path)
    from recipe_lab.contract import ExperimentSpec

    spec = ExperimentSpec.from_yaml(spec_path)
    profile = json.loads((profile_dir / "profile.json").read_text(encoding="utf-8"))
    return bindings_mod.plan_bindings(
        spec=spec,
        profile=profile,
        template=fake_template(),
        lam=lam,
        exclude=exclude,
    )


# --------------------------------------------------------------------------
# Golden scenario (sub-issue-8 §2.7): model A is fine on facts/fast, model B
# is clearly better on code/hard and 4x pricier.
# --------------------------------------------------------------------------

def golden_profile(tmp_path):
    n = {"code": 50, "hard": 50, "facts": 50, "fast": 50, "frontier": 0}
    lane_stats = {}
    for lane, (a_acc, b_acc) in {
        "code": (0.60, 0.90),
        "hard": (0.55, 0.85),
        "facts": (0.80, 0.82),
        "fast": (0.85, 0.84),
    }.items():
        lane_stats[lane] = {
            "n": n[lane],
            "local/model-a": {"accuracy": a_acc, "cost_per_question": 0.001},
            "local/model-b": {"accuracy": b_acc, "cost_per_question": 0.004},
        }
    overall = {
        "local/model-a": {"accuracy": 0.72, "cost_per_question": 0.001},
        "local/model-b": {"accuracy": 0.85, "cost_per_question": 0.004},
    }
    return write_profile(tmp_path, lane_stats=lane_stats, overall=overall)


def test_golden_small_lambda_binds_strong_model_to_hard_lanes(tmp_path):
    profile_dir = golden_profile(tmp_path)
    result = plan(tmp_path, profile_dir=profile_dir, lam=0.0)
    plans = result["plans"]
    assert plans["code"]["selected"] == ["local/model-b"]
    assert plans["hard"]["selected"] == ["local/model-b"]
    assert plans["fast"]["selected"] == ["local/model-a"]
    # unmeasured lane (frontier n=0 < 20): default rule by overall score,
    # marked untested.
    assert plans["frontier"]["measured"] is False
    assert plans["frontier"]["selected"] == ["local/model-b", "local/model-a"]
    assert "unmeasured" in plans["frontier"]["basis"]
    assert result["routing_space"]["verdict"] == "route"


def test_golden_large_lambda_rebinds_hard_to_cheapest(tmp_path):
    profile_dir = golden_profile(tmp_path)
    # lambda = 100: 0.30 accuracy advantage (code) < 100 * (0.004-0.001) = 0.3
    # normalized cost gap 1.0 vs 0.25 -> hard's 0.30 advantage loses to 0.75.
    result = plan(tmp_path, profile_dir=profile_dir, lam=100.0)
    plans = result["plans"]
    # code keeps B (0.30 advantage vs 0.75 cost penalty? 0.90-100*1.0=-10 vs
    # 0.60-100*0.25=-24.4 -> B still wins); hard: A 0.55-25=-24.45 vs B
    # 0.85-100=-99.15 -> A wins.
    assert plans["hard"]["selected"] == ["local/model-a"]
    assert plans["fast"]["selected"] == ["local/model-a"]


def test_frontier_requires_two_distinct_candidates(tmp_path):
    profile_dir = golden_profile(tmp_path)
    result = plan(tmp_path, profile_dir=profile_dir, lam=0.0)
    selected = result["plans"]["frontier"]["selected"]
    assert len(selected) == 2
    assert len(set(selected)) == 2


def test_capability_filter_refuses_unsatisfied_lane_without_explicit_exclude(tmp_path):
    profile_dir = golden_profile(tmp_path)
    spec_path = write_spec(tmp_path, capabilities={})
    from recipe_lab.contract import ExperimentSpec

    spec = ExperimentSpec.from_yaml(spec_path)
    with pytest.raises(ContractError, match="agentic"):
        bindings_mod.plan_bindings(
            spec=spec,
            profile=json.loads(
                (profile_dir / "profile.json").read_text(encoding="utf-8")
            ),
            template=fake_template(),
            lam=0.0,
        )


def test_capability_filter_with_explicit_exclude(tmp_path):
    profile_dir = golden_profile(tmp_path)
    spec_path = write_spec(
        tmp_path,
        capabilities={
            "local/model-a": {
                "vision": True,
                "tools": True,
                "context_window": 262144,
            },
            "local/model-b": {"tools": True, "context_window": 262144},
        },
    )
    result = plan(tmp_path, profile_dir=profile_dir, lam=0.0, spec_path=spec_path)
    # vision now has one eligible model (A); agentic has both.
    assert result["plans"]["vision"]["selected"] == ["local/model-a"]
    assert result["plans"]["agentic"]["measured"] is False
    assert len(result["plans"]["agentic"]["selected"]) == 1


def test_long_context_requires_declared_window(tmp_path):
    requirement = lane_requirements("long_context")
    from recipe_lab.contract import ModelCapabilities

    assert not capability_satisfied(
        requirement, ModelCapabilities(context_window=131072)
    )
    assert capability_satisfied(
        requirement, ModelCapabilities(context_window=262144)
    )
    # Unlisted lanes carry no requirement.
    assert lane_requirements("fast") == {
        "capabilities": (),
        "min_context_window": None,
    }


def test_no_routing_space_report_and_no_candidates(tmp_path, monkeypatch):
    # Degenerate: model B is better everywhere by far and A is never within
    # 5pp -> routing degenerates to one model.
    lane_stats = {
        lane: {
            "n": 50,
            "local/model-a": {"accuracy": 0.60, "cost_per_question": 0.001},
            "local/model-b": {"accuracy": 0.95, "cost_per_question": 0.004},
        }
        for lane in ("code", "hard", "facts", "fast")
    }
    overall = {
        "local/model-a": {"accuracy": 0.60, "cost_per_question": 0.001},
        "local/model-b": {"accuracy": 0.95, "cost_per_question": 0.004},
    }
    profile_dir = write_profile(tmp_path, lane_stats=lane_stats, overall=overall)
    spec_path = write_spec(tmp_path, lambdas=[0.0])
    from recipe_lab.contract import ExperimentSpec

    spec = ExperimentSpec.from_yaml(spec_path)
    monkeypatch.setattr(
        bindings_mod, "template_info", lambda **kwargs: fake_template()
    )
    result = bindings_mod.run_bindings(
        spec=spec,
        profile_dir=profile_dir,
        out_dir=tmp_path / "cands",
        lambdas_raw=None,
        vllm_sr_bin="unused",
        run_json=fake_run_json_factory({}),
    )
    assert result["candidates"] == []
    assert result["routing_space"]["verdict"] == "no_routing_space"
    assert "No benefit to routing" in result["report"]


def test_run_bindings_generates_and_validates_candidates(tmp_path, monkeypatch):
    profile_dir = golden_profile(tmp_path)
    spec_path = write_spec(
        tmp_path,
        capabilities={
            "local/model-a": {
                "vision": True,
                "tools": True,
                "context_window": 262144,
            },
            "local/model-b": {"tools": True, "context_window": 262144},
        },
        lambdas=[0.0, 100.0],
    )
    from recipe_lab.contract import ExperimentSpec

    spec = ExperimentSpec.from_yaml(spec_path)
    monkeypatch.setattr(
        bindings_mod, "template_info", lambda **kwargs: fake_template()
    )
    captured: list[list[str]] = []

    def run_json(argv, *, timeout_s, what):
        captured.append(list(argv))
        if "init" in argv:
            out = Path(argv[argv.index("--output") + 1])
            out.write_text("version: v0.3\n", encoding="utf-8")
            return {"valid": True}
        return {"bundles": []}

    monkeypatch.setattr(
        bindings_mod, "validate_config", lambda path, bin: None
    )
    result = bindings_mod.run_bindings(
        spec=spec,
        profile_dir=profile_dir,
        out_dir=tmp_path / "cands",
        lambdas_raw=None,
        vllm_sr_bin="fake-vllm-sr",
        run_json=run_json,
    )
    assert [c["lambda"] for c in result["candidates"]] == [0.0, 100.0]
    inits = [argv for argv in captured if "init" in argv]
    assert len(inits) == 2
    for argv in inits:
        assert "--config" in argv and "--bindings" in argv
        assert "--model-name" in argv
    # The eval copy disabled the bundle's response cache.
    eval_copy = yaml.safe_load(
        (tmp_path / "cands" / "providers-eval.yaml").read_text(encoding="utf-8")
    )
    assert eval_copy["global"]["stores"]["response_cache"]["enabled"] is False
    # The input provider config is untouched (endpoints come only from there).
    original = yaml.safe_load(
        (tmp_path / "providers.yaml").read_text(encoding="utf-8")
    )
    assert original["global"]["stores"]["response_cache"]["enabled"] is True
    # Bindings are the raw {lane: refs} mapping init consumes.
    bindings_doc = yaml.safe_load(
        (tmp_path / "cands" / "bindings-lambda-0.yaml").read_text(encoding="utf-8")
    )
    assert bindings_doc["frontier"][0]["model"]
    assert bindings_doc["code"] == [{"model": "local/model-b"}]
    # Sidecar records the template hash and the per-lane basis.
    sidecar = json.loads(
        (tmp_path / "cands" / "bindings-lambda-0.provenance.json").read_text(
            encoding="utf-8"
        )
    )
    assert sidecar["template"]["sha256"] == "sha256:template"
    assert sidecar["lanes"]["frontier"]["measured"] is False
    assert sidecar["unmeasured_lanes"] == [
        "agentic",
        "frontier",
        "long_context",
        "recovery",
        "standard",
        "vision",
    ]


def test_lambda_above_zero_without_costs_is_rejected(tmp_path):
    # Measured lane without cost data + lambda > 0 -> explicit error, never
    # a silent degrade to pure accuracy (the §3.2-5 lesson).
    lane_stats = {
        "code": {
            "n": 50,
            "local/model-a": {"accuracy": 0.60, "cost_per_question": None},
            "local/model-b": {"accuracy": 0.90, "cost_per_question": 0.004},
        },
        "hard": {
            "n": 50,
            "local/model-a": {"accuracy": 0.55, "cost_per_question": None},
            "local/model-b": {"accuracy": 0.85, "cost_per_question": 0.004},
        },
        "facts": {
            "n": 50,
            "local/model-a": {"accuracy": 0.80, "cost_per_question": None},
            "local/model-b": {"accuracy": 0.82, "cost_per_question": 0.004},
        },
        "fast": {
            "n": 50,
            "local/model-a": {"accuracy": 0.85, "cost_per_question": None},
            "local/model-b": {"accuracy": 0.84, "cost_per_question": 0.004},
        },
    }
    overall = {
        "local/model-a": {"accuracy": 0.72, "cost_per_question": None},
        "local/model-b": {"accuracy": 0.85, "cost_per_question": 0.004},
    }
    profile_dir = write_profile(tmp_path, lane_stats=lane_stats, overall=overall)
    spec_path = write_spec(tmp_path, lambdas=[0.5])
    from recipe_lab.contract import ExperimentSpec

    spec = ExperimentSpec.from_yaml(spec_path)
    with pytest.raises(ContractError, match="cost"):
        bindings_mod.plan_bindings(
            spec=spec,
            profile=json.loads(
                (profile_dir / "profile.json").read_text(encoding="utf-8")
            ),
            template=fake_template(),
            lam=0.5,
        )


def test_unknown_exclude_lane_is_rejected(tmp_path):
    profile_dir = golden_profile(tmp_path)
    spec_path = write_spec(tmp_path)
    from recipe_lab.contract import ExperimentSpec

    spec = ExperimentSpec.from_yaml(spec_path)
    with pytest.raises(ContractError, match="does not declare"):
        bindings_mod.plan_bindings(
            spec=spec,
            profile=json.loads(
                (profile_dir / "profile.json").read_text(encoding="utf-8")
            ),
            template=fake_template(),
            lam=0.0,
            exclude=("nonexistent-lane",),
        )


def test_routing_space_no_advantage(tmp_path):
    # Condition A: the best model's advantage over the cheapest is < 5pp on
    # every measured lane.
    lane_stats = {
        lane: {
            "n": 50,
            "local/model-a": {"accuracy": 0.80, "cost_per_question": 0.001},
            "local/model-b": {"accuracy": 0.83, "cost_per_question": 0.004},
        }
        for lane in ("code", "hard")
    }
    overall = {
        "local/model-a": {"accuracy": 0.80, "cost_per_question": 0.001},
        "local/model-b": {"accuracy": 0.83, "cost_per_question": 0.004},
    }
    profile_dir = write_profile(tmp_path, lane_stats=lane_stats, overall=overall)
    result = plan(tmp_path, profile_dir=profile_dir, lam=0.0)
    assert result["routing_space"]["verdict"] == "no_routing_space"


def test_bindings_output_is_byte_stable(tmp_path, monkeypatch):
    profile_dir = golden_profile(tmp_path)
    spec_path = write_spec(tmp_path, lambdas=[0.0])
    from recipe_lab.contract import ExperimentSpec

    spec = ExperimentSpec.from_yaml(spec_path)
    monkeypatch.setattr(
        bindings_mod, "template_info", lambda **kwargs: fake_template()
    )
    monkeypatch.setattr(bindings_mod, "validate_config", lambda path, bin: None)

    def run_json(argv, *, timeout_s, what):
        if "init" in argv:
            Path(argv[argv.index("--output") + 1]).write_text(
                "version: v0.3\n", encoding="utf-8"
            )
            return {"valid": True}
        return {"bundles": []}

    digests = []
    for run in ("run1", "run2"):
        out = tmp_path / run
        bindings_mod.run_bindings(
            spec=spec,
            profile_dir=profile_dir,
            out_dir=out,
            lambdas_raw=None,
            vllm_sr_bin="fake-vllm-sr",
            run_json=run_json,
        )
        # Volatile fields (absolute output paths, hash-of-paths) are excluded;
        # everything selection-relevant must be identical across runs.
        stable_keys = (
            "template",
            "routing_space",
            "measured_lanes",
            "unmeasured_lanes",
            "excluded_lanes",
            "lanes",
        )
        digests.append(
            (out / "bindings-lambda-0.yaml").read_bytes()
            + json.dumps(
                {
                    k: v
                    for k, v in json.loads(
                        (out / "bindings-lambda-0.provenance.json").read_text(
                            encoding="utf-8"
                        )
                    ).items()
                    if k in stable_keys
                },
                sort_keys=True,
            ).encode("utf-8")
        )
    assert digests[0] == digests[1]
