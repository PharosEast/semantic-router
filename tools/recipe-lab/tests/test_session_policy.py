"""AG6: the session-policy proposer - diagnosis-triggered one-variable
candidates, and the golden flow (a high-switch diagnosis produces the
stability candidate, and the raised candidate switches less on replay)."""

from __future__ import annotations

import sys

import pytest
import yaml

from recipe_lab.proposers.session_policy import propose_session_policy

BASE_CONFIG = {
    "version": "v0.2",
    "global": {
        "router": {
            "learning": {
                "adaptation": {"enabled": False},
                "protection": {
                    "enabled": True,
                    "scope": "conversation",
                    "identity": {
                        "headers": {
                            "session": "x-session-id",
                            "conversation": "x-conversation-id",
                        }
                    },
                    "tuning": {
                        "stability_weight": 1.0,
                        "switch_margin": 0.05,
                        "min_turns_before_switch": 1,
                    },
                },
            }
        }
    },
}

HIGH_SWITCH_SUMMARY = {
    "records": 3,
    "sessions": 1,
    "model_switches": 2,
    "tool_loop_switches": 2,
    "selection_rejected": 0,
    "tool_loop_interrupted": 0,
    "mean_switches_per_session": 2.0,
}

NOT_SEPARATE_DIAGNOSIS = {
    "agentic": {
        "agent_separation": {
            "status": "not_separate",
            "agent_requests": 0,
            "requests_missing_identity": 4,
            "reason": "protection identity missing on 4 request(s)",
        }
    }
}


def _variables_of(candidates):
    return [c.variable for c in candidates]


def test_no_evidence_proposes_nothing():
    assert propose_session_policy(BASE_CONFIG) == []
    assert propose_session_policy(BASE_CONFIG, replay_summary={}) == []


def test_tool_loop_switches_raise_stability_weight():
    candidates = propose_session_policy(BASE_CONFIG, replay_summary=HIGH_SWITCH_SUMMARY)
    variables = _variables_of(candidates)
    assert "stability_weight" in variables
    candidate = next(c for c in candidates if c.variable == "stability_weight")
    assert candidate.old == 1.0
    assert candidate.new == 2.0
    assert candidate.trigger["reason"] == "tool_loop_switches"
    # exactly one variable changed in the candidate config
    changed = _diff(BASE_CONFIG, candidate.config)
    assert changed == [("global.router.learning.protection.tuning", "stability_weight")]
    assert "stability_weight: 2" in yaml.safe_dump(candidate.config)


def test_model_flips_raise_switch_margin():
    summary = dict(HIGH_SWITCH_SUMMARY, tool_loop_switches=0)
    candidates = propose_session_policy(BASE_CONFIG, replay_summary=summary)
    assert "switch_margin" in _variables_of(candidates)
    candidate = next(c for c in candidates if c.variable == "switch_margin")
    assert candidate.old == 0.05
    assert candidate.new == 0.1


def test_not_separate_separation_adds_the_agent_header():
    candidates = propose_session_policy(BASE_CONFIG, diagnosis=NOT_SEPARATE_DIAGNOSIS)
    assert "identity.headers.agent" in _variables_of(candidates)
    candidate = next(c for c in candidates if c.variable == "identity.headers.agent")
    protection = candidate.config["global"]["router"]["learning"]["protection"]
    assert protection["identity"]["headers"]["agent"] == "x-claude-code-agent-id"
    assert (
        "agent"
        not in BASE_CONFIG["global"]["router"]["learning"]["protection"]["identity"][
            "headers"
        ]
    )


def test_rejected_selections_raise_min_turns():
    summary = dict(HIGH_SWITCH_SUMMARY, tool_loop_switches=0, selection_rejected=3)
    candidates = propose_session_policy(BASE_CONFIG, replay_summary=summary)
    candidate = next(c for c in candidates if c.variable == "min_turns_before_switch")
    assert candidate.old == 1
    assert candidate.new == 2


def test_exhausted_knobs_produce_no_candidate():
    raised = yaml.safe_load(yaml.safe_dump(BASE_CONFIG))  # deep copy
    raised["global"]["router"]["learning"]["protection"]["tuning"][
        "stability_weight"
    ] = 2.0
    candidates = propose_session_policy(raised, replay_summary=HIGH_SWITCH_SUMMARY)
    # the stability knob is at its raised value: no stability candidate even
    # though the tool-loop switches still fire (nothing left to change)
    assert "stability_weight" not in _variables_of(candidates)
    # and with every knob raised there is nothing left to propose
    raised["global"]["router"]["learning"]["protection"]["tuning"][
        "switch_margin"
    ] = 0.2
    summary = dict(HIGH_SWITCH_SUMMARY, selection_rejected=0)
    assert propose_session_policy(raised, replay_summary=summary) == []


def _diff(base, candidate):
    """The (path, key) pairs where the candidate differs from the base."""
    diffs = []

    def walk(b, c, path):
        if isinstance(b, dict) and isinstance(c, dict):
            for key in set(b) | set(c):
                walk(b.get(key), c.get(key), path + [str(key)])
        elif b != c:
            diffs.append((".".join(path[:-1]), path[-1]))

    walk(base, candidate, [])
    return diffs


# ---- golden flow: the raised candidate switches less on replay ----


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="the runner's process-group teardown needs os.killpg (POSIX)",
)
def test_high_switch_diagnosis_candidate_switches_less_on_replay(tmp_path):
    from pathlib import Path

    from tests.test_route_replay import _build_route_stack, _session_case

    cases = [_session_case("s1", "conv-ag6")]
    stack = _build_route_stack(tmp_path)

    # the base candidate flaps mid-loop
    base_config, base_env = stack("unprotected")
    base_result = evaluate(base_config, base_env, tmp_path, "base")
    assert base_result["summary"]["tool_loop_switches"] >= 1

    # the high-switch evidence triggers the stability candidate
    base_spec = yaml.safe_load(Path(base_config).read_text(encoding="utf-8"))
    candidates = propose_session_policy(
        base_spec, replay_summary=base_result["summary"]
    )
    stability = next(c for c in candidates if c.variable == "stability_weight")
    assert stability.new == 2.0

    # the raised candidate holds the lane through the tool loop
    raised_config, raised_env = stack("unprotected", stability_weight=2.0)
    raised = evaluate(raised_config, raised_env, tmp_path, "raised")
    assert raised["summary"]["tool_loop_switches"] == 0


def evaluate(config, env, tmp_path, name):
    from recipe_lab.runner import evaluate_route_replay
    from tests.test_route_replay import _session_case

    env = RunnerEnvLike(env, tmp_path, name)
    return evaluate_route_replay(
        candidate_config=config,
        cases=[_session_case("s1", "conv-ag6")],
        env=env.env,
        router_model="vllm-sr/auto",
        request_timeout=30.0,
        trial_timeout_s=60,
        request_id_prefix=f"rl-rt-{name}-",
        gates={"max_tool_loop_switches": 0},
    )


class RunnerEnvLike:
    """A per-evaluation RunnerEnv (fresh work dir and backend port)."""

    def __init__(self, env, tmp_path, name):
        from tests.test_route_replay import _free_port

        import dataclasses

        env_dict = dataclasses.asdict(env)
        env_dict["work_dir"] = tmp_path / f"work-{name}"
        env_dict["backend_port"] = _free_port()
        self.env = dataclasses.replace(env, **env_dict)
