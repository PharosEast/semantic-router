"""Session-policy proposer (AG6): diagnosis-triggered one-variable candidates.

Each candidate changes exactly one session-protection variable of the base
config, and is only proposed when the evidence asks for it:

- tool-loop switches (route-replay summary) -> raise `stability_weight`
- model flips between sessions (route-replay summary) -> raise
  `switch_margin`
- sub-agent separation reported broken (the AG4 diagnosis's
  `agent_separation`) -> add the agent identity header
- rejected selections (route-replay summary) -> raise
  `min_turns_before_switch`

Evidence the proposer never saw produces no candidate - a proposal always
names the trigger that asked for it. The candidate configs then go through
the AG3 route-replay gates before anything reaches the live tier; the AG8
golden flow asserts a high-switch diagnosis produces the stability
candidate and that the raised candidate actually switches less.

Pure functions, no I/O, deterministic: same inputs -> same candidates.
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# The agent identity header Claude Code natively sends on delegated
# subagents (AG0-W2).
AGENT_HEADER = "x-claude-code-agent-id"


@dataclass(frozen=True)
class SessionPolicyCandidate:
    variable: str
    old: Any
    new: Any
    trigger: dict[str, Any]
    ref: str
    config: dict[str, Any]

    def summary(self) -> str:
        return (
            f"session policy: {self.variable} {self.old!r} -> {self.new!r} "
            f"(trigger: {self.trigger.get('reason', '')})"
        )


def _protection(config: dict[str, Any]) -> dict[str, Any]:
    learning = (
        config.get("global", {}).get("router", {}).get("learning")
        if isinstance(config.get("global"), dict)
        else None
    )
    protection = learning.get("protection") if isinstance(learning, dict) else None
    if not isinstance(protection, dict):
        raise ValueError(
            "config carries no global.router.learning.protection section; "
            "a session-policy candidate needs one to change"
        )
    return protection


def _tuning(config: dict[str, Any]) -> dict[str, Any]:
    protection = _protection(config)
    tuning = protection.get("tuning")
    if not isinstance(tuning, dict):
        tuning = {}
        protection["tuning"] = tuning
    return tuning


def _ref(variable: str, new: Any) -> str:
    payload = json.dumps({"variable": variable, "new": new}, sort_keys=True)
    return (
        "config-rev:session-policy:" + hashlib.sha256(payload.encode()).hexdigest()[:12]
    )


def _candidate(
    base: dict[str, Any], variable: str, old: Any, new: Any, trigger: dict[str, Any]
) -> SessionPolicyCandidate:
    return SessionPolicyCandidate(
        variable=variable,
        old=old,
        new=new,
        trigger=trigger,
        ref=_ref(variable, new),
        config=copy.deepcopy(base),
    )


def _stability_candidate(base, tuning, summary) -> SessionPolicyCandidate | None:
    switches = summary.get("tool_loop_switches") or 0
    if switches <= 0:
        return None
    current = tuning.get("stability_weight", 1.0)
    if current >= 2.0:
        return None  # already at the raised weight; the knob is exhausted
    new = 2.0
    candidate = _candidate(
        base,
        "stability_weight",
        current,
        new,
        {
            "reason": "tool_loop_switches",
            "tool_loop_switches": switches,
            "note": "switches inside the tool loop violate the stability gate",
        },
    )
    _tuning(candidate.config)["stability_weight"] = new
    return candidate


def _switch_margin_candidate(base, tuning, summary) -> SessionPolicyCandidate | None:
    flips = summary.get("model_switches") or 0
    mean = summary.get("mean_switches_per_session")
    if flips < 2 and not (mean is not None and mean > 0.5):
        return None
    current = tuning.get("switch_margin", 0.05)
    if current >= 0.1:
        return None  # the margin is already at its proposed value
    candidate = _candidate(
        base,
        "switch_margin",
        current,
        0.1,
        {
            "reason": "model_switches",
            "model_switches": flips,
            "mean_switches_per_session": mean,
            "note": "sessions flip between models; a wider margin holds the lane",
        },
    )
    _tuning(candidate.config)["switch_margin"] = 0.1
    return candidate


def _agent_header_candidate(base, diagnosis) -> SessionPolicyCandidate | None:
    separation = ((diagnosis or {}).get("agentic") or {}).get("agent_separation") or {}
    if separation.get("status") != "not_separate":
        return None
    protection = _protection(base)
    identity = protection.get("identity")
    if not isinstance(identity, dict):
        identity = {}
        protection["identity"] = identity
    headers = identity.get("headers")
    if not isinstance(headers, dict):
        headers = {}
        identity["headers"] = headers
    if headers.get("agent"):
        return None  # the agent header is already configured
    candidate = _candidate(
        base,
        "identity.headers.agent",
        None,
        AGENT_HEADER,
        {
            "reason": "agent_separation",
            "evidence": separation.get("reason", ""),
            "note": "sub-agents are not kept separate; hold per-agent identity",
        },
    )
    _protection(candidate.config)["identity"]["headers"]["agent"] = AGENT_HEADER
    return candidate


def _min_turns_candidate(base, tuning, summary) -> SessionPolicyCandidate | None:
    rejected = summary.get("selection_rejected") or 0
    if rejected <= 0:
        return None
    current = tuning.get("min_turns_before_switch", 1)
    if current >= 2:
        return None  # already holding past the first turn
    candidate = _candidate(
        base,
        "min_turns_before_switch",
        current,
        current + 1,
        {
            "reason": "selection_rejected",
            "selection_rejected": rejected,
            "note": "rejected selections follow early switches; hold longer first",
        },
    )
    _tuning(candidate.config)["min_turns_before_switch"] = current + 1
    return candidate


def propose_session_policy(
    config: dict[str, Any],
    *,
    diagnosis: dict[str, Any] | None = None,
    replay_summary: dict[str, Any] | None = None,
) -> list[SessionPolicyCandidate]:
    """Diagnosis-triggered one-variable session-policy candidates.

    `replay_summary` is the AG3 route-replay evidence summary (switches,
    rejected selections); `diagnosis` is the AG4 diagnosis artifact whose
    `agentic.agent_separation` reports sub-agent retention. Evidence absent
    produces no candidate; a knob already at its raised value produces no
    candidate either (nothing left to change).
    """
    _protection(config)  # fail fast on a config without a protection section
    tuning = _tuning(config)
    candidates = []
    for candidate in (
        _stability_candidate(config, tuning, replay_summary or {}),
        _switch_margin_candidate(config, tuning, replay_summary or {}),
        _agent_header_candidate(config, diagnosis),
        _min_turns_candidate(config, tuning, replay_summary or {}),
    ):
        if candidate is not None:
            candidates.append(candidate)
    return candidates


def write_candidates(
    candidates: list[SessionPolicyCandidate], output_dir: str | Path
) -> list[Path]:
    """Write each candidate config plus its sidecar metadata; returns config paths."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for index, candidate in enumerate(candidates):
        path = out_dir / f"candidate-{index:03d}-{candidate.variable}.yaml"
        with path.open("w", encoding="utf-8") as fh:
            yaml.safe_dump(candidate.config, fh, allow_unicode=True, sort_keys=False)
        sidecar = {
            "kind": "config",
            "ref": candidate.ref,
            "change_summary": candidate.summary(),
            "provenance": {
                "proposer": "session-policy",
                "variable": candidate.variable,
                "old": candidate.old,
                "new": candidate.new,
                "trigger": candidate.trigger,
                "insight_refs": [],
            },
        }
        path.with_suffix(".candidate.json").write_text(
            json.dumps(sidecar, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        paths.append(path)
    return paths
