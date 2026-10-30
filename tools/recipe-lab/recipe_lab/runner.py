"""Replay-tier runner: evaluate candidate configs against the answer bank.

One candidate = one router lifecycle (recipe-lab design,
Sub-issue 2). The runner owns no evaluation logic of its own:

1. start the mock/real backend and the router with a per-trial port-patched
   copy of the candidate config (-gateway standalone, so the OpenAI listener
   is served without Envoy);
2. wait for the decision model to report ready via the management API;
3. send every labeled case through the router (requests only trigger
   routing; correctness always comes from the answer bank);
4. pull the replay records, join with cases, grade from the bank,
   classify and summarize;
5. stop the processes (kill the process group so the spawned model runtime
   dies with the router) and return the summary.

Config reload (dynamic-config) instead of restarts is a later optimization;
restart-per-candidate is the correct-but-slow v1.
"""

from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .adapters.router_replay import fetch_replays, records_to_cases
from .attribution import CaseInput, attribute, summarize
from .bank import apply_bank
from .contract import ContractError, OraclePolicy


class RunnerError(RuntimeError):
    """Raised when a candidate evaluation could not complete."""


class RunnerTimeout(RunnerError):
    """The whole trial (readiness + requests + replay fetch) exceeded the
    budget's per-trial deadline. Recorded as a `status: timeout` ledger row
    instead of a metric so a hung stack can never rank as a worse recipe."""


@dataclass
class RunnerEnv:
    """Process-level environment for one candidate evaluation.

    Gateway and management ports are allocated per trial and patched into a
    copy of the candidate config (see _patch_ports), so they can never collide
    with an unrelated process or a stale router on the defaults. The mock
    backend keeps an explicit port: configs that point at it must reference
    the same one.
    """

    router_bin: str
    mock_backend_script: str
    python_bin: str
    work_dir: Path
    backend_port: int = 8000
    ready_timeout_s: int = 300
    hf_home: str | None = None
    env_extra: dict[str, str] = field(default_factory=dict)
    mock_extra_args: tuple[str, ...] = field(default_factory=tuple)


def _http_json(url: str, timeout: float = 10.0) -> Any:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _free_port() -> int:
    """Reserve an ephemeral TCP port (bind-and-release race is negligible)."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _patch_ports(candidate_config: Path, work_dir: Path) -> tuple[Path, int, int]:
    """Copy the candidate config into work_dir with per-trial ports patched in.

    The router binds its gateway listener(s) and the management API from the
    config itself, so fixed defaults (8899/8080) collide with anything already
    on those ports - and worse, a stale router on the defaults answers the
    readiness probe for a router that never started. Returns (patched config
    path, primary gateway port, management port).
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load(candidate_config.read_text(encoding="utf-8"))
    router_port = 0
    for listener in config.get("listeners") or []:
        listener["port"] = _free_port()
        if not router_port:
            router_port = listener["port"]
    if not router_port:
        raise RunnerError(f"candidate config has no gateway listeners: {candidate_config}")
    management_port = _free_port()
    config.setdefault("management_api", {})["port"] = management_port
    patched = work_dir / f"{candidate_config.name}.patched.yaml"
    patched.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return patched, router_port, management_port


def _wait_model_runtime_ready(
    management_url: str,
    timeout_s: int,
    process: subprocess.Popen,
    status_file: Path | None = None,
) -> None:
    """Wait for the model runtime to report ready.

    Primary source: the router's own startup-status file (router-runtime.json
    next to the config; State.ready == true, pkg/startupstatus). Fallback: a
    loose '"ready":true' probe on the inventory endpoint (verify-models.sh
    style), which survives inventory payload-shape drift.
    """
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RunnerError(f"router exited early with code {process.returncode}")
        if status_file is not None and status_file.is_file():
            try:
                state = json.loads(status_file.read_text(encoding="utf-8"))
                if state.get("ready") is True:
                    return
            except (json.JSONDecodeError, OSError):
                pass  # mid-write or partial state; keep polling
        try:
            with urllib.request.urlopen(management_url + "/api/v1/inventory/model-runtime", timeout=10.0) as response:
                if '"ready":true' in response.read().decode("utf-8", errors="replace"):
                    return
        except (urllib.error.URLError, TimeoutError, OSError):
            pass
        time.sleep(5)
    raise RunnerError(f"model runtime not ready within {timeout_s}s")


def _send_prompt(router_url: str, model: str, prompt: str, timeout: float) -> int:
    body = json.dumps(
        {
            "model": model,
            "max_tokens": 64,
            "messages": [{"role": "user", "content": prompt}],
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        router_url + "/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return int(response.status)
    except urllib.error.HTTPError as exc:
        return int(exc.code)


def evaluate_candidate(
    *,
    candidate_config: str | Path,
    cases: list[dict[str, Any]],
    oracle_policy: OraclePolicy,
    bank: dict[tuple[str, str], Any],
    env: RunnerEnv,
    router_model: str = "vllm-sr/auto",
    family: str = "d3",
    request_timeout: float = 180.0,
    trial_timeout_s: float | None = None,
) -> dict[str, Any]:
    """Evaluate one candidate config; returns rows + summary + request log.

    trial_timeout_s (A5, budget.max_minutes_per_trial * 60) bounds the whole
    trial: readiness wait + request loop + replay fetch. Exceeding it raises
    RunnerTimeout; processes are always cleaned up in `finally`.
    """
    env.work_dir.mkdir(parents=True, exist_ok=True)
    patched_config, router_port, management_port = _patch_ports(Path(candidate_config), env.work_dir)
    management_url = f"http://127.0.0.1:{management_port}"
    router_url = f"http://127.0.0.1:{router_port}"
    deadline = time.monotonic() + trial_timeout_s if trial_timeout_s is not None else None

    def remaining_s() -> float | None:
        return None if deadline is None else deadline - time.monotonic()

    def check_deadline(stage: str) -> None:
        left = remaining_s()
        if left is not None and left <= 0:
            raise RunnerTimeout(f"trial exceeded {trial_timeout_s:.0f}s budget before {stage}")

    procs: list[subprocess.Popen] = []
    try:
        # the router writes startup state (ready flag) next to its config file;
        # clear the previous candidate's state so we only see fresh readiness
        status_file = patched_config.parent / "router-runtime.json"
        if status_file.exists():
            status_file.unlink(missing_ok=True)

        # backend first: the router probes it during startup. Each child gets
        # its own session/process group, so the group-kill in `finally` cannot
        # reach the scheduler task this runner lives in.
        mock = subprocess.Popen(
            [env.python_bin, env.mock_backend_script, "--port", str(env.backend_port), *env.mock_extra_args],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        procs.append(mock)
        time.sleep(1.0)

        router_env = dict(os.environ)
        if env.hf_home:
            router_env["HF_HOME"] = env.hf_home
        router_env.update(env.env_extra)
        router = subprocess.Popen(
            [
                env.router_bin,
                "-config",
                str(patched_config),
                "-gateway",
                "standalone",
            ],
            stdout=open(Path(env.work_dir) / "router.log", "w", encoding="utf-8"),
            stderr=subprocess.STDOUT,
            env=router_env,
            start_new_session=True,  # the spawned model runtime dies with the group
        )
        procs.append(router)
        # readiness wait is bounded by the per-trial deadline too; a plain
        # not-ready error still applies when the deadline is not the cause
        ready_budget = env.ready_timeout_s
        left = remaining_s()
        if left is not None:
            if left <= 0:
                raise RunnerTimeout(f"trial exceeded {trial_timeout_s:.0f}s budget before model-runtime readiness")
            ready_budget = min(env.ready_timeout_s, int(left) + 1)
        try:
            _wait_model_runtime_ready(management_url, ready_budget, router, status_file=status_file)
        except RunnerError as exc:
            left = remaining_s()
            if "not ready within" in str(exc) and left is not None and left <= 0:
                raise RunnerTimeout(f"trial exceeded {trial_timeout_s:.0f}s budget waiting for model-runtime readiness") from exc
            raise

        statuses: dict[str, int] = {}
        for case in cases:
            check_deadline(f"case {case['case_id']}")
            statuses[case["case_id"]] = _send_prompt(
                router_url, router_model, case["prompt"], request_timeout
            )

        check_deadline("replay fetch")
        records = fetch_replays(management_url)
        join_cases = [
            type("CaseJoin", (), {"case_id": c["case_id"], "prompt": c["prompt"], "truth": c["truth"]})
            for c in cases
        ]
        rows, join_stats = records_to_cases(records, join_cases, family=family)
        rows, bank_stats = apply_bank(rows, bank)
        classified = attribute(
            [CaseInput.from_dict(r, i) for i, r in enumerate(rows)],
            oracle_policy,
            dataset_hash="",  # stamped by the caller via the spec
        )
        summary = summarize(classified)
        return {
            "rows": classified,
            "summary": summary,
            "statuses": statuses,
            "join_stats": join_stats,
            "bank_stats": bank_stats,
        }
    finally:
        for proc in procs:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                try:
                    proc.terminate()
                except ProcessLookupError:
                    pass
        for proc in procs:
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass


def attribution_delta(baseline_summary: dict, candidate_summary: dict) -> dict[str, int]:
    """Fixed-error counts of a candidate against the baseline summary."""
    def count(summary: dict, key: str) -> int:
        return int(summary.get("class_counts", {}).get(key, 0))

    e1_b, e1_c = count(baseline_summary, "E1_signal"), count(candidate_summary, "E1_signal")
    e2_b, e2_c = count(baseline_summary, "E2_policy"), count(candidate_summary, "E2_policy")
    mm_b, mm_c = count(baseline_summary, "decision_mismatch"), count(candidate_summary, "decision_mismatch")
    return {
        "e1_fixed": max(0, e1_b - e1_c),
        "e2_fixed": max(0, e2_b - e2_c),
        "decision_mismatch_fixed": max(0, mm_b - mm_c),
        "e3_remaining": count(candidate_summary, "E3_capability"),
    }


# A1: where each contracted replay metric lives in the run summary. The
# metric name is the ledger key, the summary key is the runner's own field.
REPLAY_SUMMARY_KEYS = {"routed_accuracy": "accuracy"}


def replay_metric_value(summary: dict, replay_metric: str) -> float | None:
    """A contracted replay metric (objective.replay_metric) from a run summary."""
    summary_key = REPLAY_SUMMARY_KEYS.get(replay_metric)
    if summary_key is None:
        raise ContractError(f"unknown replay metric {replay_metric!r}; known: {sorted(REPLAY_SUMMARY_KEYS)}")
    value = summary.get(summary_key)
    return None if value is None else round(float(value), 6)


def candidate_accuracy(summary: dict, metric_key: str) -> float | None:
    """The objective metric for a candidate: routed-outcome accuracy (live tier)."""
    accuracy = summary.get("accuracy")
    return None if accuracy is None else round(float(accuracy), 6)
