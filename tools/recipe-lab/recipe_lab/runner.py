"""Replay-tier runner: evaluate candidate configs against the answer bank.

One candidate = one router lifecycle (recipe-lab design,
Sub-issue 2). The runner owns no evaluation logic of its own:

1. start the mock/real backend and the router with the candidate config
   (-gateway standalone, so the OpenAI listener is served without Envoy);
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
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .adapters.router_replay import fetch_replays, records_to_cases
from .attribution import CaseInput, attribute, summarize
from .bank import apply_bank
from .contract import OraclePolicy


class RunnerError(RuntimeError):
    """Raised when a candidate evaluation could not complete."""


@dataclass
class RunnerEnv:
    """Process-level environment for one candidate evaluation."""

    router_bin: str
    mock_backend_script: str
    python_bin: str
    work_dir: Path
    router_port: int = 8899
    management_port: int = 8080
    backend_port: int = 8000
    ready_timeout_s: int = 300
    hf_home: str | None = None
    env_extra: dict[str, str] = field(default_factory=dict)
    mock_extra_args: tuple[str, ...] = field(default_factory=tuple)


def _http_json(url: str, timeout: float = 10.0) -> Any:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


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
) -> dict[str, Any]:
    """Evaluate one candidate config; returns rows + summary + request log."""
    env.work_dir.mkdir(parents=True, exist_ok=True)
    management_url = f"http://127.0.0.1:{env.management_port}"
    router_url = f"http://127.0.0.1:{env.router_port}"
    backend_url = f"http://127.0.0.1:{env.backend_port}"

    procs: list[subprocess.Popen] = []
    try:
        # the router writes startup state (ready flag) next to its config file;
        # clear the previous candidate's state so we only see fresh readiness
        status_file = Path(candidate_config).parent / "router-runtime.json"
        if status_file.exists():
            status_file.unlink(missing_ok=True)

        # backend first: the router probes it during startup. Each child gets
        # its own session/process group, so the group-kill in `finally` cannot
        # reach the srun task this runner lives in.
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
                str(candidate_config),
                "-gateway",
                "standalone",
            ],
            stdout=open(Path(env.work_dir) / "router.log", "w", encoding="utf-8"),
            stderr=subprocess.STDOUT,
            env=router_env,
            start_new_session=True,  # the spawned model runtime dies with the group
        )
        procs.append(router)
        _wait_model_runtime_ready(management_url, env.ready_timeout_s, router, status_file=status_file)

        statuses: dict[str, int] = {}
        for case in cases:
            statuses[case["case_id"]] = _send_prompt(
                router_url, router_model, case["prompt"], request_timeout
            )

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
    return {
        "e1_fixed": max(0, e1_b - e1_c),
        "e2_fixed": max(0, e2_b - e2_c),
        "e3_remaining": count(candidate_summary, "E3_capability"),
    }


def candidate_accuracy(summary: dict, metric_key: str) -> float | None:
    """The objective metric for a candidate: routed-outcome accuracy."""
    accuracy = summary.get("accuracy")
    return None if accuracy is None else round(float(accuracy), 6)
