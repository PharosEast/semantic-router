"""Fixtures for the real-component (V2) integration tests.

These tests drive the real semantic-router binary and, for the sr-bench
scenarios, a live sr-bench service. They skip unless the corresponding
environment is present:

- ``RECIPE_LAB_ROUTER_BIN``: path to the semantic-router binary (R1, R2)
- ``RECIPE_LAB_SRBENCH_URL``: base URL of a running sr-bench service (R2)
- ``RECIPE_LAB_SRBENCH_STORE``: sr-bench store directory (R2, target registry)
- ``RECIPE_LAB_SRBENCH_BIN``: vllm-sr CLI for target registration (R2)
- ``RECIPE_LAB_HF_HOME``: optional HF_HOME forwarded to router processes
"""

from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

RECIPE_LAB_ROOT = Path(__file__).resolve().parents[2]
SMOKE_DIR = RECIPE_LAB_ROOT / "examples" / "smoke"


def free_port() -> int:
    """Reserve an ephemeral TCP port (bind-and-release race is negligible)."""
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _probe(url: str) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=2.0):
            return True
    except (urllib.error.URLError, OSError):
        return False


def wait_http_ready(url: str, timeout_s: float = 30.0) -> None:
    """Poll ``url`` until it answers, or raise."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if _probe(url):
            return
        time.sleep(0.2)
    raise RuntimeError(f"{url} not ready within {timeout_s}s")


@pytest.fixture(scope="session")
def router_bin() -> str:
    """Path to the real semantic-router binary."""
    value = os.environ.get("RECIPE_LAB_ROUTER_BIN", "")
    if not value:
        pytest.skip("RECIPE_LAB_ROUTER_BIN is not set; real router required")
    return value


@pytest.fixture(scope="session")
def srbench_url() -> str:
    """Base URL of the live sr-bench service."""
    value = os.environ.get("RECIPE_LAB_SRBENCH_URL", "")
    if not value:
        pytest.skip("RECIPE_LAB_SRBENCH_URL is not set; live sr-bench required")
    return value


@pytest.fixture(scope="session")
def hf_home() -> str | None:
    """Optional HF_HOME forwarded to router processes (model artifacts)."""
    return os.environ.get("RECIPE_LAB_HF_HOME") or None


@pytest.fixture(scope="session")
def smoke_dir() -> Path:
    """The checked-in smoke assets this suite renders its stack from."""
    return SMOKE_DIR


@pytest.fixture(scope="session")
def smoke_cases(smoke_dir: Path) -> Path:
    return smoke_dir / "cases.jsonl"


@pytest.fixture(scope="session")
def mock_backend(smoke_cases: Path) -> dict:
    """One shared mock OpenAI backend on an ephemeral port.

    Serves the answer-bank build (R1) and the sr-bench single-model
    reference target (R2). Trial backends inside the runner use their own
    instances on separate ports.
    """
    port = free_port()
    proc = subprocess.Popen(
        [
            sys.executable,
            str(SMOKE_DIR / "mock_backend.py"),
            "--port",
            str(port),
            "--cases",
            str(smoke_cases),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        wait_http_ready(f"http://127.0.0.1:{port}/v1/models")
        yield {"port": port, "endpoint": f"http://127.0.0.1:{port}"}
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


@pytest.fixture()
def backend_port() -> int:
    """Ephemeral port for one test's runner-managed mock backend."""
    return free_port()


# ---- R2: sr-bench registry and real standalone routers ----


def _http_get(url: str) -> str | None:
    try:
        with urllib.request.urlopen(url, timeout=2.0) as resp:
            return resp.read().decode("utf-8", errors="replace")
    except (urllib.error.URLError, OSError):
        return None


def _wait_router_ready(mgmt_url: str, proc: subprocess.Popen, log_path: Path) -> None:
    """Poll the management API until the router's model runtime is ready."""
    deadline = time.monotonic() + 600.0
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"router exited early (see {log_path})")
        body = _http_get(f"{mgmt_url}/api/v1/inventory/model-runtime")
        if body and '"ready":true' in body:
            return
        time.sleep(1.0)
    raise RuntimeError(f"router not ready within 600s (see {log_path})")


def _router_config_hash(mgmt_url: str) -> str:
    """The router's frozen config document hash (64 hex) for sr-bench MoM."""
    body = _http_get(f"{mgmt_url}/api/v1/config/hash")
    if not body:
        raise RuntimeError("router /api/v1/config/hash unavailable")
    document_hash = str(json.loads(body).get("document_hash", ""))
    if len(document_hash) != 64:
        raise RuntimeError(f"router config hash is not 64 hex: {document_hash!r}")
    return document_hash


def _start_router(
    router_bin: str,
    smoke_dir: Path,
    mock_backend: dict,
    hf_home: str | None,
    work_dir: Path,
    depth_threshold: str | None = None,
) -> dict:
    """Start one standalone router on the rendered smoke config."""
    listener_port = free_port()
    config_text = (
        (smoke_dir / "smoke-config.yaml")
        .read_text(encoding="utf-8")
        .replace("127.0.0.1:8000", f"127.0.0.1:{mock_backend['port']}")
        .replace("port: 8899", f"port: {listener_port}")
    )
    if depth_threshold is not None:
        config_text = config_text.replace("gte: 0.6", f"gte: {depth_threshold}")
    config_path = work_dir / "config.yaml"
    config_path.write_text(config_text, encoding="utf-8")

    api_port = free_port()
    env = dict(os.environ)
    if hf_home:
        env["HF_HOME"] = hf_home
    log_path = work_dir / "router.log"
    proc = subprocess.Popen(
        [
            router_bin,
            "-config",
            str(config_path),
            "-gateway",
            "standalone",
            "-api-port",
            str(api_port),
        ],
        stdout=log_path.open("ab"),
        stderr=subprocess.STDOUT,
        env=env,
    )
    mgmt_url = f"http://127.0.0.1:{api_port}"
    try:
        _wait_router_ready(mgmt_url, proc, log_path)
        config_hash = _router_config_hash(mgmt_url)
    except BaseException:
        proc.terminate()
        raise
    return {
        "proc": proc,
        "listener_url": f"http://127.0.0.1:{listener_port}",
        "mgmt_url": mgmt_url,
        "config_hash": config_hash,
        "log_path": log_path,
    }


def _stop_router(stack: dict) -> None:
    proc = stack["proc"]
    proc.terminate()
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        proc.kill()


@pytest.fixture(scope="session")
def srbench_store() -> str:
    """sr-bench store directory backing the live service (target registry)."""
    value = os.environ.get("RECIPE_LAB_SRBENCH_STORE", "")
    if not value:
        pytest.skip("RECIPE_LAB_SRBENCH_STORE is not set; sr-bench store required")
    return value


@pytest.fixture(scope="session")
def srbench_bin() -> str:
    """vllm-sr CLI used to register targets and cancel runs."""
    value = os.environ.get("RECIPE_LAB_SRBENCH_BIN") or shutil.which("vllm-sr")
    if not value:
        pytest.skip("vllm-sr CLI not found; needed for sr-bench target registry")
    return value


@pytest.fixture(scope="session")
def router_a(
    router_bin: str, smoke_dir: Path, mock_backend: dict, hf_home: str | None,
    tmp_path_factory: pytest.TempPathFactory,
) -> dict:
    """Baseline router: the smoke config's own behavior (depth >= 0.6)."""
    stack = _start_router(
        router_bin, smoke_dir, mock_backend, hf_home, tmp_path_factory.mktemp("router-a")
    )
    yield stack
    _stop_router(stack)


@pytest.fixture(scope="session")
def router_b(
    router_bin: str, smoke_dir: Path, mock_backend: dict, hf_home: str | None,
    tmp_path_factory: pytest.TempPathFactory,
) -> dict:
    """Candidate router: the depth gate lowered to 0.15 routes far more
    traffic to the frontier lane, so the AB comparison has real movement."""
    stack = _start_router(
        router_bin, smoke_dir, mock_backend, hf_home,
        tmp_path_factory.mktemp("router-b"), depth_threshold="0.15",
    )
    yield stack
    _stop_router(stack)
