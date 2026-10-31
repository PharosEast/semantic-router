"""Fixtures for the real-component (V2) integration tests.

These tests drive the real semantic-router binary and, for the sr-bench
scenarios, a live sr-bench service. They skip unless the corresponding
environment is present:

- ``RECIPE_LAB_ROUTER_BIN``: path to the semantic-router binary (R1, R2)
- ``RECIPE_LAB_SRBENCH_URL``: base URL of a running sr-bench service (R2)
- ``RECIPE_LAB_HF_HOME``: optional HF_HOME forwarded to router processes
"""

from __future__ import annotations

import os
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
