"""R2: verify drives the real sr-bench service end to end (V2).

Criterion (V2-2): ``recipe-lab verify`` completes 3 AB/BA repetitions against
the live sr-bench service and auto-constructs the strong reference single-model
baseline arm from a MoM-only baseline manifest. Criterion (V2-3): a fault
injection (cancelling an in-flight pair arm) yields INCONCLUSIVE, never a
guessed verdict. Criterion (V2-4): a dataset sha256 mismatch is rejected
before any run is spent on it.

X7: the suite runs in one of two modes. Without ``RECIPE_LAB_VLLM_URL``
everything is mocked (6-case answer-bank set). With it, the single-model
reference target points at the real strong lane and the workload becomes
the store's prepared MMLU-Pro smoke dataset, so the same assertions run
against real models (the "re-run V2 per its original definition" step).
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import threading
import time
import urllib.request
from pathlib import Path

import pytest
import yaml
from recipe_lab.cli import main
from recipe_lab.splitting import load_cases
from recipe_lab.srbench import cases_to_srbench_rows

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not os.environ.get("RECIPE_LAB_ROUTER_BIN"),
        reason="RECIPE_LAB_ROUTER_BIN is not set; real router required (V2)",
    ),
    pytest.mark.skipif(
        not (
            os.environ.get("RECIPE_LAB_SRBENCH_URL")
            and os.environ.get("RECIPE_LAB_SRBENCH_STORE")
        ),
        reason="RECIPE_LAB_SRBENCH_URL/RECIPE_LAB_SRBENCH_STORE are not set (V2)",
    ),
]

_BASELINE_MANIFEST_NAME = "recipe-lab-r2-baseline"
_BUCKETS = {"input": 0, "cached_input": 0, "cache_write": 0, "output": 0}


# ---- asset builders ----


def _manifest_cases(smoke_cases: Path) -> list[dict]:
    """The smoke cases rendered as sr-bench manifest rows (X3).

    Rendered by the shared ``cases to-srbench`` transform rather than a
    bespoke copy, so the contract can declare this rendering next to the
    cases file and pair the two 1:1. The answer is pinned to "A" (the
    answer the mock backend is graded against); the user message is the
    smoke case's prompt verbatim: the capability-aware mock backend keys
    its behavior on the exact prompt text, so the two routers' depth
    gates produce real routing movement.
    """
    rows = cases_to_srbench_rows(load_cases(smoke_cases), benchmark="mmlu-pro")
    return [dict(row, answer="A") for row in rows]


def _write_dataset(path: Path, cases: list[dict]) -> Path:
    path.write_text(
        "".join(json.dumps(case, ensure_ascii=False) + "\n" for case in cases),
        encoding="utf-8",
    )
    return path


def _long_dataset(path: Path, smoke_cases: Path, repeat: int) -> Path:
    """The smoke cases duplicated with unique ids: wider in-flight windows.

    A smoke arm finishes in well under a second (six ~25 ms cases), so a
    fault-injection poll can miss every pair arm; repeating the cases keeps
    each arm in flight for seconds and makes the injection observable.
    """
    cases = _manifest_cases(smoke_cases)
    rows = [
        dict(case, id=f"{case['id']}-r{k}") for k in range(repeat) for case in cases
    ]
    return _write_dataset(path, rows)


def _render_spec(
    smoke_dir: Path, workload: dict, dataset: Path | None, work: Path
) -> Path:
    """The smoke spec bound to the active workload mode + R2 knobs.

    Mock mode keeps the dual-format contract (X3): the recipe-lab cases
    file (``manifest`` + ``content_hash``) and its sr-bench rendering
    (``srbench_dataset`` + ``srbench_sha256``) declared 1:1; min_cases=3
    matches the 6-case dataset (the shared template keeps its default of
    200 for the e2e profile and must not change). Real-lane mode (X7)
    binds legacy-style: the store's prepared MMLU-Pro smoke file is both
    the cases file and the sr-bench dataset, so ``live_dataset_hash``
    falls back to the cases content hash and no rendering is declared;
    min_cases matches the dataset's case count.

    ``dataset=None`` means "the workload's own dataset" (the mock tests
    pass their per-test rendering file explicitly).
    """
    cases = workload["cases"]
    if dataset is None:
        dataset = workload["dataset"]
        if dataset is None:  # pragma: no cover - guarded by the fixtures
            raise RuntimeError("no dataset bound for the active workload mode")
    cases_hash = "sha256:" + hashlib.sha256(cases.read_bytes()).hexdigest()
    text = (
        (smoke_dir / "smoke-spec.yaml.tmpl")
        .read_text(encoding="utf-8")
        .replace("${CASES_FILE}", str(cases))
        .replace("${CASES_HASH}", cases_hash)
    )
    spec = yaml.safe_load(text)
    spec["objective"]["min_cases"] = workload["min_cases"]
    spec["references"] = {"strong": "rl-single", "weak": "rl-single"}
    if workload["declare_rendering"]:
        spec["workload"]["srbench_dataset"] = str(dataset)
        spec["workload"]["srbench_sha256"] = (
            "sha256:" + hashlib.sha256(dataset.read_bytes()).hexdigest()
        )
    spec_path = work / "spec.yaml"
    spec_path.write_text(yaml.safe_dump(spec, sort_keys=False), encoding="utf-8")
    return spec_path


def _mom_target(target_id: str, router: dict) -> dict:
    return {
        "id": target_id,
        "kind": "mom",
        "model": "vllm-sr/auto",
        "base_url": router["listener_url"] + "/v1",
        "config_hash": router["config_hash"],
        "max_inference_calls": 1,
        "prices": {"vllm-sr/auto": _BUCKETS},
        "preview_url": router["mgmt_url"] + "/api/v1/routing/preview",
    }


def _write_registry(path: Path, router_a: dict, router_b: dict, single: dict) -> Path:
    """Register the mode-selected single target (the auto-built S arm)
    and both routers."""
    targets = [
        single,
        _mom_target("rl-mom-baseline", router_a),
        _mom_target("rl-mom-candidate", router_b),
    ]
    path.write_text(json.dumps({"targets": targets}, indent=1) + "\n", encoding="utf-8")
    return path


def _register_targets(srbench_bin: str, srbench_store: str, registry: Path) -> None:
    completed = subprocess.run(
        [
            srbench_bin,
            "benchmark",
            "--store",
            srbench_store,
            "target",
            "register",
            "--file",
            str(registry),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr or completed.stdout


def _mom_only_manifest(
    path: Path, dataset: Path, router_a: dict, limits: dict
) -> Path:
    """A plain MoM-router baseline manifest: no single target (the S arm
    must be auto-constructed from the registry by verify). Limits come
    from the workload mode: the real-lane runs need wider timeouts and
    some concurrency (X7)."""
    manifest = {
        "version": "sr-bench-1.0",
        "name": _BASELINE_MANIFEST_NAME,
        "cost_policy": "capability_only",
        "targets": [_mom_target("rl-mom-baseline", router_a)],
        "dataset": {
            "path": str(dataset),
            "sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
        },
        "limits": dict(limits),
    }
    path.write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    return path


def _candidate_json(path: Path) -> Path:
    path.write_text(
        json.dumps(
            {
                "kind": "config",
                "ref": "rl/15-integration",
                "change_summary": "R2 integration candidate: depth gate lowered",
                "srbench_targets": ["rl-mom-candidate"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def _verify_argv(
    spec_path: Path,
    manifest_path: Path,
    candidate_path: Path,
    work: Path,
    srbench_url: str,
) -> list[str]:
    return [
        "verify",
        "--spec",
        str(spec_path),
        "--baseline-manifest",
        str(manifest_path),
        "--candidate",
        str(candidate_path),
        "--candidate-targets",
        "rl-mom-candidate",
        "--repetitions",
        "3",
        "--work-dir",
        str(work),
        "--srbench-url",
        srbench_url,
    ]


def _run_verify(capsys: pytest.CaptureFixture[str], argv: list[str]) -> tuple[int, dict]:
    """Run verify in-process; any of the three verdict exit codes is valid.

    The result document is pretty-printed (``indent=2``), so stdout must be
    parsed as a whole document, not line by line.
    """
    rc = main(argv)
    assert rc in (0, 1, 2), f"verify exited {rc} (a crash is not a verdict)"
    out = capsys.readouterr().out
    try:
        document = json.loads(out)
    except json.JSONDecodeError as exc:
        raise AssertionError("verify printed no verdict document") from exc
    assert isinstance(document, dict) and "verdict" in document, document
    return rc, document


@pytest.fixture(scope="session")
def registered_targets(
    srbench_bin: str,
    srbench_store: str,
    router_a: dict,
    router_b: dict,
    mock_backend: dict,
    vllm_lanes: dict | None,
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    """Register the R2 targets once: the S arm and both routers.

    The S arm is the strong reference: the mock backend in mock mode,
    the strong lane serving the frontier model in real-lane mode (X7).
    """
    if vllm_lanes is None:
        single = {
            "id": "rl-single",
            "kind": "single",
            "model": "local/private-qwen",
            "base_url": mock_backend["endpoint"] + "/v1",
            "prices": {"local/private-qwen": _BUCKETS},
        }
    else:
        single = {
            "id": "rl-single",
            "kind": "single",
            "model": "cloud/frontier-reasoning",
            "base_url": vllm_lanes["strong"] + "/v1",
            "prices": {"cloud/frontier-reasoning": _BUCKETS},
        }
    registry = tmp_path_factory.mktemp("r2-registry") / "targets.json"
    _register_targets(srbench_bin, srbench_store, _write_registry(registry, router_a, router_b, single))


# ---- V2-2: verify completes 3 AB/BA pairs and builds the S arm ----


def _bound_dataset(workload: dict, dataset: Path | None) -> Path:
    """The dataset file the manifest binds: the test's own rendering in
    mock mode, the workload's prepared MMLU-Pro file in real-lane mode."""
    bound = dataset if dataset is not None else workload["dataset"]
    if bound is None:  # pragma: no cover - guarded by the fixtures
        raise RuntimeError("no dataset bound for the active workload mode")
    return bound


def test_verify_srbench_end_to_end(
    smoke_dir: Path,
    workload: dict,
    srbench_url: str,
    router_a: dict,
    registered_targets: None,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    dataset = (
        _write_dataset(tmp_path / "dataset.jsonl", _manifest_cases(workload["cases"]))
        if workload["mode"] == "mock"
        else None
    )
    spec_path = _render_spec(smoke_dir, workload, dataset, tmp_path)
    manifest_path = _mom_only_manifest(
        tmp_path / "baseline-manifest-src.json",
        _bound_dataset(workload, dataset),
        router_a,
        workload["limits"],
    )
    candidate_path = _candidate_json(tmp_path / "candidate.json")
    work = tmp_path / "verify-work"
    work.mkdir()

    rc, result = _run_verify(
        capsys,
        _verify_argv(spec_path, manifest_path, candidate_path, work, srbench_url),
    )

    # the frozen baseline manifest carries the injected registry S arm first
    frozen = json.loads(
        (work / "baseline-manifest.submitted.json").read_text("utf-8")
    )
    assert frozen["targets"][0]["id"] == "rl-single"
    assert frozen["targets"][1]["id"] == "rl-mom-baseline"

    # three AB/BA repetitions against the real service
    assert len(result["trials"]) == 3
    assert result["baseline_run_id"], "verify must resolve the submitted baseline"
    assert result["baseline_run_id"] in result["bench_refs"]
    # the reference single-model arm was auto-constructed and completed
    assert result["references"]["strong"]["run_id"], result["references"]
    assert result["verdict"] in {"PASS", "FAIL", "INCONCLUSIVE"}
    assert rc == {"PASS": 0, "FAIL": 1, "INCONCLUSIVE": 2}[result["verdict"]]


# ---- V2-3: fault injection during a pair arm is INCONCLUSIVE ----


def test_verify_fault_injection_is_inconclusive(
    smoke_dir: Path,
    workload: dict,
    srbench_url: str,
    router_a: dict,
    registered_targets: None,
    tmp_path: Path,
) -> None:
    if workload["mode"] == "mock":
        dataset = _long_dataset(tmp_path / "dataset.jsonl", workload["cases"], 16)
    else:
        # the real smoke set at real model latency keeps every arm in
        # flight for seconds - no case duplication needed (X7)
        dataset = None
    poll_deadline = 1800.0 if workload["mode"] == "vllm" else 600.0
    spec_path = _render_spec(smoke_dir, workload, dataset, tmp_path)
    manifest_path = _mom_only_manifest(
        tmp_path / "baseline-manifest-src.json",
        _bound_dataset(workload, dataset),
        router_a,
        workload["limits"],
    )
    candidate_path = _candidate_json(tmp_path / "candidate.json")
    work = tmp_path / "verify-work"
    work.mkdir()
    argv = _verify_argv(spec_path, manifest_path, candidate_path, work, srbench_url)

    outcome: dict = {}

    def driver() -> None:
        try:
            outcome["rc"] = main(argv)
        except SystemExit as exc:  # contract errors surface as SystemExit
            outcome["rc"] = exc.code if isinstance(exc.code, int) else 2
        except Exception as exc:  # surfaced via the assert below
            outcome["error"] = exc

    thread = threading.Thread(target=driver, daemon=True)
    thread.start()

    # Poll the service until a pair arm re-runs the baseline manifest (the
    # same frozen manifest submitted again after the preview) and cancel it.
    # Direct HTTP for both the observation and the cancel: the CLI's own
    # startup takes seconds, longer than the arms stay in flight. All
    # routes live under the sr-bench service's API prefix (mirrors
    # vllm-sr's service PREFIX; a bare /run path 404s into an empty list).
    api = srbench_url.rstrip("/") + "/api/sr-bench/v1"
    runs_url = api + "/runs"
    baseline_id = None
    cancelled = False
    deadline = time.monotonic() + poll_deadline
    while time.monotonic() < deadline and thread.is_alive() and not cancelled:
        try:
            with urllib.request.urlopen(runs_url, timeout=5.0) as resp:
                rows = json.loads(resp.read().decode("utf-8")).get("runs") or []
        except (OSError, json.JSONDecodeError):
            # transient service hiccups; keep polling
            time.sleep(1.0)
            continue
        for doc in rows:
            if doc.get("status") not in ("queued", "running"):
                continue
            manifest = doc.get("manifest") or {}
            if manifest.get("name") != _BASELINE_MANIFEST_NAME:
                continue
            run_id = str(doc.get("id"))
            if baseline_id is None:
                baseline_id = run_id
            elif run_id != baseline_id:
                # a re-run of the same frozen manifest is a pair's base arm;
                # an arm that finished in between refuses the cancel, so
                # fall through to the next observation
                request = urllib.request.Request(
                    f"{runs_url}/{run_id}/cancel", method="POST", data=b""
                )
                try:
                    urllib.request.urlopen(request, timeout=5.0)
                except OSError:
                    continue
                cancelled = True
                break
        if not cancelled:
            time.sleep(1.0)
    assert cancelled, "no in-flight pair arm was observed to cancel"
    thread.join(timeout=poll_deadline)
    assert not thread.is_alive(), "verify did not finish after the injection"
    assert "error" not in outcome, outcome.get("error")
    assert outcome.get("rc") == 2, "an incomplete pair must be INCONCLUSIVE (exit 2)"


# ---- V2-4: dataset mismatch is rejected before any run is spent ----


def test_verify_dataset_mismatch_rejected_before_submission(
    smoke_dir: Path,
    workload: dict,
    srbench_url: str,
    router_a: dict,
    registered_targets: None,
    tmp_path: Path,
) -> None:
    dataset = (
        _write_dataset(tmp_path / "dataset.jsonl", _manifest_cases(workload["cases"]))
        if workload["mode"] == "mock"
        else None
    )
    spec_path = _render_spec(smoke_dir, workload, dataset, tmp_path)
    manifest_path = _mom_only_manifest(
        tmp_path / "baseline-manifest-src.json",
        _bound_dataset(workload, dataset),
        router_a,
        workload["limits"],
    )
    # bind the manifest to a dataset identity the contract does not declare
    raw = json.loads(manifest_path.read_text("utf-8"))
    raw["dataset"]["sha256"] = "e" * 64
    manifest_path.write_text(json.dumps(raw, indent=1) + "\n", encoding="utf-8")
    candidate_path = _candidate_json(tmp_path / "candidate.json")
    work = tmp_path / "verify-work"
    work.mkdir()

    with pytest.raises(SystemExit, match="binds dataset"):
        main(
            _verify_argv(spec_path, manifest_path, candidate_path, work, srbench_url)
        )
    # zero-cost rejection: the submission snapshot was never written, so no
    # sr-bench run was spent on a dataset the contract cannot verify
    assert not (work / "baseline-manifest.submitted.json").exists()
