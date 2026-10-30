"""sr-bench adapter tests: subprocess mocking + metric extraction."""

from __future__ import annotations

import json
from unittest import mock

import pytest
from recipe_lab.contract import ContractError
from recipe_lab.srbench import SrbenchCli, SrbenchError, extract_metric, require_metric


def fake_run(returncode: int = 0, stdout: str = "{}", stderr: str = ""):
    completed = mock.MagicMock()
    completed.returncode = returncode
    completed.stdout = stdout
    completed.stderr = stderr
    return mock.patch("subprocess.run", return_value=completed)


def test_compare_parses_json_stdout():
    payload = {
        "baseline": {"capability.score": 0.80},
        "candidate": {"capability.score": 0.86},
    }
    with fake_run(stdout=json.dumps(payload)):
        result = SrbenchCli().compare("run-a", "run-b")
    assert result == payload


def test_compare_nonzero_exit_raises():
    with (
        fake_run(returncode=3, stderr="boom"),
        pytest.raises(SrbenchError, match="returned 3"),
    ):
        SrbenchCli().compare("run-a", "run-b")


def test_compare_non_json_is_kept_raw():
    with fake_run(stdout="not json"):
        result = SrbenchCli().compare("run-a", "run-b")
    assert isinstance(result, str) and "not json" in result


def test_extract_metric_recursive():
    payload = {"a": {"b": [{"capability.score": 0.9}]}}
    assert extract_metric(payload, "capability.score") == 0.9
    assert extract_metric(payload, "nope") is None


def test_require_metric_raises_contract_error():
    with pytest.raises(ContractError):
        require_metric({"x": 1}, "capability.score")


def test_report_reads_written_file(tmp_path):
    out = tmp_path / "report.json"
    out.write_text(json.dumps({"quality": {"accuracy": 0.83}}), encoding="utf-8")
    with fake_run():
        result = SrbenchCli().report("run-a", out)
    assert result["quality"]["accuracy"] == 0.83


def test_preview_candidate_plan_experiment_arg_shapes():
    """The thin wrappers pass the right sr-bench subcommand and arguments."""
    with fake_run(stdout="{}") as run_mock:
        SrbenchCli().preview("run-a")
        cmd = run_mock.call_args[0][0]
        assert cmd[2:] == ["preview", "run-a"]
    with fake_run(stdout="{}") as run_mock:
        SrbenchCli().candidate_plan("run-base")
        assert run_mock.call_args[0][0][2:] == ["candidate-plan", "run-base"]
    with fake_run(stdout="{}") as run_mock:
        SrbenchCli().experiment(["attach", "exp-1", "--run", "run-a"])
        assert run_mock.call_args[0][0][2:] == [
            "experiment",
            "attach",
            "exp-1",
            "--run",
            "run-a",
        ]
