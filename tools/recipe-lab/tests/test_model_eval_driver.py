"""Tests for the model_eval verify driver (C7)."""

import json
from pathlib import Path

import pytest
from recipe_lab.contract import API_VERSION, ContractError, ExperimentSpec
from recipe_lab.live import paired_stats_from_comparison
from recipe_lab.model_eval_driver import (
    arm_quality_rates,
    paired_stats_from_judged,
    run_arm,
    verify_live_model_eval,
)
from recipe_lab.verdict import compute_paired_stats

BASE_ENDPOINT = "http://127.0.0.1:9001/v1"
CAND_ENDPOINT = "http://127.0.0.1:9002/v1"
ROUTER_MODEL = "vllm-sr/auto"
STRONG = "Qwen/Qwen3.5-4B"
WEAK = "plainmodel"
N_TASKS = 30  # radius sqrt(2*ln(40)/30) < 0.5, so a 0.5 delta passes a 0pp bar


def _spec_dict(**overrides):
    spec = {
        "api_version": API_VERSION,
        "objective": {
            "metric_key": "accuracy",
            "direction": "maximize",
            "min_cases": 10,
            "min_improvement_pp": 0.0,
        },
        "workloads": [
            {
                "name": "cold",
                "profile": "quick",
                "driver": "model_eval",
                "splits": {
                    "dev": {
                        "manifest": "dev.jsonl",
                        "content_hash": "sha256:" + "0" * 64,
                    }
                },
            }
        ],
        "attribution": {"oracle_policy": {"math": STRONG}},
        "budget": {"rounds": 1, "max_minutes_per_trial": 5},
        "models": [
            {"name": STRONG, "endpoint": "http://127.0.0.1:8001/v1"},
            {
                "name": WEAK,
                "endpoint": "http://127.0.0.1:8002/v1",
                "served_name": "plain-served",
            },
        ],
        "generation": {"temperature": 0.0, "max_tokens": 512},
        "references": {"strong": STRONG, "weak": WEAK},
        "cost_basis": {
            "source": "provider_pricing",
            "assumed_pricing": {
                STRONG: {"prompt_per_1m": 0.5, "completion_per_1m": 1.0}
            },
        },
    }
    spec.update(overrides)
    return spec


def _write_tasks(path: Path, n: int = N_TASKS) -> None:
    rows = [
        {
            "task_id": f"t{i}",
            "prompt_user": f"question {i}",
            "category": "math" if i % 2 == 0 else "law",
            "answer": "A",
        }
        for i in range(n)
    ]
    path.write_text(
        "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows), encoding="utf-8"
    )


def _judged_rows(tasks: list[dict], correct_ids: set[str]) -> list[dict]:
    return [
        {
            "task_id": task["task_id"],
            "category": task["category"],
            "is_correct": task["task_id"] in correct_ids,
            "no_answer": False,
            "finish_reason": "stop",
            "completion_tokens": 10,
            "latency_s": 0.5,
        }
        for task in tasks
    ]


def _fake_run_arm(correct_by_arm: dict[str, set[str]]):
    """A run_arm stand-in: writes judged+records driven by the plan.

    `correct_by_arm` keys match the endpoint for router arms and the
    served model name for reference arms. Returns (fake, calls).
    """
    calls: list[dict] = []

    def fake(
        *,
        python_bin,
        model_eval_dir,
        tasks_path,
        out_dir,
        endpoint,
        model,
        params_path,
        concurrency,
        request_timeout,
        timeout_s,
        force=False,
    ):
        calls.append({"endpoint": endpoint, "model": model})
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        tasks = [
            json.loads(line)
            for line in Path(tasks_path).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        correct = correct_by_arm.get(endpoint, correct_by_arm.get(model, set()))
        judged = _judged_rows(tasks, correct)
        (out / "judged.jsonl").write_text(
            "".join(json.dumps(r, sort_keys=True) + "\n" for r in judged),
            encoding="utf-8",
        )
        records = [
            {
                "task_id": task["task_id"],
                "prompt_tokens": 100,
                "completion_tokens": 10,
                "latency_s": 0.5,
                "error": None,
            }
            for task in tasks
        ]
        (out / "records.jsonl").write_text(
            "".join(json.dumps(r, sort_keys=True) + "\n" for r in records),
            encoding="utf-8",
        )
        return out / "judged.jsonl"

    return fake, calls


def _golden_plan():
    tasks = [
        {"task_id": f"t{i}", "category": "math" if i % 2 == 0 else "law"}
        for i in range(N_TASKS)
    ]
    return tasks, {
        # baseline: correct on every even case; candidate: always correct
        BASE_ENDPOINT: {t["task_id"] for t in tasks if int(t["task_id"][1:]) % 2 == 0},
        CAND_ENDPOINT: {t["task_id"] for t in tasks},
        STRONG: {t["task_id"] for t in tasks},
        "plain-served": set(),
    }


def test_paired_stats_from_judged_matches_compute_paired_stats():
    base = [
        {"task_id": "a", "is_correct": True},
        {"task_id": "b", "is_correct": False},
        {"task_id": "c", "is_correct": True},
        {"task_id": "d", "is_correct": True},
    ]
    cand = [
        {"task_id": "a", "is_correct": True},
        {"task_id": "b", "is_correct": True},
        {"task_id": "c", "is_correct": False},
        {"task_id": "d", "is_correct": True},
    ]
    stats = paired_stats_from_judged(base, cand)
    expected = compute_paired_stats([0, 1, -1, 0])  # cand - base per case
    assert stats == expected
    assert (stats.wins, stats.losses, stats.ties) == (1, 1, 2)
    assert stats.paired_cases == 4


def test_paired_stats_from_judged_matches_srbench_compare_shape():
    # C7 completion criterion (unit form): on the same per-case results the
    # driver's stats carry the same fields an sr-bench compare payload would.
    base = [
        {"task_id": "a", "is_correct": True},
        {"task_id": "b", "is_correct": False},
        {"task_id": "c", "is_correct": True},
        {"task_id": "d", "is_correct": True},
    ]
    cand = [
        {"task_id": "a", "is_correct": True},
        {"task_id": "b", "is_correct": True},
        {"task_id": "c", "is_correct": True},
        {"task_id": "d", "is_correct": True},
    ]
    stats = paired_stats_from_judged(base, cand)
    comparison = {
        "comparisons": [
            {
                "paired_cases": 4,
                "quality_delta": 0.25,
                "quality_delta_ci95": [0.0, 0.5],
                "wins": 1,
                "losses": 0,
                "ties": 3,
            }
        ]
    }
    from_compare = paired_stats_from_comparison(comparison)
    assert from_compare is not None
    assert stats.paired_cases == from_compare.paired_cases
    assert stats.quality_delta == from_compare.quality_delta
    assert (stats.wins, stats.losses, stats.ties) == (
        from_compare.wins,
        from_compare.losses,
        from_compare.ties,
    )


def test_paired_stats_from_judged_partial_overlap_and_disjoint():
    base = [{"task_id": "a", "is_correct": True}, {"task_id": "b", "is_correct": True}]
    cand = [{"task_id": "b", "is_correct": False}, {"task_id": "c", "is_correct": True}]
    stats = paired_stats_from_judged(base, cand)
    assert stats is not None
    assert stats.paired_cases == 1  # only "b" is common
    assert (stats.wins, stats.losses, stats.ties) == (0, 1, 0)
    assert paired_stats_from_judged(base, [{"task_id": "z", "is_correct": True}]) is None


def test_arm_quality_rates_rates_and_missing():
    rows = [
        {"finish_reason": "stop", "no_answer": False},
        {"finish_reason": "length", "no_answer": True},
        {"finish_reason": "stop", "no_answer": False},
        {"finish_reason": "stop", "no_answer": False},
    ]
    rates = arm_quality_rates(rows, planned_cases=5)
    assert rates == {
        "truncation_rate": 0.25,
        "missing_result_rate": 0.2,
        "no_answer_rate": 0.25,
        "finish_reason_length_rate": 0.25,
    }
    # no judged rows at all: the arm is entirely missing, the per-case rates
    # stay unknown (never guessed as zero)
    assert arm_quality_rates([], planned_cases=5) == {
        "truncation_rate": None,
        "missing_result_rate": 1.0,
        "no_answer_rate": None,
        "finish_reason_length_rate": None,
    }
    assert arm_quality_rates(rows, planned_cases=0) is None


def test_run_arm_argv_outputs_and_missing_script(tmp_path, monkeypatch):
    model_eval_dir = tmp_path / "scripts"
    model_eval_dir.mkdir()
    (model_eval_dir / "multi_benchmark_eval.py").write_text("# gen\n")
    (model_eval_dir / "mcq_judge.py").write_text("# judge\n")
    tasks = tmp_path / "tasks.jsonl"
    _write_tasks(tasks, n=2)
    captured: list[list[str]] = []

    def fake_run(argv, *, timeout_s, what):
        captured.append(argv)
        # emulate the judge writing its output
        out = Path(argv[argv.index("--out") + 1])
        out.write_text("", encoding="utf-8")

    monkeypatch.setattr("recipe_lab.model_eval_driver._run", fake_run)
    judged = run_arm(
        python_bin="python",
        model_eval_dir=model_eval_dir,
        tasks_path=tasks,
        out_dir=tmp_path / "arm",
        endpoint="http://ep/v1",
        model="served",
        params_path=tmp_path / "generation.json",
        concurrency=4,
        request_timeout=30.0,
        timeout_s=60.0,
    )
    assert judged == tmp_path / "arm" / "judged.jsonl"
    gen_argv, judge_argv = captured
    assert gen_argv[:2] == ["python", str(model_eval_dir / "multi_benchmark_eval.py")]
    assert gen_argv[2] == "--split"
    assert gen_argv[3] == str(tasks)
    assert gen_argv[gen_argv.index("--endpoint") + 1] == "http://ep/v1"
    assert gen_argv[gen_argv.index("--model") + 1] == "served"
    assert "--force" not in gen_argv
    assert judge_argv[1] == str(model_eval_dir / "mcq_judge.py")
    assert judge_argv[judge_argv.index("--tasks") + 1] == str(tasks)
    # --analysis is a profile-only output; the driver skips it
    assert "--analysis" not in judge_argv

    forced = run_arm(
        python_bin="python",
        model_eval_dir=model_eval_dir,
        tasks_path=tasks,
        out_dir=tmp_path / "arm2",
        endpoint="http://ep/v1",
        model="served",
        params_path=tmp_path / "generation.json",
        concurrency=4,
        request_timeout=30.0,
        timeout_s=60.0,
        force=True,
    )
    assert forced.is_file()
    assert captured[-2][-1] == "--force"

    # a missing script is an operational error, not a verdict input
    (model_eval_dir / "mcq_judge.py").unlink()
    with pytest.raises(ContractError, match="model_eval script not found"):
        run_arm(
            python_bin="python",
            model_eval_dir=model_eval_dir,
            tasks_path=tasks,
            out_dir=tmp_path / "arm3",
            endpoint="http://ep/v1",
            model="served",
            params_path=tmp_path / "generation.json",
            concurrency=4,
            request_timeout=30.0,
            timeout_s=60.0,
        )


def test_verify_live_model_eval_end_to_end(tmp_path, monkeypatch):
    tasks_path = tmp_path / "tasks.jsonl"
    _write_tasks(tasks_path)
    _, plan = _golden_plan()
    fake, calls = _fake_run_arm(plan)
    monkeypatch.setattr("recipe_lab.model_eval_driver.run_arm", fake)
    spec = ExperimentSpec.from_dict(_spec_dict())
    run_cache = tmp_path / "run-cache.json"

    result = verify_live_model_eval(
        spec=spec,
        tasks_path=tasks_path,
        baseline_endpoint=BASE_ENDPOINT,
        candidate_endpoint=CAND_ENDPOINT,
        router_model=ROUTER_MODEL,
        repetitions=1,
        measurements=[],
        model_eval_dir=tmp_path / "scripts",
        python_bin="python",
        work_dir=tmp_path / "work",
        run_cache=run_cache,
    )

    # router arms ran in AB order; references ran once each, direct
    assert [(c["endpoint"], c["model"]) for c in calls] == [
        (BASE_ENDPOINT, ROUTER_MODEL),
        (CAND_ENDPOINT, ROUTER_MODEL),
        ("http://127.0.0.1:8001/v1", STRONG),
        ("http://127.0.0.1:8002/v1", "plain-served"),
    ]
    assert result["verdict"] == "PASS"
    assert result["metrics"] == {"baseline": 0.5, "candidate": 1.0}
    assert result["driver"] == "model_eval"
    assert result["planned_cases"] == N_TASKS
    assert result["tasks_hash"].startswith("sha256:")
    assert result["preview"] is None
    assert result["baseline_run_id"] is None
    # M2: merged single-pair stats in the sr-bench compare shape
    paired = result["paired"]
    assert paired["paired_cases"] == N_TASKS
    assert paired["quality_delta"] == pytest.approx(0.5)
    assert paired["wins"] == 15
    assert paired["losses"] == 0
    assert paired["ties"] == 15
    assert paired["repetitions"] == 1
    assert paired["paired_observations"] == N_TASKS
    # M3: every rate measured and clean
    assert result["evidence_quality"]["candidate"]["truncation_rate"] == 0.0
    assert result["evidence_quality"]["candidate"]["missing_result_rate"] == 0.0
    # M4: references anchored, strong cost assumed from token counts
    assert result["references"]["strong"]["metric"] == 1.0
    assert result["references"]["strong"]["cached"] is False
    assert result["references"]["strong"]["cost_usd"] == pytest.approx(
        N_TASKS * (100 * 0.5 + 10 * 1.0) / 1e6
    )
    assert result["references"]["strong"]["cost_note"] == "assumed"
    # no assumed pricing for the weak model: cost unknown, never zero
    assert result["references"]["weak"]["metric"] == 0.0
    assert result["references"]["weak"]["cost_usd"] is None
    assert result["references"]["weak"]["cost_note"] is None
    pareto = result["pareto"]
    assert [row["id"] for row in pareto] == [
        f"router:{BASE_ENDPOINT}",
        f"router:{CAND_ENDPOINT}",
        STRONG,
        WEAK,
    ]
    assert pareto[0]["cost_usd"] is None  # router hides the served model
    assert pareto[2]["role"] == "strong"
    bench_refs = result["bench_refs"]
    assert bench_refs[0].startswith("meval:tasks:sha256:")
    assert bench_refs[1:] == [
        "meval:baseline-rep0",
        "meval:candidate-rep0",
        "meval:reference:strong",
        "meval:reference:weak",
    ]
    cached_keys = json.loads(run_cache.read_text(encoding="utf-8"))
    assert len(cached_keys) == 2
    assert all(key.startswith(result["tasks_hash"]) for key in cached_keys)

    # a second verify reuses the reference arms (same tasks hash + alias)
    result2 = verify_live_model_eval(
        spec=spec,
        tasks_path=tasks_path,
        baseline_endpoint=BASE_ENDPOINT,
        candidate_endpoint=CAND_ENDPOINT,
        router_model=ROUTER_MODEL,
        repetitions=1,
        measurements=[],
        model_eval_dir=tmp_path / "scripts",
        python_bin="python",
        work_dir=tmp_path / "work2",
        run_cache=run_cache,
    )
    assert result2["references"]["strong"]["cached"] is True
    assert result2["references"]["weak"]["cached"] is True
    assert result2["verdict"] == "PASS"
    ref_calls = [c for c in calls if c["model"] != ROUTER_MODEL]
    assert len(ref_calls) == 2  # not re-run


def test_verify_live_model_eval_input_guards(tmp_path):
    tasks_path = tmp_path / "tasks.jsonl"
    tasks_path.write_text("", encoding="utf-8")
    spec = ExperimentSpec.from_dict(_spec_dict())
    with pytest.raises(ContractError, match="empty"):
        verify_live_model_eval(
            spec=spec,
            tasks_path=tasks_path,
            baseline_endpoint=BASE_ENDPOINT,
            candidate_endpoint=CAND_ENDPOINT,
            router_model=ROUTER_MODEL,
            repetitions=1,
            measurements=[],
            model_eval_dir=tmp_path,
            python_bin="python",
            work_dir=tmp_path / "w",
        )
    with pytest.raises(ContractError, match="repetition"):
        verify_live_model_eval(
            spec=spec,
            tasks_path=tasks_path,
            baseline_endpoint=BASE_ENDPOINT,
            candidate_endpoint=CAND_ENDPOINT,
            router_model=ROUTER_MODEL,
            repetitions=0,
            measurements=[],
            model_eval_dir=tmp_path,
            python_bin="python",
            work_dir=tmp_path / "w",
        )
    _write_tasks(tasks_path)
    no_gen = _spec_dict()
    del no_gen["generation"]
    with pytest.raises(ContractError, match="generation"):
        verify_live_model_eval(
            spec=ExperimentSpec.from_dict(no_gen),
            tasks_path=tasks_path,
            baseline_endpoint=BASE_ENDPOINT,
            candidate_endpoint=CAND_ENDPOINT,
            router_model=ROUTER_MODEL,
            repetitions=1,
            measurements=[],
            model_eval_dir=tmp_path,
            python_bin="python",
            work_dir=tmp_path / "w",
        )


def test_verify_live_model_eval_empty_arm_is_inconclusive(tmp_path, monkeypatch):
    tasks_path = tmp_path / "tasks.jsonl"
    _write_tasks(tasks_path)

    def fake(**kwargs):
        out = Path(kwargs["out_dir"])
        out.mkdir(parents=True, exist_ok=True)
        judged = out / "judged.jsonl"
        if kwargs["endpoint"] == CAND_ENDPOINT:
            judged.write_text("", encoding="utf-8")  # zero judged rows
        else:
            tasks = [
                json.loads(line)
                for line in Path(kwargs["tasks_path"])
                .read_text(encoding="utf-8")
                .splitlines()
                if line.strip()
            ]
            rows = _judged_rows(tasks, {t["task_id"] for t in tasks})
            judged.write_text(
                "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows),
                encoding="utf-8",
            )
        return judged

    monkeypatch.setattr("recipe_lab.model_eval_driver.run_arm", fake)
    spec = ExperimentSpec.from_dict(_spec_dict())
    result = verify_live_model_eval(
        spec=spec,
        tasks_path=tasks_path,
        baseline_endpoint=BASE_ENDPOINT,
        candidate_endpoint=CAND_ENDPOINT,
        router_model=ROUTER_MODEL,
        repetitions=1,
        measurements=[],
        model_eval_dir=tmp_path / "scripts",
        python_bin="python",
        work_dir=tmp_path / "work",
    )
    assert result["verdict"] == "INCONCLUSIVE"
    assert any("no judged results" in reason for reason in result["reasons"])
    assert result["paired"] is None
    assert result["metrics"]["candidate"] is None


def test_verify_live_model_eval_constraint_not_measurable(tmp_path, monkeypatch):
    tasks_path = tmp_path / "tasks.jsonl"
    _write_tasks(tasks_path)
    _, plan = _golden_plan()
    fake, _ = _fake_run_arm(plan)
    monkeypatch.setattr("recipe_lab.model_eval_driver.run_arm", fake)
    spec_dict = _spec_dict()
    spec_dict["objective"]["constraints"] = [
        {"name": "cheap", "metric": "routed_cost_index", "op": "lte", "bound": 1.0}
    ]
    spec = ExperimentSpec.from_dict(spec_dict)
    result = verify_live_model_eval(
        spec=spec,
        tasks_path=tasks_path,
        baseline_endpoint=BASE_ENDPOINT,
        candidate_endpoint=CAND_ENDPOINT,
        router_model=ROUTER_MODEL,
        repetitions=1,
        measurements=[],
        model_eval_dir=tmp_path / "scripts",
        python_bin="python",
        work_dir=tmp_path / "work",
    )
    assert result["verdict"] == "INCONCLUSIVE"
    constraint = result["constraints"][0]
    assert constraint["status"] == "not_evaluated"
    assert any("not measurable" in reason for reason in result["reasons"])
