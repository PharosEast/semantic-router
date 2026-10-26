"""Router Replay adapter tests: fetch mocking + prompt-join semantics."""

from __future__ import annotations

import json
from unittest import mock

import pytest

from recipe_lab.adapters.router_replay import (
    fetch_replays,
    load_cases,
    record_to_case,
    records_to_cases,
)
from recipe_lab.contract import ContractError


def fake_record(request_id: str, prompt: str, decision: str, selected: str) -> dict:
    return {
        "request_id": request_id,
        "decision": decision,
        "selected_model": selected,
        "prompt": prompt,
        "signals": {"decision": ["depth"]},
    }


CASES = [
    {"case_id": "c1", "prompt": "What is the capital of France?", "truth": "local_standard"},
    {"case_id": "c2", "prompt": "Analyse quicksort vs mergesort from first principles.", "truth": "cloud_frontier_reasoning"},
]


def test_record_to_case_joins_by_prompt():
    row, state = record_to_case(fake_record("r1", "What is the capital of France?", "local_standard", "local/private-qwen"), {c["prompt"]: type("C", (), c) for c in CASES})
    assert state == "matched"
    assert row["case_id"] == "c1"
    assert row["truth"] == "local_standard"
    assert row["pred"] == "local_standard"
    assert row["routed"] == "local/private-qwen"
    assert row["answer_ok"] is None  # correctness joins later


def test_record_to_case_skip_reasons():
    by_prompt = {c["prompt"]: type("C", (), c) for c in CASES}
    row, state = record_to_case(fake_record("r1", "", "d", "m"), by_prompt)
    assert row is None and state == "no_prompt"
    row, state = record_to_case(fake_record("r2", "unlabeled prompt", "d", "m"), by_prompt)
    assert row is None and state == "unlabeled"


def test_records_to_cases_dedup_and_stats():
    records = [
        fake_record("r1", "What is the capital of France?", "local_standard", "local/private-qwen"),
        fake_record("r2", "What is the capital of France?", "local_standard", "local/private-qwen"),
        fake_record("r3", "Analyse quicksort vs mergesort from first principles.", "local_standard", "local/private-qwen"),
        {"request_id": "r4"},
    ]
    rows, stats = records_to_cases(records, [type("C", (), c) for c in CASES])
    assert len(rows) == 2
    assert rows[0]["case_id"] == "c1"
    assert rows[1]["case_id"] == "c2"
    assert stats == {"matched": 2, "no_prompt": 1, "unlabeled": 0, "duplicate_prompt": 1}


def test_load_cases_rejects_missing_truth(tmp_path):
    path = tmp_path / "cases.jsonl"
    path.write_text(json.dumps({"case_id": "c1", "prompt": "p"}), encoding="utf-8")
    with pytest.raises(ContractError, match="truth"):
        load_cases(path)


def test_fetch_replays_parses_list():
    payload = {"data": [fake_record("r1", "p", "d", "m")], "total": 1}
    response = mock.MagicMock()
    response.read.return_value = json.dumps(payload).encode()
    response.__enter__ = lambda s: s
    response.__exit__ = lambda s, *a: None
    with mock.patch("urllib.request.urlopen", return_value=response):
        records = fetch_replays("http://127.0.0.1:8080")
    assert len(records) == 1 and records[0]["request_id"] == "r1"
