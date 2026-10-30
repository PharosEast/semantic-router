"""Answer bank tests: build/grade/apply and the evidence-first join."""

from __future__ import annotations

import json

import pytest

from recipe_lab.bank import apply_bank, grade, load_bank, write_bank
from recipe_lab.contract import ContractError
from recipe_lab.bank import BankEntry


def test_grade_substring_case_insensitive():
    assert grade("The capital of France is Paris.", "paris") is True
    assert grade("Bonjour!", "paris") is False
    assert grade("anything", None) is None


def test_write_load_bank_roundtrip(tmp_path):
    entries = [
        BankEntry(case_id="c1", model="m-a", answer="Paris.", answer_ok=True),
        BankEntry(case_id="c1", model="m-b", answer="London.", answer_ok=False),
    ]
    path = write_bank(entries, tmp_path / "bank.jsonl")
    bank = load_bank(path)
    assert bank[("c1", "m-a")].answer_ok is True
    assert bank[("c1", "m-b")].answer_ok is False


def test_load_bank_rejects_duplicate_key(tmp_path):
    path = tmp_path / "bank.jsonl"
    entry = {"case_id": "c1", "model": "m-a", "answer": "x", "answer_ok": True}
    path.write_text(
        "\n".join([json.dumps(entry), json.dumps(entry)]),
        encoding="utf-8",
    )
    with pytest.raises(ContractError, match="duplicate"):
        load_bank(path)


def test_apply_bank_joins_by_routed_model():
    bank = {
        ("c1", "m-a"): BankEntry(case_id="c1", model="m-a", answer="ok", answer_ok=True),
        ("c1", "m-b"): BankEntry(case_id="c1", model="m-b", answer="bad", answer_ok=False),
    }
    rows = [
        {"case_id": "c1", "routed": "m-a"},
        {"case_id": "c1", "routed": "m-b"},
        {"case_id": "c2", "routed": "m-a"},  # no bank entry
    ]
    merged, stats = apply_bank(rows, bank)
    assert merged[0]["answer_ok"] is True
    assert merged[1]["answer_ok"] is False
    assert merged[2]["answer_ok"] is None  # evidence-first: missing entry stays no-evidence
    assert stats == {"joined": 2, "no_bank_entry": 1}
