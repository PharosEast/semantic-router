"""Adapter for the router's Router Replay observability API.

The router records one RoutingRecord per request for every decision that
mounts the `router_replay` plugin, into a store (memory/redis/postgres/...)
and exposes them through the management API:

    GET /api/v1/observability/replays?showDetails=true

This adapter pulls those records and joins them against a hand-labeled case
list (prompt -> truth label) to produce attribution input records:

    {"case_id": ..., "truth": ..., "matched_decision": <record.decision>,
     "routed": <record.selected_model>, "family": ..., "answer_ok": null}

The row field is matched_decision (F1): with string-only truth, a mismatched
decision is decision_mismatch, never E1_signal.

answer_ok stays null here: correctness is joined later from saved answers
(Sub-issue 2's answer bank). The evidence-first rule handles the null.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..contract import ContractError

REPLAYS_PATH = "/api/v1/observability/replays"


class ReplayFetchError(RuntimeError):
    """Raised when the management API cannot be reached or returns bad JSON."""


def fetch_replays(
    management_url: str,
    *,
    show_details: bool = True,
    timeout: float = 30.0,
    api_key_env: str | None = None,
) -> list[dict[str, Any]]:
    """Fetch replay records from the router management API.

    `management_url` is the router management base URL, e.g.
    http://127.0.0.1:8080. Returns the list of RoutingRecord dicts.
    """
    url = management_url.rstrip("/") + REPLAYS_PATH
    if show_details:
        url += "?showDetails=true"
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    import os

    if api_key_env and os.environ.get(api_key_env):
        request.add_header("Authorization", f"Bearer {os.environ[api_key_env]}")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise ReplayFetchError(f"management API returned HTTP {exc.code}: {url}") from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise ReplayFetchError(f"cannot reach management API at {url}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ReplayFetchError(f"management API returned non-JSON body: {exc}") from exc

    records = payload.get("data", payload) if isinstance(payload, dict) else payload
    if not isinstance(records, list):
        raise ReplayFetchError("unexpected replay list payload: expected {'data': [...]}" )
    return [record for record in records if isinstance(record, dict)]


@dataclass(frozen=True)
class CaseJoin:
    """Hand-labeled case list: one row per prompt sent through the router."""

    case_id: str
    prompt: str
    truth: str


def load_cases(path: str | Path) -> list[CaseJoin]:
    """Load the labeled case list (JSONL: case_id / prompt / truth)."""
    path = Path(path)
    if not path.is_file():
        raise ContractError(f"cases file not found: {path}")
    cases: list[CaseJoin] = []
    with path.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ContractError(f"cases line {lineno} is not valid JSON: {exc}") from exc
            for name in ("case_id", "prompt", "truth"):
                if not isinstance(raw.get(name), str) or not raw[name].strip():
                    raise ContractError(f"cases line {lineno}: {name} must be a non-empty string")
            cases.append(CaseJoin(case_id=raw["case_id"], prompt=raw["prompt"], truth=raw["truth"]))
    return cases


def _record_field(record: dict, *names: str) -> Any:
    """Return the first present field (tolerates camelCase/snake_case wires)."""
    for name in names:
        if name in record and record[name] is not None:
            return record[name]
    return None


def record_to_case(
    record: dict[str, Any],
    cases_by_prompt: dict[str, CaseJoin],
    *,
    family: str = "d3",
) -> tuple[dict[str, Any] | None, str]:
    """Convert one RoutingRecord into an attribution input row.

    Returns (row, join_state) where join_state is "matched" or the skip
    reason ("no_prompt", "unlabeled"). Only matched records become rows.
    """
    prompt = _record_field(record, "prompt", "Prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        return None, "no_prompt"
    case = cases_by_prompt.get(prompt)
    if case is None:
        return None, "unlabeled"
    decision = _record_field(record, "decision", "Decision") or ""
    selected = _record_field(record, "selected_model", "SelectedModel") or ""
    request_id = _record_field(record, "request_id", "RequestID") or ""
    row = {
        "case_id": case.case_id,
        "truth": case.truth,
        "matched_decision": decision,
        "routed": selected,
        "family": family,
        "answer_ok": None,  # correctness is joined later from the answer bank
        "request_id": str(request_id),
    }
    return row, "matched"


def records_to_cases(
    records: list[dict[str, Any]],
    cases: list[CaseJoin],
    *,
    family: str = "d3",
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Join replay records against the labeled case list.

    Returns (rows, join_stats). Duplicate prompts keep their first record;
    later duplicates count as skipped with reason "duplicate_prompt".
    """
    by_prompt: dict[str, CaseJoin] = {}
    for case in cases:
        by_prompt.setdefault(case.prompt, case)
    rows: list[dict[str, Any]] = []
    stats: dict[str, int] = {"matched": 0, "no_prompt": 0, "unlabeled": 0, "duplicate_prompt": 0}
    seen_prompts: set[str] = set()
    for record in records:
        row, state = record_to_case(record, by_prompt, family=family)
        if state == "matched":
            prompt = _record_field(record, "prompt", "Prompt")
            if prompt in seen_prompts:
                stats["duplicate_prompt"] += 1
                continue
            seen_prompts.add(prompt)
            rows.append(row)
        stats[state] = stats.get(state, 0) + 1
    return rows, stats


def write_cases(cases: list[CaseJoin], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for case in cases:
            fh.write(
                json.dumps(
                    {"case_id": case.case_id, "prompt": case.prompt, "truth": case.truth},
                    ensure_ascii=False,
                )
                + "\n"
            )
    return path
