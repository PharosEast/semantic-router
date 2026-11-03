"""AG3: route-replay tier - cases from-replays, session rebuild, protection
detection over a fake router stack."""

from __future__ import annotations

import json
import socket
import sys
from pathlib import Path

import pytest

from recipe_lab.runner import RunnerEnv, evaluate_route_replay, route_replay_evidence
from recipe_lab.session_cases import (
    TOOL_LOOP_PHASE,
    USER_TURN_PHASE,
    cases_from_replays,
    load_route_cases,
    request_phase_of,
)

FIXTURES = Path(__file__).resolve().parent / "fixtures"
J3_TRIMMED = FIXTURES / "j3-records-trimmed.jsonl"


def _record(
    request_id: str,
    conversation: str,
    turn: int,
    prompt: str,
    model: str = "local/model-a",
    *,
    body: bool = True,
    ts: str = "",
):
    record = {
        "request_id": request_id,
        "conversation_id": conversation,
        "session_id": conversation,
        "turn_index": turn,
        "timestamp": ts,
        "prompt": prompt,
        "selected_model": model,
        "decision": "domain_code",
    }
    if body:
        record["request_body"] = json.dumps(
            {"messages": [{"role": "user", "content": prompt}]}
        )
    return record


# ---- cases from-replays: the three granularities ----


def test_session_granularity_groups_and_orders():
    records = [
        _record("r3", "conv-1", 1, "follow up", ts="03"),
        _record("r1", "conv-1", 0, "first ask", ts="01"),
        _record("r2", "conv-2", 0, "other session", ts="02"),
    ]
    cases, provenance = cases_from_replays(records, "session")
    assert provenance["cases"] == 2
    assert provenance["granularity"] == "session"
    assert provenance["dedup_rule"].startswith("one case per conversation")
    by_id = {case["session_id"]: case for case in cases}
    # requests keep replay order inside the conversation, not arrival order
    assert [r["prompt"] for r in by_id["conv-1"]["requests"]] == [
        "first ask",
        "follow up",
    ]
    assert by_id["conv-1"]["case_id"].startswith("sess-")
    assert all(r["phase"] == USER_TURN_PHASE for r in by_id["conv-1"]["requests"])


def test_prompt_granularity_is_the_j3_dedup():
    records = [
        _record("r1", "conv-1", 0, "same prompt"),
        _record("r2", "conv-1", 1, "same prompt"),
        _record("r3", "conv-1", 2, "same prompt"),
        _record("r4", "conv-1", 3, "same prompt"),
        _record("r5", "conv-2", 0, "other"),
    ]
    cases, provenance = cases_from_replays(records, "prompt")
    assert [c["prompt"] for c in cases] == ["same prompt", "other"]
    assert all(c["case_id"].startswith("uq-") for c in cases)
    # four of five records were duplicate prompts at this granularity
    assert provenance["duplicate_pct"] == pytest.approx(60.0)
    assert provenance["duplicate_warning"] is True

    request_cases, request_prov = cases_from_replays(records, "request")
    assert len(request_cases) == 5
    assert request_prov["duplicate_warning"] is False


def test_records_without_identity_or_body_are_skipped_with_reasons():
    records = [
        {"prompt": "no conversation identity"},
        _record("r1", "conv-1", 0, "real", body=False),
        _record("r2", "conv-2", 0, "synthesized from prompt", body=False),
    ]
    cases, provenance = cases_from_replays(records, "session")
    # the body-less record still rebuilds from its recorded prompt
    assert len(cases) == 2
    assert provenance["skipped"]["no_identity"] == 1
    assert provenance["skipped"]["no_messages"] == 0


def test_tool_loop_phase_rule_mirrors_the_router():
    assert request_phase_of([{"role": "user", "content": "go"}]) == USER_TURN_PHASE
    assert (
        request_phase_of(
            [
                {"role": "user", "content": "go"},
                {"role": "assistant", "tool_calls": [{"id": "t"}]},
                {"role": "tool", "tool_call_id": "t", "content": "out"},
            ]
        )
        == TOOL_LOOP_PHASE
    )
    assert (
        request_phase_of(
            [
                {"role": "user", "content": "go"},
                {"role": "assistant", "tool_calls": [{"id": "t"}]},
            ]
        )
        == TOOL_LOOP_PHASE
    )


def test_cli_cases_from_replays_writes_provenance(tmp_path, capsys):
    from recipe_lab.cli import main

    records_path = tmp_path / "records.jsonl"
    records_path.write_text(
        "\n".join(
            json.dumps(r)
            for r in [
                _record("r1", "conv-1", 0, "first ask"),
                _record("r2", "conv-1", 1, "follow up"),
                _record("r3", "conv-2", 0, "other session"),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    out = tmp_path / "cases.jsonl"
    assert (
        main(
            [
                "cases",
                "from-replays",
                "--records",
                str(records_path),
                "--granularity",
                "session",
                "--output",
                str(out),
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["cases_count"] == 2
    assert payload["granularity"] == "session"
    loaded = load_route_cases(str(out))
    assert len(loaded) == 2
    provenance = json.loads(
        Path(payload["provenance"]).read_text(encoding="utf-8")
    )
    assert provenance["dedup_rule"].startswith("one case per conversation")


# ---- the J3 pilot's 344 records rebuild to the pilot's sessions ----


def test_j3_records_rebuild_to_pilot_sessions_and_unique_prompts():
    records = [
        json.loads(line)
        for line in J3_TRIMMED.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(records) == 344
    # session granularity: one case per pilot task conversation
    # (12 SWE-bench tasks + 1 auxiliary session in the pilot capture)
    session_cases, session_prov = cases_from_replays(records, "session")
    assert session_prov["cases"] == 13
    assert session_prov["duplicate_pct"] == pytest.approx(
        (1 - 13 / 344) * 100, abs=0.1
    )
    multi = [
        case for case in session_cases if len(case["requests"]) > 1
    ]
    assert len(multi) == 12  # one pilot session captured a single request
    # prompt granularity: the J3 rebuild's 14 unique prompts
    prompt_cases, prompt_prov = cases_from_replays(records, "prompt")
    assert prompt_prov["cases"] == 14
    # the default granularity for identity-bearing (agentic) records is session
    auto_cases, auto_prov = cases_from_replays(records, "session")
    assert auto_prov["granularity"] == "session"
    assert len(auto_cases) == len(session_cases)


# ---- route_replay_evidence: protection detection ----


def _replay_row(request_id, conversation, turn, model, decision="domain_code"):
    return {
        "request_id": request_id,
        "routed": model,
        "matched_decision": decision,
        "record_ts": f"2026-11-03T00:00:{turn:02d}",
        "session": {
            "conversation_id": conversation,
            "session_id": conversation,
            "turn_index": turn,
        },
    }


def test_evidence_counts_tool_loop_switches_and_resolves_gates():
    # one conversation: user turn -> tool loop (switch!) -> tool loop (same model)
    records = [
        {
            "request_id": "rl-rt-x-s1-0",
            "prompt": "start",
            "decision": "domain_code",
            "selected_model": "local/model-a",
            "conversation_id": "s1",
            "turn_index": 0,
            "timestamp": "01",
        },
        {
            "request_id": "rl-rt-x-s1-1",
            "prompt": "tool follow",
            "decision": "domain_code",
            "selected_model": "local/model-b",
            "conversation_id": "s1",
            "turn_index": 0,
            "timestamp": "02",
        },
        {
            "request_id": "rl-rt-x-s1-2",
            "prompt": "tool again",
            "decision": "domain_code",
            "selected_model": "local/model-b",
            "conversation_id": "s1",
            "turn_index": 0,
            "timestamp": "03",
        },
    ]
    phases = {
        "rl-rt-x-s1-0": USER_TURN_PHASE,
        "rl-rt-x-s1-1": TOOL_LOOP_PHASE,
        "rl-rt-x-s1-2": TOOL_LOOP_PHASE,
    }
    evidence = route_replay_evidence(
        records,
        phases_by_request_id=phases,
        gates={"max_tool_loop_switches": 0, "max_mean_switches_per_task": 1.0},
    )
    summary = evidence["summary"]
    assert summary["sessions"] == 1
    assert summary["tool_loop_switches"] == 1
    assert summary["mean_switches_per_session"] == 1.0
    assert summary["model_shares"] == {
        "local/model-a": pytest.approx(1 / 3),
        "local/model-b": pytest.approx(2 / 3),
    }
    assert summary["first_turn_decisions"] == {"domain_code": 1}
    gates = {g.name: g for g in evidence["gate_results"]}
    assert gates["max_tool_loop_switches"].passed is False
    assert gates["max_mean_switches_per_task"].passed is True

    # a protected candidate keeps one model through the whole tool loop
    protected_records = [
        dict(record, selected_model="local/model-a") for record in records
    ]
    protected = route_replay_evidence(
        protected_records,
        phases_by_request_id=phases,
        gates={"max_tool_loop_switches": 0},
    )
    assert protected["summary"]["tool_loop_switches"] == 0
    protected_gates = {g.name: g for g in protected["gate_results"]}
    assert protected_gates["max_tool_loop_switches"].passed is True


# ---- truth namespace: behavior labels never ground decision_mismatch ----


def test_behavior_truth_space_constrains_the_model_route_only():
    from recipe_lab.attribution import attribute
    from recipe_lab.contract import OraclePolicy

    # a behavior label ("correct" outcome) mapped to a model: the mapping
    # constrains the route only - a decision-name mismatch is not a thing
    policy = OraclePolicy.from_dict(
        {"correct": "local/model-a"}, truth_space="behavior"
    )
    rows = attribute(
        [
            {
                "case_id": "c1",
                "truth": "correct",
                "matched_decision": "unrelated_decision",
                "routed": "local/model-b",
                "answer_ok": True,
            },
            {
                "case_id": "c2",
                "truth": "correct",
                "matched_decision": "unrelated_decision",
                "routed": "local/model-a",
                "answer_ok": False,
            },
            {
                "case_id": "c3",
                "truth": "unmapped behavior",
                "matched_decision": "domain_code",
                "routed": "local/model-a",
                "answer_ok": True,
            },
        ],
        policy,
    )
    classes = {row["case_id"]: row["error_class"] for row in rows}
    # routed off the mapped model with a right answer: suboptimal, never
    # decision_mismatch on the behavior label
    assert classes["c1"] == "suboptimal_routing"
    # routed to the mapped model with a wrong answer: capability, not policy
    assert classes["c2"] == "E3_capability"
    # an unmapped behavior label stays policy_gap
    assert classes["c3"] == "policy_gap"

    # the legacy decision namespace keeps the F1 mapping semantics: a missed
    # decision with an unknown-outcome answer is decision_mismatch
    decision_policy = OraclePolicy.from_dict(
        {"domain_code": "local/model-a"}, truth_space="decision"
    )
    decision_rows = attribute(
        [
            {
                "case_id": "c1",
                "truth": "domain_code",
                "matched_decision": "other",
                "routed": "local/model-a",
                "answer_ok": False,
            }
        ],
        decision_policy,
    )
    assert decision_rows[0]["error_class"] == "decision_mismatch"


def test_contract_truth_space_validation():
    import copy

    from recipe_lab.contract import ContractError, ExperimentSpec
    from tests.fixtures import VALID_SPEC

    spec_dict = copy.deepcopy(VALID_SPEC)
    spec_dict["attribution"] = {
        "truth_space": "behavior",
        "oracle_policy": {"correct": "local/model-a"},
    }
    spec = ExperimentSpec.from_dict(spec_dict)
    assert spec.oracle_policy.truth_space == "behavior"
    assert spec.to_dict()["attribution"]["truth_space"] == "behavior"

    bad = copy.deepcopy(spec_dict)
    bad["attribution"] = {
        "truth_space": "vibes",
        "oracle_policy": {"correct": "local/model-a"},
    }
    with pytest.raises(ContractError, match="truth_space"):
        ExperimentSpec.from_dict(bad)


# ---- integration: the probe over a fake router stack ----

FAKE_ROUTE_REPLAY_ROUTER = """#!/usr/bin/env python3
import hashlib, json, os, sys, threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import yaml

config_path = sys.argv[sys.argv.index("-config") + 1]
cfg = yaml.safe_load(open(config_path, encoding="utf-8"))
GATE = int(cfg["listeners"][0]["port"])
API = int(sys.argv[sys.argv.index("-api-port") + 1])
MODE = cfg.get("x-route-mode", "protected")
# AG6: the fake router honours the real protection knob - a raised
# stability weight holds the lane through the tool loop
try:
    WEIGHT = float(
        cfg["global"]["router"]["learning"]["protection"]["tuning"]["stability_weight"]
    )
except (KeyError, TypeError, ValueError):
    WEIGHT = 1.0
STATE = os.environ["FAKE_STATE"]
lock = threading.Lock()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def _send(self, code, payload):
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        if not self.path.startswith("/v1/chat/completions"):
            self._send(404, {"error": "not found"})
            return
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))) or b"{}")
        messages = body.get("messages") or []
        prompt = next(
            (m.get("content", "") for m in reversed(messages) if m.get("role") == "user"),
            "",
        )
        last = messages[-1] if messages else {}
        in_tool_loop = last.get("role") == "tool" or (
            last.get("role") == "assistant" and last.get("tool_calls")
        )
        conversation = self.headers.get("x-conversation-id", "")
        with lock:
            state = json.load(open(STATE, encoding="utf-8"))
            counter = state["turns"].get(conversation, 0)
            state["turns"][conversation] = counter + 1
            if MODE == "protected" or not in_tool_loop or WEIGHT >= 2.0:
                model = "local/model-a"
            else:
                # an unprotected, low-stability candidate flaps onto the
                # strong model mid-loop
                model = "local/model-strong"
            state["records"].append(
                {
                    "request_id": self.headers.get("x-request-id", ""),
                    "prompt": prompt,
                    "decision": "domain_code",
                    "selected_model": model,
                    "conversation_id": conversation,
                    "session_id": self.headers.get("x-session-id", ""),
                    "turn_index": counter,
                    "timestamp": "2026-11-03T00:00:%02d" % counter,
                }
            )
            json.dump(state, open(STATE, "w", encoding="utf-8"))
        out = json.dumps({"choices": [{"message": {"content": "ok"}}]}).encode()
        self._send(200, json.loads(out))

    def do_GET(self):
        if self.path.startswith("/api/v1/observability/replays"):
            state = json.load(open(STATE, encoding="utf-8"))
            self._send(200, {"data": state["records"], "has_more": False})
            return
        self._send(404, {"error": "not found"})


gateway = ThreadingHTTPServer(("127.0.0.1", GATE), Handler)
management = ThreadingHTTPServer(("127.0.0.1", API), Handler)
with open(os.path.join(os.path.dirname(config_path), "router-runtime.json"), "w", encoding="utf-8") as fh:
    json.dump({"ready": True}, fh)
threading.Thread(target=management.serve_forever, daemon=True).start()
gateway.serve_forever()
"""


def _free_port() -> int:
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def _session_case(case_id: str, identity: str) -> dict:
    """A recorded session: one user turn then two in-loop requests."""
    return {
        "case_id": case_id,
        "session_id": identity,
        "prompt": "trace the rendering bug",
        "requests": [
            {
                "prompt": "trace the rendering bug",
                "phase": USER_TURN_PHASE,
                "messages": [{"role": "user", "content": "trace the rendering bug"}],
            },
            {
                "prompt": "apply the fix",
                "phase": TOOL_LOOP_PHASE,
                "messages": [
                    {"role": "user", "content": "trace the rendering bug"},
                    {"role": "assistant", "tool_calls": [{"id": "t1"}]},
                    {"role": "tool", "tool_call_id": "t1", "content": "patched"},
                ],
            },
            {
                "prompt": "run the tests",
                "phase": TOOL_LOOP_PHASE,
                "messages": [
                    {"role": "user", "content": "trace the rendering bug"},
                    {"role": "assistant", "tool_calls": [{"id": "t1"}]},
                    {"role": "tool", "tool_call_id": "t1", "content": "patched"},
                    {"role": "assistant", "content": "running tests"},
                    {"role": "tool", "tool_call_id": "t2", "content": "green"},
                ],
            },
        ],
    }


@pytest.fixture()
def route_stack(tmp_path):
    router = tmp_path / "fake_router.py"
    router.write_text(FAKE_ROUTE_REPLAY_ROUTER)
    router.chmod(0o755)
    mock = tmp_path / "mock_backend.py"
    mock.write_text("import time\ntime.sleep(300)\n")

    def make(mode: str, *, stability_weight: float = 1.0) -> tuple[Path, RunnerEnv]:
        state = tmp_path / f"state-{mode}.json"
        state.write_text(json.dumps({"records": [], "turns": {}}), encoding="utf-8")
        config = tmp_path / f"candidate-{mode}.yaml"
        config.write_text(
            "version: v0.2\n"
            "listeners:\n"
            "  - name: http\n"
            "    address: 0.0.0.0\n"
            "    port: 8899\n"
            f"x-route-mode: {mode}\n"
            "global:\n"
            "  router:\n"
            "    learning:\n"
            "      protection:\n"
            "        enabled: true\n"
            "        scope: conversation\n"
            "        tuning:\n"
            f"          stability_weight: {stability_weight}\n"
        )
        env = RunnerEnv(
            router_bin=str(router),
            mock_backend_script=str(mock),
            python_bin=sys.executable,
            work_dir=tmp_path / f"work-{mode}",
            backend_port=_free_port(),
            ready_timeout_s=30,
            env_extra={"FAKE_STATE": str(state)},
        )
        return config, env

    return make


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="the runner's process-group teardown needs os.killpg (POSIX)",
)
@pytest.mark.parametrize("mode", ["protected", "unprotected"])
def test_route_replay_detects_tool_loop_switches(route_stack, tmp_path, mode):
    cases = [_session_case("s1", "conv-replay-1")]
    config, env = route_stack(mode)
    result = evaluate_route_replay(
        candidate_config=config,
        cases=cases,
        env=env,
        router_model="vllm-sr/auto",
        request_timeout=30.0,
        trial_timeout_s=60,
        request_id_prefix="rl-rt-test-",
        gates={"max_tool_loop_switches": 0},
    )
    summary = result["summary"]
    assert summary["sessions"] == 1
    assert summary["records"] == 3
    gate = {g.name: g for g in result["gate_results"]}["max_tool_loop_switches"]
    if mode == "protected":
        assert summary["tool_loop_switches"] == 0
        assert summary["model_shares"] == {"local/model-a": 1.0}
        assert gate.passed is True
    else:
        # the unprotected candidate flapped onto the strong model mid-loop
        assert summary["tool_loop_switches"] >= 1
        assert summary["model_shares"]["local/model-strong"] > 0
        assert gate.passed is False
