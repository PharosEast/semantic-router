"""Agentic-traffic probe for the recipe-lab smoke stack (Sub-issue 6 groundwork).

Drives three multi-turn / tool-calling scenarios through the router and
reports per-turn routing decisions from the x-vsr-model-usage response header
plus the router_replay dump:

- A. one coding conversation (stable x-session-id / x-conversation-id) where
  difficulty jumps mid-conversation: session protection should keep the
  conversation on one lane instead of flapping;
- B. the same hard question in a FRESH conversation: without session state it
  should route on its own merits (the sticky-vs-fresh contrast pair);
- C. a tool-calling turn (OpenAI `tools` + assistant `tool_calls` + `role:
  tool` result) to check what router_replay captures for agent traffic.

No GPU and no external agent binary needed - this is the v1 boundary probe
(recipe-lab design §6.1): what the current single-turn loop can
and cannot see about agentic traffic.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROUTER = "http://127.0.0.1:8899"
MGMT = "http://127.0.0.1:8080"

TOOLS_PAYLOAD = {
    "tools": [
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read a file from the workspace",
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "run_tests",
                "description": "Run the test suite",
                "parameters": {"type": "object", "properties": {}},
            },
        },
    ]
}


def post(body: dict, session: str | None, conversation: str | None) -> tuple[int, dict, dict]:
    headers = {"Content-Type": "application/json"}
    if session:
        headers["x-session-id"] = session
    if conversation:
        headers["x-conversation-id"] = conversation
    request = urllib.request.Request(
        ROUTER + "/v1/chat/completions", data=json.dumps(body).encode("utf-8"), headers=headers
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            return response.status, json.loads(response.read().decode("utf-8")), dict(response.headers)
    except urllib.error.HTTPError as exc:
        return exc.code, {"error": exc.read().decode("utf-8", "replace")[:400]}, dict(exc.headers)


def model_from(headers: dict, body: dict) -> str:
    usage = headers.get("x-vsr-model-usage") or ""
    if usage:
        return "hdr:" + usage[:120]
    return "body:" + str(body.get("model", ""))


def dump_replays(output: Path) -> list[dict]:
    try:
        with urllib.request.urlopen(MGMT + "/api/v1/observability/replays?showDetails=true", timeout=30) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, json.JSONDecodeError) as exc:
        print(f"replay dump failed: {exc}", file=sys.stderr)
        return []
    rows = payload.get("replays") or payload.get("records") or payload
    if isinstance(rows, dict):
        rows = list(rows.values())
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    session = "probe-session-001"
    conversation = "probe-conv-001"
    transcript: list[dict] = []

    def turn(name: str, messages: list[dict], **extra: object) -> None:
        body = {"model": "vllm-sr/auto", "max_tokens": 64, "messages": messages, **extra}
        status, data, headers = post(body, session, conversation)
        entry = {
            "turn": name,
            "status": status,
            "model": model_from(headers, data),
            "vsr_headers": {k: v for k, v in headers.items() if k.startswith("x-vsr-")},
        }
        transcript.append(entry)
        print(json.dumps(entry, ensure_ascii=False))

    # --- A. one conversation, difficulty jumps mid-way ---
    turn("A1-simple-greeting", [{"role": "user", "content": "Say hello in French."}])
    turn(
        "A2-hard-architecture",
        [
            {"role": "user", "content": "Say hello in French."},
            {"role": "assistant", "content": "Bonjour!"},
            {
                "role": "user",
                "content": "Now redesign our distributed rate limiter: compare token bucket, sliding window "
                "and GCRA from first principles, analyse the failure modes under clock skew, and justify "
                "a recommendation for a multi-region deployment.",
            },
        ],
    )
    # --- C. tool-calling turns inside the same conversation ---
    turn(
        "A3-tool-call-request",
        [
            {"role": "user", "content": "Fix the failing test in src/limiter.py, then run the suite."},
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "call-1", "type": "function", "function": {"name": "read_file", "arguments": '{"path": "src/limiter.py"}'}}
            ]},
            {"role": "tool", "tool_call_id": "call-1", "content": "def limit(r): return r > 10  # bug: off-by-one"},
        ],
        **TOOLS_PAYLOAD,
    )
    turn(
        "A4-tool-result-followup",
        [
            {"role": "user", "content": "Fix the failing test in src/limiter.py, then run the suite."},
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "call-1", "type": "function", "function": {"name": "read_file", "arguments": '{"path": "src/limiter.py"}'}}
            ]},
            {"role": "tool", "tool_call_id": "call-1", "content": "def limit(r): return r > 10"},
            {"role": "assistant", "content": "The boundary check is off by one; I'll fix it and run tests."},
            {"role": "user", "content": "Good. Now summarise what you changed."},
        ],
        **TOOLS_PAYLOAD,
    )

    # --- B. fresh conversation with the same hard question ---
    body = {
        "model": "vllm-sr/auto",
        "max_tokens": 64,
        "messages": [
            {
                "role": "user",
                "content": "Redesign our distributed rate limiter: compare token bucket, sliding window "
                "and GCRA from first principles, analyse the failure modes under clock skew, and justify "
                "a recommendation for a multi-region deployment.",
            }
        ],
    }
    status, data, headers = post(body, "probe-session-002", "probe-conv-002")
    transcript.append({
        "turn": "B-fresh-hard",
        "status": status,
        "model": model_from(headers, data),
        "vsr_headers": {k: v for k, v in headers.items() if k.startswith("x-vsr-")},
    })
    print(json.dumps(transcript[-1], ensure_ascii=False))

    (out / "probe-transcript.json").write_text(json.dumps(transcript, ensure_ascii=False, indent=1), encoding="utf-8")

    # --- replay dump: what the router recorded about this agent traffic ---
    rows = dump_replays(out / "probe-replays.json")
    summary = []
    for row in rows:
        details = row.get("details", row) if isinstance(row, dict) else {}
        summary.append(
            {
                "conversation": details.get("conversation_id") or details.get("x-conversation-id"),
                "selected_model": details.get("selected_model") or details.get("routed"),
                "decision": details.get("decision"),
                "has_tools": "tools" in json.dumps(details)[:20000],
                "body_bytes_captured": len(json.dumps(details, ensure_ascii=False)),
            }
        )
    (out / "probe-replay-summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print(f"replay records: {len(rows)}; summary written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
