"""Session-aware attribution over replay records (Sub-issue 6, P3).

Single-turn attribution (attribution.py) keys evidence per request. Agentic
traffic lives per (conversation, turn): the replay record already carries the
session evidence (conversation_id / turn_index / session_action /
selection_reasoning), so this module aggregates rows into sessions and layers
three agentic evidence classes on top of the single-turn E1/E2/E3 classes -
it never redefines them:

- lane_flap            adjacent turns of one conversation selected different
                       models (may be a legitimate switch or a protection
                       miss; reported with the router's own session reasoning)
- tool_loop_interrupted a turn gap inside a conversation: the recorded
                       turn_index sequence jumps, which is what a rejected
                       (503) selection looks like from the replay side on
                       routers without the standalone rejection-recording fix
                       (lifecycle_state=failed +
                       terminal_reason=selection_rejected)
- selection_failed     the explicit rejection records the router emits for
                       503 no_eligible_model selections once the standalone
                       router fix is merged (lifecycle_state=failed +
                       terminal_reason=selection_rejected); on unfixed
                       routers nothing emits them, the class exists so
                       downstream consumers can already branch on it

Pure functions, no I/O, deterministic: same rows -> same report bytes.
"""

from __future__ import annotations

import json
from collections import Counter
from typing import Any

SESSION_CLASS_LANE_FLAP = "lane_flap"
SESSION_CLASS_TOOL_LOOP_INTERRUPTED = "tool_loop_interrupted"
SESSION_CLASS_SELECTION_FAILED = "selection_failed"
SESSION_CLASSES = (
    SESSION_CLASS_LANE_FLAP,
    SESSION_CLASS_TOOL_LOOP_INTERRUPTED,
    SESSION_CLASS_SELECTION_FAILED,
)


def _session_of(row: dict[str, Any]) -> dict[str, Any] | None:
    session = row.get("session")
    return session if isinstance(session, dict) else None


def _conversation_key(row: dict[str, Any]) -> str | None:
    session = _session_of(row)
    if not session:
        return None
    for key in ("conversation_id", "session_id"):
        value = session.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return None


def _turn_index(row: dict[str, Any]) -> int | None:
    session = _session_of(row)
    if not session:
        return None
    value = session.get("turn_index")
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _event(
    conversation: str,
    kind: str,
    *,
    turn_index: int | None = None,
    detail: dict[str, Any] | None = None,
) -> dict[str, Any]:
    event: dict[str, Any] = {"conversation": conversation, "class": kind}
    if turn_index is not None:
        event["turn_index"] = turn_index
    if detail:
        event["detail"] = detail
    return event


def analyze_sessions(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate attribution rows into per-conversation agentic evidence.

    Rows without session fields are counted as single-turn and ignored here
    (the single-turn attribution owns them). Deterministic output: sessions
    sorted by key, turns by index.
    """
    sessions: dict[str, list[dict[str, Any]]] = {}
    single_turn = 0
    for row in rows:
        key = _conversation_key(row)
        if key is None:
            single_turn += 1
            continue
        sessions.setdefault(key, []).append(row)

    events: list[dict[str, Any]] = []
    session_summaries: list[dict[str, Any]] = []
    for conversation in sorted(sessions):
        turns = sorted(
            (row for row in sessions[conversation] if _turn_index(row) is not None),
            key=_turn_index,
        )
        lane_flaps = 0
        turn_gaps = 0
        for i in range(1, len(turns)):
            prev, curr = turns[i - 1], turns[i]
            prev_model = str(prev.get("routed") or "")
            curr_model = str(curr.get("routed") or "")
            prev_index, curr_index = _turn_index(prev), _turn_index(curr)
            if prev_model and curr_model and prev_model != curr_model:
                lane_flaps += 1
                session = _session_of(curr) or {}
                events.append(
                    _event(
                        conversation,
                        SESSION_CLASS_LANE_FLAP,
                        turn_index=curr_index,
                        detail={
                            "from": prev_model,
                            "to": curr_model,
                            "session_action": session.get("session_action"),
                            "selection_reasoning": session.get("selection_reasoning"),
                        },
                    )
                )
            if curr_index - prev_index > 1:
                turn_gaps += 1
                events.append(
                    _event(
                        conversation,
                        SESSION_CLASS_TOOL_LOOP_INTERRUPTED,
                        turn_index=curr_index,
                        detail={"after_turn": prev_index, "resumed_at": curr_index},
                    )
                )
        session_summaries.append(
            {
                "conversation": conversation,
                "turns_recorded": len(turns),
                "turn_span": [_turn_index(turns[0]), _turn_index(turns[-1])] if turns else [],
                "lane_flaps": lane_flaps,
                "turn_gaps": turn_gaps,
            }
        )
    counts = Counter(event["class"] for event in events)
    return {
        "sessions": session_summaries,
        "events": events,
        "event_counts": {cls: counts.get(cls, 0) for cls in SESSION_CLASSES},
        "single_turn_rows": single_turn,
        "session_rows": sum(len(v) for v in sessions.values()),
    }


def render_session_report(analysis: dict[str, Any]) -> str:
    """Render the session-level agentic evidence report (markdown)."""
    lines = [
        "# Session-aware agentic attribution report",
        "",
        f"- conversations: {len(analysis['sessions'])} ({analysis['session_rows']} session rows, "
        f"{analysis['single_turn_rows']} single-turn rows ignored here)",
        "- evidence classes: lane_flap / tool_loop_interrupted / selection_failed "
        "(layered on the single-turn E1/E2/E3 classes, never replacing them)",
        "",
        "## Event counts",
        "",
        "| class | count |",
        "|---|---|",
    ]
    for cls in SESSION_CLASSES:
        lines.append(f"| {cls} | {analysis['event_counts'][cls]} |")
    lines += ["", "## Sessions", "", "| conversation | turns | span | lane_flaps | turn_gaps |", "|---|---|---|---|---|"]
    for session in analysis["sessions"]:
        span = session["turn_span"]
        span_text = f"{span[0]}..{span[1]}" if span else "-"
        lines.append(
            f"| `{session['conversation']}` | {session['turns_recorded']} | {span_text} "
            f"| {session['lane_flaps']} | {session['turn_gaps']} |"
        )
    if analysis["events"]:
        lines += ["", "## Events", ""]
        for event in analysis["events"]:
            detail = json.dumps(event.get("detail") or {}, ensure_ascii=False, sort_keys=True)
            lines.append(
                f"- `{event['class']}` conversation `{event['conversation']}`"
                + (f" turn {event['turn_index']}" if event.get("turn_index") is not None else "")
                + f" {detail}"
            )
    else:
        lines += ["", "## Events", "", "- none"]
    lines += [
        "",
        "---",
        "",
        "Replayed from recipe-lab records only; descriptive, statistical "
        "significance is not established.",
    ]
    return "\n".join(lines) + "\n"


def write_session_report(rows: list[dict[str, Any]], output_dir: str) -> dict[str, Any]:
    """Analyze + render + write session attribution artifacts; returns summary."""
    from pathlib import Path

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    analysis = analyze_sessions(rows)
    report_path = out / "session-report.md"
    report_path.write_text(render_session_report(analysis), encoding="utf-8")
    events_path = out / "session-events.json"
    events_path.write_text(
        json.dumps(analysis["events"], ensure_ascii=False, indent=1, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return {
        "sessions": len(analysis["sessions"]),
        "events": analysis["event_counts"],
        "session_report": str(report_path),
        "session_events": str(events_path),
    }
