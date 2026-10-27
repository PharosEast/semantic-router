"""Sub-issue 6 P3 tests: session-aware agentic attribution.

Evidence classes are layered on the single-turn classes, keyed by
(conversation, turn) from the router_replay session fields.
"""

from __future__ import annotations

import json

from recipe_lab.session_attribution import (
    SESSION_CLASS_LANE_FLAP,
    SESSION_CLASS_TOOL_LOOP_INTERRUPTED,
    analyze_sessions,
    render_session_report,
    write_session_report,
)


def row(conversation: str, turn: int, routed: str, **session_extra) -> dict:
    session = {"conversation_id": conversation, "turn_index": turn, **session_extra}
    return {"case_id": f"{conversation}-t{turn}", "routed": routed, "session": session}


def test_lane_flap_detected_with_router_reasoning():
    rows = [
        row("conv-A", 0, "local/private-qwen", session_action="establish_baseline"),
        row("conv-A", 1, "cloud/frontier-reasoning", session_action="switch",
            selection_reasoning="session_aware: previous_model_not_in_candidates"),
    ]
    analysis = analyze_sessions(rows)
    assert analysis["event_counts"][SESSION_CLASS_LANE_FLAP] == 1
    event = analysis["events"][0]
    assert event["turn_index"] == 1
    assert event["detail"]["from"] == "local/private-qwen"
    assert event["detail"]["to"] == "cloud/frontier-reasoning"
    assert event["detail"]["session_action"] == "switch"


def test_turn_gap_is_tool_loop_interrupted():
    # the 503-rejected turn never reaches replay: a recorded turn-index jump
    # is what that rejection looks like from the replay side (until P2 makes
    # rejections first-class records). The same adjacent pair may also carry a
    # lane flap - the two classes are independent evidence.
    rows = [
        row("conv-A", 0, "local/private-qwen"),
        row("conv-A", 1, "cloud/frontier-reasoning"),
        row("conv-A", 3, "local/private-qwen"),  # turn 2 missing
    ]
    analysis = analyze_sessions(rows)
    assert analysis["event_counts"][SESSION_CLASS_TOOL_LOOP_INTERRUPTED] == 1
    gap = next(e for e in analysis["events"] if e["class"] == SESSION_CLASS_TOOL_LOOP_INTERRUPTED)
    assert gap["turn_index"] == 3
    assert gap["detail"] == {"after_turn": 1, "resumed_at": 3}


def test_single_turn_rows_are_ignored_here():
    rows = [{"case_id": "solo", "routed": "local/private-qwen"}]
    analysis = analyze_sessions(rows)
    assert analysis["single_turn_rows"] == 1
    assert analysis["sessions"] == []
    assert analysis["events"] == []


def test_report_is_deterministic_and_honest():
    rows = [
        row("conv-B", 0, "local/private-qwen"),
        row("conv-A", 0, "local/private-qwen"),
        row("conv-A", 2, "cloud/frontier-reasoning", session_action="switch"),
    ]
    first = render_session_report(analyze_sessions(rows))
    second = render_session_report(analyze_sessions(rows))
    assert first == second  # byte-identical across renders
    assert first.startswith("# Session-aware agentic attribution report")
    assert "statistical significance is not established" in first
    # conv-A sorts before conv-B; the flap detail carries the router reasoning
    assert "`conv-A`" in first and "`conv-B`" in first
    assert "previous_model_not_in_candidates" not in first  # no reasoning given on this flap


def test_write_session_report_artifacts(tmp_path):
    rows = [row("conv-A", 0, "local"), row("conv-A", 2, "cloud")]
    summary = write_session_report(rows, str(tmp_path))
    assert summary["sessions"] == 1
    assert summary["events"][SESSION_CLASS_TOOL_LOOP_INTERRUPTED] == 1
    report = (tmp_path / "session-report.md").read_text(encoding="utf-8")
    events = json.loads((tmp_path / "session-events.json").read_text(encoding="utf-8"))
    assert "Session-aware agentic attribution report" in report
    assert len(events) == 2  # a lane flap and the turn gap, independently recorded
