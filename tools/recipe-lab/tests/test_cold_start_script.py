"""The cold-start orchestration script (C10) stays honest.

The script chains the generation flow (split -> observe-labels -> profile ->
bootstrap -> bank import -> run -> verify --final -> bundle) with inputs from
environment variables only. These checks keep it aligned with the CLI surface
and free of private content without executing any step.
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
from pathlib import Path

import pytest
from recipe_lab.bundle import scan_private_content
from recipe_lab.cli import build_parser

SCRIPT = Path(__file__).resolve().parents[1] / "examples/cold-start/cold-start.sh"


def _script_text() -> str:
    return SCRIPT.read_text(encoding="utf-8")


def _parser_map() -> dict:
    for action in build_parser()._actions:
        if isinstance(action, argparse._SubParsersAction):
            return dict(action.choices)
    raise AssertionError("build_parser() registers no subcommands")


def test_script_syntax():
    if shutil.which("bash") is None:
        pytest.skip("bash is not available")
    proc = subprocess.run(
        ["bash", "-n", str(SCRIPT)],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stderr


def test_script_invocations_exist():
    """Every recipe-lab call in the script names a real subcommand (S4 style).

    Quoted form catches both shell calls (`"$RL/recipe-lab" split`) and the
    prose examples inside the echo messages; the bare comment header
    ("recipe-lab cold start") is deliberately not matched.
    """
    parsers = _parser_map()
    unknown = []
    text = _script_text()
    for m in re.finditer(r"[\"']recipe-lab\s+([a-z][a-z0-9-]*)(\s+(\S+))?", text):
        name, second = m.group(1), m.group(3)
        if name not in parsers:
            unknown.append(f"recipe-lab {name}")
            continue
        sub_names = set()
        for action in parsers[name]._actions:
            if isinstance(action, argparse._SubParsersAction):
                sub_names = set(action.choices)
        if (
            second
            and re.fullmatch(r"[a-z][a-z0-9-]*", second)
            and sub_names
            and second not in sub_names
        ):
            unknown.append(f"recipe-lab {name} {second}")
    assert not unknown, "cold-start script uses nonexistent commands: " + ", ".join(
        unknown
    )


def test_script_has_no_private_content():
    """The script never embeds private paths (inputs come from env vars)."""
    text = _script_text()
    scan_private_content({"cold-start.sh": text}, [])
    # beyond the bundle's own patterns: no Windows drive-letter path at all
    scan_private_content({"cold-start.sh": text}, [r"[A-Za-z]:\\"])


def test_script_covers_y1_model_eval_chain():
    """Y1: the default model_eval driver observes the frozen split too (its
    verify rewrites the task categories from those C2 labels), pins the
    answer-plan fixture and the validate binary, ranks only bootstrap
    candidates, auto-picks the winner from `ledger top` and patches verify
    copies so the Y4 hash check names the file the candidate router served."""
    text = _script_text()
    # Y1 gate: frozen_test is observed through the seed config as well
    assert 'OBSERVE_SPLITS="calib dev frozen_test"' in text
    # fixture inputs and validator override of the model_eval driver
    for token in (
        "DRIVER",
        "ANSWER_PLAN",
        "CLASSIFICATION_CONFIG",
        "VALIDATE_BIN",
        "VERIFY_REPETITIONS",
    ):
        assert token in text
    # only candidate-lambda-*.yaml files are ranked (never the scaffolds or
    # the seed config that share the bootstrap out dir)
    assert "candidate-lambda-*.yaml" in text
    # winner auto-pick + patched verify copies (Y4 hash binding)
    for token in (
        "ledger top",
        "to-model-eval",
        "patch_listener",
        "--candidate-management-endpoint",
        "--candidate-config",
    ):
        assert token in text
