"""sr-bench CLI adapter (Sub-issue 3).

Thin, honest wrapper around `vllm-sr benchmark`: recipe-lab owns the loop,
sr-bench owns evaluation - this module only shells out, parses, and reads.

Verified against the actual CLI (I, src/vllm-sr/cli/commands/benchmark.py):

- `preview` and `run` take `--manifest <path>` (never a run id) and, without
  `--detach`, block until the run finishes and print its report. This module
  always submits with `--detach` (returns the run document immediately) and
  then polls `show <run_id>` until the status is terminal, so the caller -
  not a black-box CLI wait - owns the loop and can record per-trial state.
- `candidate-plan <baseline_run_id> --target <id> (--target ...)* [--mode
  live|preview] [--name] [--hypothesis]` reuses a terminal baseline's frozen
  protocol; `--target` is required and repeatable, and the service response
  is the new run document (submitted, executing asynchronously).
- terminal run states: completed / failed / cancelled / interrupted
  (cli/sr_bench/store.py TERMINAL).

Metric extraction is a recursive key search - sr-bench payloads are nested
and versioned, and hard-coding one path would couple us to a schema that
may move. Missing keys return None; the verdict layer turns that into
INCONCLUSIVE, never into a bad number.
"""

from __future__ import annotations

import json
import subprocess
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .contract import ContractError


class SrbenchError(RuntimeError):
    """Raised when the sr-bench CLI cannot be executed or parsed."""


# Verified: the sr-bench service marks these run states terminal.
TERMINAL_RUN_STATES = frozenset({"completed", "failed", "cancelled", "interrupted"})


def cases_to_srbench_rows(
    cases: list[dict[str, Any]], benchmark: str = "recipe-lab"
) -> list[dict[str, Any]]:
    """Render recipe-lab cases as sr-bench dataset rows (X3).

    `case_id` becomes `id`, `prompt` becomes a single user message, and a
    string `truth` becomes `answer` (the row shape the real service accepts,
    verified against the R2 integration harness). Deterministic: same cases
    in the same order => byte-identical JSONL once serialized with fixed
    keys. Non-string (signal-fact) truths cannot be judged by sr-bench's
    answer matching and are rejected here, never silently dropped.
    """
    rows: list[dict[str, Any]] = []
    for index, case in enumerate(cases, 1):
        truth = case.get("truth")
        if not isinstance(truth, str) or not truth:
            raise ContractError(
                f"case {index} ({case.get('case_id')!r}) has no string truth; "
                "an sr-bench dataset needs an answer to grade against"
            )
        rows.append(
            {
                "id": str(case["case_id"]),
                "benchmark": benchmark,
                "messages": [{"role": "user", "content": str(case["prompt"])}],
                "answer": truth,
            }
        )
    return rows


@dataclass
class SrbenchCli:
    bin: str = "vllm-sr"
    url: str | None = None
    store: str | None = None
    timeout_s: float = 3600.0

    def _base_args(self) -> list[str]:
        args = [self.bin, "benchmark"]
        if self.url:
            args += ["--url", self.url]
        if self.store:
            args += ["--store", self.store]
        return args

    def run_command(self, args: list[str]) -> dict[str, Any] | str:
        """Run one benchmark subcommand; parse JSON stdout when possible."""
        cmd = self._base_args() + args
        try:
            completed = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout_s,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise SrbenchError(
                f"sr-bench CLI failed ({' '.join(args)}): {exc}"
            ) from exc
        if completed.returncode != 0:
            tail = (completed.stderr or completed.stdout or "")[-500:]
            raise SrbenchError(
                f"sr-bench CLI returned {completed.returncode} ({' '.join(args)}): {tail}"
            )
        try:
            return json.loads(completed.stdout)
        except json.JSONDecodeError:
            return completed.stdout

    def compare(
        self, baseline_run_id: str, candidate_run_id: str
    ) -> dict[str, Any] | str:
        return self.run_command(["compare", baseline_run_id, candidate_run_id])

    def show(self, run_id: str) -> dict[str, Any] | str:
        """One run document (status, manifest, ...); no evidence paging."""
        return self.run_command(["show", run_id])

    def targets(self) -> list[dict[str, Any]]:
        """The operator-registered targets (`target list` -> GET /targets).

        R2: the registry is the only acceptable source for the auto-built
        single-model baseline arm - a submitted manifest target must match
        the registry field-for-field, so the registry document is embedded
        verbatim.
        """
        payload = self.run_command(["target", "list"])
        rows = payload.get("targets") if isinstance(payload, dict) else payload
        if not isinstance(rows, list):
            raise SrbenchError("sr-bench target list did not return targets")
        return [row for row in rows if isinstance(row, dict)]

    def runs(self) -> list[dict[str, Any]]:
        """All run summaries (`runs` -> GET /runs), newest page of the store."""
        payload = self.run_command(["runs"])
        rows = payload.get("runs") if isinstance(payload, dict) else payload
        if not isinstance(rows, list):
            raise SrbenchError("sr-bench runs did not return a list")
        return [row for row in rows if isinstance(row, dict)]

    def wait_for_run(
        self, run_id: str, *, poll_interval_s: float = 2.0
    ) -> dict[str, Any]:
        """Poll `show` until the run reaches a terminal state (I).

        The caller owns the loop, so each trial's outcome is observable and
        recordable. Returns the final run document; a non-dict payload, a
        missing status, or a non-terminal status that stops updating is an
        SrbenchError, never a guessed state.
        """
        while True:
            doc = self.show(run_id)
            if not isinstance(doc, dict):
                raise SrbenchError(
                    f"sr-bench show {run_id} did not return a run document"
                )
            status = doc.get("status")
            if status is None:
                raise SrbenchError(f"sr-bench run {run_id} carries no status")
            if status in TERMINAL_RUN_STATES:
                return doc
            time.sleep(poll_interval_s)

    def _submit_and_wait(self, payload: Any, what: str) -> dict[str, Any]:
        run_id = payload.get("id") if isinstance(payload, dict) else None
        if not run_id or not isinstance(run_id, str):
            raise SrbenchError(
                f"sr-bench did not return a run id for {what}: {payload!r}"
            )
        return self.wait_for_run(run_id)

    def run_manifest(self, manifest_path: str | Path) -> dict[str, Any]:
        """Submit one frozen manifest (`run --manifest --detach`) and wait."""
        payload = self.run_command(
            ["run", "--manifest", str(manifest_path), "--detach"]
        )
        return self._submit_and_wait(payload, f"run --manifest {manifest_path}")

    def preview(self, manifest_path: str | Path) -> dict[str, Any]:
        """Submit one preview manifest (routing only, no quality) and wait."""
        payload = self.run_command(
            ["preview", "--manifest", str(manifest_path), "--detach"]
        )
        return self._submit_and_wait(payload, f"preview --manifest {manifest_path}")

    def candidate_plan(
        self,
        baseline_run_id: str,
        *,
        targets: Iterable[str],
        mode: str = "live",
        name: str | None = None,
        hypothesis: str = "",
    ) -> dict[str, Any]:
        """Reuse a terminal baseline's frozen protocol for a new run (I).

        `targets` are registered MoM target ids (the CLI requires at least
        one `--target`); the response is the submitted run document, waited
        to a terminal state here.
        """
        target_list = list(targets)
        if not target_list:
            raise SrbenchError("candidate-plan requires at least one target id")
        args = ["candidate-plan", baseline_run_id]
        for target in target_list:
            args += ["--target", target]
        args += ["--mode", mode]
        if name:
            args += ["--name", name]
        if hypothesis:
            args += ["--hypothesis", hypothesis]
        payload = self.run_command(args)
        return self._submit_and_wait(payload, f"candidate-plan {baseline_run_id}")

    def experiment(self, args: list[str]) -> dict[str, Any] | str:
        """Group baselines/routing-checks/candidates (create, attach, show...)."""
        return self.run_command(["experiment", *args])

    def report(self, run_id: str, output_path: str | Path) -> dict[str, Any] | str:
        self.run_command(["report", run_id, "--output", str(output_path)])
        path = Path(output_path)
        if not path.is_file():
            raise SrbenchError(f"sr-bench report did not write {path}")
        text = path.read_text(encoding="utf-8")
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return text


def extract_metric(payload: Any, metric_key: str) -> float | None:
    """Depth-first search for `metric_key` with a numeric leaf value."""
    if isinstance(payload, dict):
        if metric_key in payload and isinstance(payload[metric_key], (int, float)):
            return float(payload[metric_key])
        for value in payload.values():
            found = extract_metric(value, metric_key)
            if found is not None:
                return found
    elif isinstance(payload, list):
        for item in payload:
            found = extract_metric(item, metric_key)
            if found is not None:
                return found
    return None


def require_metric(payload: Any, metric_key: str) -> float:
    value = extract_metric(payload, metric_key)
    if value is None:
        raise ContractError(f"metric {metric_key!r} not found in sr-bench payload")
    return value


def extract_metric_values(payload: Any, metric_key: str) -> list[float]:
    """All numeric occurrences of `metric_key`, in traversal order."""
    values: list[float] = []
    if isinstance(payload, dict):
        if metric_key in payload and isinstance(payload[metric_key], (int, float)):
            values.append(float(payload[metric_key]))
        for value in payload.values():
            values.extend(extract_metric_values(value, metric_key))
    elif isinstance(payload, list):
        for item in payload:
            values.extend(extract_metric_values(item, metric_key))
    return values
