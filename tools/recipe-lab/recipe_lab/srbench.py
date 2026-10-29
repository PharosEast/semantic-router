"""sr-bench CLI adapter (Sub-issue 3).

Thin, honest wrapper around `vllm-sr benchmark`: recipe-lab owns the loop,
sr-bench owns evaluation - this module only shells out, parses, and reads.

- compare: JSON when the CLI emits it; otherwise the raw text is returned
  under {"raw": ...} so callers can record what actually happened instead
  of guessing.
- report: written to a file via --output, then read back.

Metric extraction is a recursive key search - sr-bench payloads are nested
and versioned, and hard-coding one path would couple us to a schema that
may move. Missing keys return None; the verdict layer turns that into
INCONCLUSIVE, never into a bad number.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .contract import ContractError


class SrbenchError(RuntimeError):
    """Raised when the sr-bench CLI cannot be executed or parsed."""


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
            raise SrbenchError(f"sr-bench CLI failed ({' '.join(args)}): {exc}") from exc
        if completed.returncode != 0:
            tail = (completed.stderr or completed.stdout or "")[-500:]
            raise SrbenchError(
                f"sr-bench CLI returned {completed.returncode} ({' '.join(args)}): {tail}"
            )
        try:
            return json.loads(completed.stdout)
        except json.JSONDecodeError:
            return completed.stdout

    def compare(self, baseline_run_id: str, candidate_run_id: str) -> dict[str, Any] | str:
        return self.run_command(["compare", baseline_run_id, candidate_run_id])

    def preview(self, run_id: str) -> dict[str, Any] | str:
        """Routing-distribution diagnosis for a run (no quality numbers)."""
        return self.run_command(["preview", run_id])

    def candidate_plan(self, baseline_run_id: str) -> dict[str, Any] | str:
        """Reuse a terminal baseline's frozen protocol for a candidate run."""
        return self.run_command(["candidate-plan", baseline_run_id])

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
