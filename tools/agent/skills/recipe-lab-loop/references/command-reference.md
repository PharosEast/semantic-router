# recipe-lab command reference

All commands print a single structured JSON document on stdout
(allow_nan=False, UTF-8); progress and logs go to stderr; no command reads
stdin. Exit codes: `0` success (for `live-verdict`: PASS), `1` FAIL verdict
(`live-verdict` only), `2` usage, contract or tool error, and INCONCLUSIVE
for `live-verdict`.

## Evidence intake

- `recipe-lab attribute --spec S --records R --output-dir D`
  E1/E2/E3 misroute attribution; writes `attribution.jsonl` + `report.md`.
- `recipe-lab collect-replays --url http://127.0.0.1:8080 --cases C
  --output R --family d3`
  Pulls router_replay records from the management API and joins them with
  labeled cases. Skipped rows are counted (`no_prompt`/`unlabeled`/
  `duplicate_prompt`), never guessed.
- `recipe-lab bank build --cases C --models M1,M2 --endpoint U --output B`
  Direct per-case/per-model queries graded by `expected_contains`.
- `recipe-lab bank apply --bank B --records R [--output R2]`
  Joins correctness by (case_id, routed model); missing entries stay
  `no_evidence`.

## Proposal and replay tier

- `recipe-lab propose-thresholds --base-config C --signal S --deltas=-0.1,0.1
  --output-dir D`
  Deterministic single-variable threshold candidates (baseline dropped,
  clamped to [--lo, --hi]).
- `recipe-lab run --tier replay --spec S --cases C --bank B --candidates D
  [--baseline-config B] --proposer P --ledger L --output-dir D
  --router-bin BIN --mock-backend M [--mock-extra-args "..."]`
  One router lifecycle per candidate; RANKED records land in the ledger.
  `--candidates` accepts a directory, a comma-separated list, or a single
  config path — the agent's own proposal entry: pass one config plus
  `--proposer agent:<id>` so the ledger records who proposed it.

## Live tier

- `recipe-lab live-verdict --spec S --baseline-run-id A --candidate-run-id B
  [--guards G] [--candidate J] [--ledger L] [--srbench-bin vllm-sr
  --srbench-url U --srbench-store P]`
  Compares two sr-bench runs, applies guard floors, records the three-state
  verdict with gates and bench_refs.

## Reporting

- `recipe-lab ledger show|append|top`
- `recipe-lab report --ledger L --output R` - ledger-only replay with a
  consistency check.
- `recipe-lab next --ledger L --output C` - agent context (untrusted header).
- `recipe-lab provenance --ledger L --output M` - recipe-lab-run-v1 manifest.
- `recipe-lab record-draft --ledger L --output D` - promotion draft from the
  latest PASS live record (human checklist included).

## Notes

- Replay tier needs no LLM backend: the router and signal model run on CPU.
- A store with prior sr-bench execution evidence refuses autostart; start it
  explicitly with `SR_BENCH_STORE=<path> vllm-sr benchmark serve`.
- The router's benchmark binding supports `X-SR-Bench-Max-Inference-Calls: 1`
  only (`pkg/extproc/benchmark_config_binding.go`).
