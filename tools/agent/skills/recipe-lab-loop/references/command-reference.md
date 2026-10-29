# recipe-lab command reference

All commands print a single structured JSON document on stdout
(allow_nan=False, UTF-8); progress and logs go to stderr; no command reads
stdin. Exit codes: `0` success (for `live-verdict`: PASS), `1` FAIL verdict
(`live-verdict` only), `2` usage, contract or tool error, and INCONCLUSIVE
for `live-verdict`.

## Dataset identity and splits (stage 2)

- `recipe-lab hash --file F`
  Frozen dataset identity: `sha256:` + hash of the raw bytes (A4). The spec's
  workload records this hash; every command that touches the dataset re-hashes
  and refuses drift (exit 2).
- `recipe-lab split --cases C --output-dir D [--group-key source_id
  --ratios 0.4,0.3,0.3 --seed 42]`
  Deterministic group-level three-way split (B): `calib.jsonl`, `dev.jsonl`,
  `frozen_test.jsonl`. Same input + same flags -> byte-identical output.
  `frozen_test` is quarantined: only `live-verdict --final` may cite it.

## Evidence intake

- `recipe-lab attribute --spec S [--split calib|dev] --records R
  [--cases C] --output-dir D`
  E1/E2/E3 misroute attribution; writes `attribution.jsonl` + `report.md`.
  The split manifest is re-hashed (A4); optional `--cases` verifies the
  attribution input against the split identity. `frozen_test` is refused.
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
- `recipe-lab propose-rule-thresholds --base-config C --signal S
  [--from V] --values V1,V2 --output-dir D [--decisions D1,D2
  --allow-partial --op gte --lo L --hi H --list-sites]`
  Moves every rule-level predicate site of one signal to each target value,
  including sites under NOT branches (C; the BUG-3 golden case moves four
  `difficulty gte 2.5` sites to 1.75 in one shot). Each candidate config
  ships a `<stem>.candidate.json` sidecar with kind/ref/change_summary/
  provenance/sites that `run` records in the ledger.
- `recipe-lab run --tier replay --spec S [--split calib|dev] --cases C
  --bank B --candidates D [--baseline-config B] --proposer P --ledger L
  --output-dir D --router-bin BIN --mock-backend M [--mock-extra-args "..."]
  [--validate [--validate-bin vllm-sr]] [--insight-refs "..."] [--round N]`
  One router lifecycle per candidate; RANKED records land in the ledger.
  The split identity is re-hashed before any stack work (A4); the trial
  deadline is `budget.max_minutes_per_trial` (A5, timed-out candidates get
  `status=timeout` and never rank). `--validate` pre-flights each candidate
  with `<bin> config validate`; rejected candidates never reach the ledger,
  all-rejected exits 2. `--candidates` accepts a directory, a comma-separated
  list, or a single config path — the agent's own proposal entry: pass one
  config plus `--proposer agent:<id>` so the ledger records who proposed it.

## Insight references (stage 2)

`--insight-refs` (comma-separated, on `run` and `ledger append`) merges
provenance links into `candidate.provenance.insight_refs`; unverifiable
references exit 2:
- `rec-<id>` - an existing ledger record_id (G1: every record carries one)
- `bench:<run_id>` - a benchmark run id (format-checked)
- `attr:<path>@sha256:<hex>` - attribution artifact, hash-verified
- `diag:<path>@sha256:<hex>#<anchor>` - diagnostic artifact with an anchor

## Live tier

- `recipe-lab live-verdict --spec S [--split dev|calib|frozen_test] --final?
  --baseline-run-id A --candidate-run-id B [--guards G] [--candidate J]
  [--ledger L] [--srbench-bin vllm-sr --srbench-url U --srbench-store P]`
  Compares two sr-bench runs, applies guard floors, records the three-state
  verdict with gates and bench_refs. `frozen_test` requires `--final`
  (the promotion verdict is its only consumer, B).

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
