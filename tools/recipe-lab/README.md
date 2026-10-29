# recipe-lab

A local experimentation harness for routing recipes. This slice covers
[Sub-issue 1](https://github.com/vllm-project/semantic-router/issues/5489)
(misroute attribution) plus the experiment-contract preview: the replay-tier
loop, the answer bank, proposers and the JSONL ledger arrive with Sub-issue 2;
the live tier with Sub-issue 3.

> The design doc lives in a private repo (`routing-rsi-design.md`); this README
> only mirrors what the code in this directory actually does.

## Layout

```text
tools/recipe-lab/
├── recipe_lab/
│   ├── __init__.py       # version
│   ├── cli.py            # single entrypoint: plan / attribute / collect-replays
│   ├── contract.py       # ExperimentSpec + OraclePolicy (Sub-issue 1 contract)
│   ├── attribution.py    # error taxonomy + report renderer
│   ├── adapters/
│   │   └── router_replay.py  # Router Replay management-API adapter
│   └── proposers/
│       └── threshold_search.py  # preview only; wired into the loop later
└── tests/                # 32 unit tests, no external services
```

## Commands

```bash
# contract preview (validates ExperimentSpec YAML)
PYTHONPATH=tools/recipe-lab python -m recipe_lab.cli plan --spec examples/experiment.example.yaml

# misroute attribution over a labeled frozen dataset
PYTHONPATH=tools/recipe-lab python -m recipe_lab.cli attribute \
  --records attribution-records.jsonl --output-dir out/

# pull replay records from the router and join them with the labeled cases
PYTHONPATH=tools/recipe-lab python -m recipe_lab.cli collect-replays \
  --url http://127.0.0.1:8080 --cases cases.jsonl --output records.jsonl

# same, but skip the labeled join: one raw diagnostic row per record
PYTHONPATH=tools/recipe-lab python -m recipe_lab.cli collect-replays \
  --url http://127.0.0.1:8080 --unlabeled --output rows.jsonl
```

`collect-replays` fetches the full replay list (the management API pages its
responses with default limit 20; the adapter follows `next_offset` until the
list is drained) and writes attribution input records, one JSON object per
line. `--cases` and `--unlabeled` are mutually exclusive; exactly one is
required.

## Attribution record format

```json
{"case_id": "c-001", "truth": "local_standard", "matched_decision": "cloud_frontier",
 "routed": "local/private-qwen", "family": "d3", "answer_ok": null,
 "request_id": "req-17",
 "signal_values": {"decision:difficulty": 1.75, "decision:domain:code": 0.97},
 "signal_confidences": {"decision:domain": 0.97}, "decision_tier": 2}
```

- `matched_decision` is the decision the router actually matched (F1 renamed
  the earlier `pred`; reading old records with a top-level `pred` field still
  works).
- `truth` is the hand-labeled expected decision for that prompt.
- `answer_ok` is `null` here; correctness is joined later from saved answers
  (Sub-issue 2's answer bank). Evidence-first: `null` is never counted as an
  error.
- Signal evidence (`signal_values` / `signal_confidences` / `decision_tier`)
  is carried verbatim from the replay record. Keys are `<type>:<name>` (e.g.
  `decision:difficulty`); choice-type decision signals store one probability
  per choice under `decision:<name>:<choice>` (e.g. `decision:domain:code`),
  while score-type signals keep the numeric score under `decision:<name>`.

## Error taxonomy

| class | meaning |
|---|---|
| `correct` | routed as intended and answered correctly |
| `decision_mismatch` | matched decision != expected decision, judged without signal-level truth |
| `E1_signal` | signal label != ground truth; needs signal-level truth, which arrives with the multi-signal oracle (F2) — defined but not produced yet |
| `E2_policy` | decision right, routed model != oracle-policy expectation, answer wrong |
| `E3_capability` | routing exactly as intended, answer still wrong |
| `suboptimal_routing` | policy deviation but still a correct answer (counted, never an error) |
| `no_evidence` | answer correctness unknown; never an error |
| `policy_gap` | truth label missing from the declared oracle policy |

`attribute` writes `out/attribution.jsonl` (one row per record with its class)
and `out/report.md` (per-family hints, `decision_mismatch` confusion pairs,
`E2` routing deviations).

## Tests

```bash
PYTHONPATH=tools/recipe-lab python -m pytest -q tools/recipe-lab/tests
```

Unit tests only, no external services; management-API calls are mocked
(including pagination).
