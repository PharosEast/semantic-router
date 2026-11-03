# recipe-lab command reference

All commands print a single structured JSON document on stdout
(allow_nan=False, UTF-8); progress and logs go to stderr; no command reads
stdin. Exit codes: `0` success, `1` FAIL verdict (`live-verdict` and
`verify` follow the verdict), `2` usage, contract or tool error (also
INCONCLUSIVE for `live-verdict`).

## Contract

- `recipe-lab plan --spec S`
  Load and validate the experiment contract and print it normalized; no
  experiment runs. Contract errors exit 2 with the reason on stderr — fix
  the spec and re-run. Every later command re-checks the same contract, so
  a spec that passes this command is the single source of truth for a round.
- `attribution.oracle_policy` (F2) takes exactly one of two forms: a
  label -> model mapping, or a rules/default mapping
  `{"rules": [{"when": {<signal>: <condition>}, "expect": {"decision":
  <decision>, "models": [<model>, ...]?}}, ...],
  "default": {"decision": ...}?}`. A condition matches one signal fact by
  label string, finite number, or a numeric predicate mapping over
  `gte|lte|gt|lt` (router semantics: `gte` fails when the value is below
  the bound). The first matching rule wins, the optional default covers
  the rest, and truth no rule covers is a policy gap.
- Optional top-level `cost_model` (E2): map of served model name to a
  finite positive relative cost. Its presence lets the replay tier compute
  `routed_cost_index` and proxy cost constraints; without it, cost is
  adjudicated only in the live tier. C6: a `cost_basis` with
  `source: provider_pricing` and declared `assumed_pricing` rates derives
  the replay cost model automatically (per-model prompt+completion rate sum,
  normalized against the cheapest positive sum; a free model stays at 0), so
  the basis alone equips replay - declaring `cost_model` alongside such a
  basis is a contract error. A basis without rates derives nothing (live-tier
  pricing only), and `source: cost_model` requires the top-level section.

## Dataset identity and splits (stage 2)

- `recipe-lab hash --file F`
  Frozen dataset identity: `sha256:` + hash of the raw bytes (A4). The spec's
  workload records this hash; every command that touches the dataset re-hashes
  and refuses drift (exit 2).
- `recipe-lab split --cases C --output-dir D [--group-key source_id
  --ratios 0.4,0.3,0.3 --seed 42]`
  Deterministic group-level three-way split (B): `calib.jsonl`, `dev.jsonl`,
  `frozen_test.jsonl`. Same input + same flags -> byte-identical output.
  `frozen_test` is quarantined: only `live-verdict --final` (or `verify
  --final`) may cite it.

## Evidence intake

- `recipe-lab attribute --spec S [--split calib|dev] --records R
  [--cases C] --output-dir D`
  E1/E2/E3 misroute attribution; writes `attribution.jsonl`, `summary.json`
  (the summarize() result, read by `next`, G3) and `report.md`.
  The split manifest is re-hashed (A4); optional `--cases` verifies the
  attribution input against the split identity. `frozen_test` is refused.
- `recipe-lab attribute-session --records R --output-dir D`
  Session-aware agentic attribution (P2): aggregates replay rows that carry
  session evidence (conversation_id / turn_index) into conversations and
  layers three evidence classes on top of single-turn E1/E2/E3 (never
  redefining them): `lane_flap` (adjacent requests of one conversation chose
  different models), `selection_failed` (recorded failed selections) and
  `tool_loop_interrupted` (a rejected selection whose turn resumed
  afterwards; needs the rejection-recording router fix). Writes
  `session-report.md` and `session-events.json`; deterministic - same rows,
  same report bytes.
- `recipe-lab collect-replays --url http://127.0.0.1:8080
  [--cases C | --unlabeled] --output R --family d3 [--api-key-env NAME]`
  Pulls router_replay records from the management API and joins them with
  labeled cases. Skipped rows are counted (`no_prompt`/`unlabeled`/
  `duplicate_prompt`), never guessed. `--unlabeled` skips the labeled join
  and emits one raw diagnostic row per replay record - the `diagnose`
  input; mutually exclusive with `--cases`. `--api-key-env` names the
  environment variable holding the management API bearer token.
- `recipe-lab bank build --cases C --models M1,M2 --endpoint U --output B`
  Direct per-case/per-model queries. Each case declares its grader (I):
  the default `contains` grades a case-insensitive substring match on
  `expected_contains`; `grader: "mcq"` grades the answer letter extracted
  by the router repo's `mcq_judge.extract_letter` against `expected`
  (`src/` must be on `PYTHONPATH`; an unextractable letter is wrong, never
  unevaluated - the sr-bench no-answer convention). Any other grader name
  is a contract error.
- `recipe-lab bank import --spec S --profile-dir P [--split dev] --output B`
  Cold-start answer bank (C6): converts each spec model's direct-model
  `judged.jsonl` from the profile run (`P/<split>/results/<dirname>/`) into
  bank entries - no endpoint traffic, the C3 evidence *is* the bank. Rows
  map task_id -> case_id, is_correct -> answer_ok, completion -> answer,
  completion_tokens -> tokens (carried through `apply` for per-question
  cost work). Duplicate task_ids, missing task_id, or a non-boolean
  is_correct are contract errors; every spec model must have a judged file
  (exit 2 listing the missing one), and the output is written only after
  all models convert.
- `recipe-lab bank apply --bank B --records R [--output R2]`
  Joins correctness by (case_id, routed model); missing entries stay
  `no_evidence`.
- `recipe-lab observe-labels --spec S --split calib --out D
  --router-bin B --mock-backend M [--workload W] [--request-timeout S]`
  Cold-start observation (C2): writes the seed config (14 domain decisions +
  an `unmatched` catch-all, the decision model pinned, response cache
  disabled), boots the router against the mock backend and sends each split
  case tagged with `x-request-id: rl-observe-<case_id>`, then joins the
  router_replay records by request id. Each case gets `router_label` = the
  matched decision's domain (the `unmatched` catch-all records `other`) and
  `confidence` when the record carries one. Writes `seed-config.yaml`,
  `labels-<split>.jsonl` (sorted by case_id) and
  `labels-<split>-summary.json`; its `label_coverage` (non-other labels over
  all cases; missing records count as uncovered) is the number the
  provenance records. `frozen_test` is refused (exit 2).
- `recipe-lab observe-lanes --spec S --split calib --out D
  --router-bin B --mock-backend M [--workload W] [--vllm-sr-bin V]
  [--catalog-version L] [--model-name N] [--request-timeout S]`
  Cold-start observation, template path (C2, sub-issue-8 §2.2). Requires
  the spec's `template:` section (built-in bundle + recipe). Reads the
  template's lanes via `vllm-sr recipe builtin list`, generates the
  observation config via `vllm-sr recipe builtin init` (every lane bound to
  placeholder models on the mock backend; the bundle's pinned decision
  model decides the lane, so no Y2 label-agreement check is needed), then
  probes each split case with `x-request-id: rl-observe-lanes-<case_id>`
  and joins the replay records by request id. Writes
  `providers-observe.yaml`, `bindings-observe.yaml`, `observe-config.yaml`
  (response cache forced off), `lanes-<split>.jsonl`
  (case_id/lane/confidence) and `lanes-<split>-summary.json` (template
  hash, init command, lane_coverage, lane_counts). `frozen_test` is
  refused (exit 2).
- `recipe-lab profile --spec S --labels D --splits calib,dev --out P
  --model-eval-dir M [--workload W] [--python-bin B] [--concurrency N]
  [--request-timeout S] [--force]`
  Model profiles (C3), legacy label mode. For every spec `models:` entry,
  per split: rewrites the split tasks so `category` is the router label
  from `labels-<split>.jsonl` (labels under `profile.min_label_cases`,
  default 20, merge into `other` with the merge recorded; uncovered cases
  join `other`), runs `multi_benchmark_eval.py` direct against the model's
  endpoint (resumes completed records unless `--force`), then
  `mcq_judge.py --analysis`; one `cost_aware_calib.py --token-costs-only`
  invocation per split writes `token_costs/` for bootstrap. The scripts
  are invoked, never copied: the profile records script path, content
  sha256, git blob hash and every command line. Truncation
  (`finish_reason: length`) and no-answer rates beyond `evidence_quality`
  ceilings fail with exit 2 and a generation-params hint. Writes
  `generation.json` (the shared `generation:` params, also used by the
  live arms), per-split `tasks.jsonl` / `results/<alias>_direct/` /
  `token_costs/` and a byte-stable `profile.json` (recipe-lab-profile-v1:
  per model x label n, accuracy with Wilson CI, tokens, assumed-pricing
  cost, p50/p95 latency, quality rates, all input hashes). Aliases must
  survive the result_to_config dirname round-trip (one `/` or no `_`,
  never `auto`). `calib` outputs feed bootstrap (C4/C5), `dev` outputs the
  answer bank (C6); `frozen_test` is refused (exit 2).
- `recipe-lab profile --spec S --lanes D --splits calib,dev --out P ...`
  Model profiles, template lane mode (C3, sub-issue-8 §2.2 step 4):
  identical chain, but `category` is the template lane from
  `lanes-<split>.jsonl`. Rare lanes are never merged: lanes under
  `bootstrap.min_lane_cases` (default 20) are recorded in the profile as
  `unmeasured_lanes`, and cases without an observed lane get the
  `unobserved` category. Requires the spec's `template:` section.
- `recipe-lab bindings --spec S --profile P --out D [--lambdas ...]
  [--exclude-decision LANE ...] [--vllm-sr-bin V] [--catalog-version L]
  [--model-name N]`
  Cold-start candidates, template path (C4+C5, sub-issue-8 §2.2 steps
  5-7). First the routing-space check: "no benefit to routing" ends the
  command with a report and no candidates (no measured lane shows the
  `min_advantage_pp` advantage of a best model over the cheapest, or every
  measured lane prefers the same model and the cheapest is never within
  the threshold). Otherwise, per lambda: capability filter (only models
  whose declared `capabilities` satisfy the lane requirement - vision
  needs vision, agentic needs tools, long_context needs a 200K context
  window), rank measured lanes by `accuracy - lambda * normalized cost`
  (per-lane max normalization, the cost_aware_calib convention), take the
  template's `minimum_candidates`; unmeasured lanes rank by the
  lambda-adjusted overall score and are marked "default rule, untested";
  a lane no model can satisfy must be excluded via `--exclude-decision`
  (passed to `recipe builtin init`, recorded in the report). Each
  candidate is generated by `vllm-sr recipe builtin init` over the spec's
  `providers_config` (endpoints/credentials come only from there; the
  derived eval copy disables the response cache) and must pass
  `vllm-sr config validate`. Writes `providers-eval.yaml`,
  `bindings-lambda-<tag>.yaml`, `candidate-lambda-<tag>.yaml` and
  `bindings-lambda-<tag>.provenance.json` (template hash, per-lane basis
  with scores/n/measured, unmeasured and excluded lanes, init command).
- `recipe-lab bootstrap --legacy-domain-template --spec S --profile P
  --model-eval-dir M --out D [--lambdas 0,0.1,0.2,0.3] [--python-bin B]
  [--validate-bin V] [--listener-port N]`
  Cold-start candidates, LEGACY domain-template path (C4+C5 legacy,
  sub-issue-8 §2.5; requires `--legacy-domain-template` and is not part of
  the cold-start completion criteria). Consumes the C3 profile's calib
  split only (anti-leak). Per lambda: runs the real `result_to_config.py`
  (`--mode full`) over `P/calib/results` + `P/calib/token_costs` into
  `scaffold-lambda-<lambda>.yaml`; the script is invoked, never copied,
  and the sidecar records its path, sha256, git blob hash and command
  line. If the script prints the §3.2-5 `Warning: --cost-lambda` marker
  (token costs missing -> lambda silently treated as 0) the command fails
  with exit 2, and the scaffold must cover every profile calib label and
  every spec model. Completion (C5) turns each scaffold into a runnable
  canonical config: gateway listener on `--listener-port` (default 8899),
  each alias's `backend_refs` pointed at its contract endpoint with
  `pricing` from `cost_basis` (unpriced models get none), decisions
  grouping the labels that share the cost-aware `model_scores` first
  place into one `route-<model>` decision (OR-joined `type: domain`
  conditions, modelRefs in that domain's score order, router_replay
  plugin mounted for replay capture), priorities following calib traffic
  share, a `route-fallback` catch-all, `providers.defaults.model` = best
  mean over the lambda-adjusted domain scores, and the scaffold's sparse
  global merged with the C2 seed model catalog. The response cache and
  tools are DISABLED (repeat hits would poison replay/live evidence; the
  reasons sit in the sidecar for the bundle write-up - re-enable for
  production if wanted). Every candidate must pass
  `<validate-bin> config validate --config` before it is emitted; a
  rejected candidate fails with exit 2. Writes
  `candidate-lambda-<lambda>.yaml` plus
  `candidate-lambda-<lambda>.provenance.json` (proposer `bootstrap`,
  lambda, per-label selected model / accuracy / cost / traffic share,
  scaffold + profile + candidate hashes, decisions layout). Candidates
  are byte-stable on rerun; feed them to `run --split dev` for ranking.

## Diagnosis and distribution calibration (D)

Both commands need no ground truth: they turn unlabeled traffic into
proposals.

- `recipe-lab diagnose --records R --config C --output-dir D
  [--expect-rare d1,d2] [--rare-pct P] [--min-sample N]
  [--utilization-warn U] [--source "..."] [--agent-requests M]`
  Three checks over replay rows: decisions that never fire (dead) or fire
  under `--rare-pct` traffic share (rare; `--expect-rare` whitelists
  legitimately rare decisions such as security containment - reported,
  never warned), per-signal quantiles / histogram / range utilization
  (`--utilization-warn` flags a signal using less of its known range;
  `--min-sample` warns when the sample count is too small), and rule
  predicate sites no traffic ever satisfies (unreachable). `--records` is
  the `collect-replays --unlabeled` output. Z9: `--agent-requests` is the
  probe's `agent-requests-<tag>.json` manifest - the client-recorded
  sub-agent request ids join the protection traces' identity hashes so
  sub-agent retention is judged without router changes (unjoinable or
  empty manifests stay unverifiable). Writes `diagnosis.json`
  (recipe-lab-diagnosis-v1, input-hashed, byte-stable - the
  `next --diagnosis` input) and `diagnosis.md`.
- `recipe-lab propose-calibrated --diagnosis D --base-config C --signal S
  [--from V] [--quantiles Q1,Q2 | --target-share F] [--op gte|gt|lte|lt]
  [--lo L] [--hi H] [--output-dir D]`
  Rule threshold candidates at the signal's observed distribution
  quantiles, or at a target traffic share (quantile `1 - share`). Uses the
  rule-level proposer, so all rule sites sharing (signal, op, value) move
  together; each candidate sidecar cites the diagnosis it came from, and a
  diagnosis whose config hash does not match `--base-config` is rejected
  (exit 2). `--from` restricts the move to sites currently at that value
  (the 2.5 -> 1.75 BUG-3 calibration).

## Agentic cold start (AG5)

- `recipe-lab profile-agentic --reports R1.json --reports R2.json ... --output P`
  Aggregates per-model single-target sr-bench run reports (the AG1
  protocol: one run per model, one arm each) into the agentic profile
  (`recipe-lab-agentic-profile-v1`): task success rate with its interval,
  per-task actual and cache-neutral cost, calls per task, latency and the
  truncation rate. Fields a report does not carry stay None and are listed
  in the model entry's `missing` - missing evidence is never counted as
  zero.
- `recipe-lab bootstrap-agentic --spec S --profile P --template T --client
  claude-code|openai-chat|generic --out D [--medium-tolerance-pp P]
  [--local-model M] [--mcq-order M1,M2] [--validate [--validate-bin vllm-sr]]`
  Rewrites the agent recipe template from the profile (lane assignment:
  complex lanes take the strongest model, medium lanes the cheapest within
  `medium_tolerance_pp` of the best, the simple lane and fallback the
  cheapest, privacy/security the spec's `local_models` with adaptation
  bypassed, legal-health and STEM the MCQ picture when declared). Session
  protection is written per client: claude-code protects the whole session
  keyed by the verified headers (x-claude-code-session-id, agent:
  x-claude-code-agent-id, from AG0-W2); other clients protect one
  conversation keyed by the neutral headers. Adaptation is disabled, the
  progress gate starts in observe mode, and candidate requirements demand
  declared capabilities. Writes `agentic-config.yaml` and a sidecar
  (`agentic-config.bootstrap.json`) carrying every lane's rule and
  evidence. `--validate` refuses an unvalidatable candidate (exit 2).

## Proposal and replay tier

- `recipe-lab propose-session-policy --config C [--diagnosis D]
  [--replay-summary S] --output-dir D`
  AG6: diagnosis-triggered one-variable session-protection candidates.
  Evidence asks, the proposer proposes - no evidence, no candidate:
  tool-loop switches raise `stability_weight` (to 2.0), model flips raise
  `switch_margin` (0.05 -> 0.1), a `not_separate` sub-agent separation adds
  the agent identity header (x-claude-code-agent-id), rejected selections
  raise `min_turns_before_switch`. Each candidate changes exactly one
  variable and carries a sidecar naming its trigger; run the candidates
  through the route-replay gates (`run --tier route-replay`) before
  anything reaches the live tier.
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
- `recipe-lab run --spec S [--split calib|dev] --cases C
  --bank B --candidates D [--baseline-config B] --proposer P --ledger L
  --output-dir D --router-bin BIN --mock-backend M [--mock-extra-args "..."]
  [--validate [--validate-bin vllm-sr]] [--attribution-ref REF]
  [--insight-refs "..."] [--round N]
  [--reload restart|hot [--api-key-env NAME]]`
  One router lifecycle per candidate; RANKED records land in the ledger.
  The split identity is re-hashed before any stack work (A4); the trial
  deadline is `budget.max_minutes_per_trial` (A5, timed-out candidates get
  `status=timeout` and never rank). `--validate` pre-flights each candidate
  with `<bin> config validate`; rejected candidates never reach the ledger,
  all-rejected exits 2. `--candidates` accepts a directory, a comma-separated
  list, or a single config path — the agent's own proposal entry: pass one
  config plus `--proposer agent:<id>` so the ledger records who proposed it.
  - `--reload` (H): `restart` (default) restarts the router once per
    candidate; `hot` keeps one router alive for the whole candidate list
    and feeds it each candidate via `PUT /api/v1/config` - needs
    `--api-key-env` for the management bearer token (the same env var
    covers the replay fetch). A hot-path failure falls back to restart;
    the record notes always say which mode ran (`reload=`).
  - Replay cost proxy (E2): with `cost_model` in the spec, every candidate
    also reports `routed_cost_index` - the mean relative cost of the routed
    models over answered rows (a routed model missing from `cost_model`
    makes the index incomputable; unknown cost is never counted as zero).
    Declared constraints are proxied onto that measurement: the evidence
    keeps the declared live metric name and adds
    `proxy: routed_cost_index`, and `bound: baseline` resolves against the
    baseline arm's proxy. A violated candidate carries `constraint_warning`
    in its ledger record and is excluded from `best_replay` selection.
    Without `cost_model` replay does no constraint work - the `next`
    context states that cost is adjudicated only in the live tier.

## Insight references (stage 2)

`--insight-refs` (comma-separated, on `run`, `ledger append`,
`live-verdict` and `verify`) merges provenance links into
`candidate.provenance.insight_refs`; unverifiable references exit 2:

- `rec-<id>` - an existing ledger record_id (G1: every record carries one)
- `bench:<run_id>` - a benchmark run id (format-checked)
- `attr:<path>@sha256:<hex>` - attribution artifact, hash-verified
- `diag:<path>@sha256:<hex>#<anchor>` - diagnostic artifact with an anchor

## Live tier

- `recipe-lab verify --spec S [--split calib|dev|frozen_test] [--final]
  [--workload W] (--baseline-run-id A | --baseline-manifest M)
  --candidate J [--candidate-targets T1,T2] [--repetitions N] [--work-dir D]
  [--guard-measurements M] [--insight-refs R] [--attribution-ref REF]
  [--ledger L] [--run-cache P] [--srbench-bin vllm-sr] [--srbench-url U]
  [--srbench-store P] [--timeout S] [--round N]`
  Drives sr-bench end to end instead of comparing two ready-made runs (I):
  extracts the baseline manifest from the terminal-completed baseline run
  (`benchmark show`; `--baseline-manifest` submits one instead), runs one
  preview candidate-plan, then `--repetitions` AB/BA pairs - every pair
  re-runs the frozen baseline manifest and a fresh candidate-plan, then
  compares the pair. Pair metrics become medians/ranges; any incomplete
  pair forces INCONCLUSIVE. The preview is diagnostic only: its routing
  distribution lands in the record notes, and a preview failure is
  recorded as a diagnostic reason, never a veto. Prints the same JSON
  shape as `live-verdict` and follows the verdict for its exit code.
  `--candidate-targets` names the sr-bench MoM targets serving the
  candidate; it falls back to the candidate JSON's `srbench_targets`.
  - Driver `model_eval` (C7): when the workload declares
    `driver: model_eval`, sr-bench is replaced end to end - every arm is a
    `multi_benchmark_eval.py` + `mcq_judge.py` pass over one rewritten task
    set: the baseline and candidate arms hit the router endpoints serving
    their configs, the S/W reference arms hit the spec models' direct
    endpoints. Per-case correctness feeds the same M2/M3/M4 machinery as
    the sr-bench driver (paired stats in the sr-bench compare shape, the
    evidence-quality ceilings, reference arms cached by tasks hash +
    alias). Accuracy is the planned denominator (missing or unanswered
    counts as wrong). A crashed arm exits 2 (the scripts resume completed
    records); an arm without judged rows is INCONCLUSIVE evidence.
    Required flags: `--tasks T` (the profile's rewritten
    `<split>/tasks.jsonl`), `--baseline-router-endpoint U`,
    `--candidate-router-endpoint U` and `--model-eval-dir M`; optional
    `--router-model vllm-sr/auto`, `--python-bin B`, `--concurrency N`,
    `--request-timeout S`, `--force`. sr-bench flags are ignored and
    `--baseline-run-id` / `--baseline-manifest` are rejected in this mode.
    Random mix R (C8): the records now carry each response's `model` (the
    actually-routed backend), so the driver derives the candidate arm's
    routing shares, mixes the reference arms' per-case accuracy and
    per-case cost by those shares, and reports the expected accuracy/cost
    plus a paired C-vs-R bootstrap interval (R is re-assigned by the
    shares on every resample). R enters the verdict input as the
    `ref:random_mix` reference role (usable as a constraint `bound`), a
    trailing `ref:random_mix` Pareto row, and the result's `random_mix`
    section (the P3 interval). Missing evidence - no `model` in the
    records, a routed model outside `spec.models`, an unmeasured needed
    reference - keeps R unevaluable with a reason, never guessed.
  - Agentic workloads (Z4/Z5): the stability hard gates judge the arms'
    own router replay records - their only evidence source. After each
    completed MoM arm run, verify pulls the run's replay records from the
    arm router's management API (session identities `<run_id>-<digest>`
    filter the router's records) and the gates consume the last completed
    pair's rows - the same pair whose continuity sections carry the switch
    evidence. Z5: the switch ceiling splits by cause (upstream #5567
    releases the tool-loop lock on decision drift, an in-design transfer):
    `max_unprotected_tool_loop_switches` (default 0) = the continuity
    tool-loop total minus the replay traces' `decision_drift` releases
    (identity-missing / bypassed / observe switches only continuity sees
    live in the residual), and `max_decision_drift_switches_per_task`
    (default 0.2) rates the releases per replay session; exceeding it
    points at threshold calibration (propose-calibrated). Required flags:
    `--candidate-management-endpoint U` (candidate arm's router), plus
    `--baseline-management-endpoint U` (baseline arm's router) while
    `max_tool_loop_interrupted` pins `baseline` (its default); a missing
    endpoint fails the CLI before anything is spent on runs. A failed
    fetch is a diagnostic - the gates stay unevaluated, never a pass.
    Single-model reference arms never pass a router and have no replays:
    a comparison without a comparable baseline router (cold start)
    declares a numeric ceiling (e.g. 0) instead of the `baseline` pin.
    The result's `agentic` block names each side's `replay_evidence`
    (run id, endpoint, rows) and the candidate arm's `switch_evidence`
    (switches by cause).
- `recipe-lab live-verdict --spec S [--split dev|calib|frozen_test] --final?
  --baseline-run-id A --candidate-run-id B [--guard-measurements M]
  [--candidate J] [--insight-refs R]
  [--ledger L] [--srbench-bin vllm-sr --srbench-url U --srbench-store P]`
  Compares two sr-bench runs, applies guard floors, records the three-state
  verdict with gates and bench_refs. `frozen_test` requires `--final`
  (the promotion verdict is its only consumer, B).
- Guards (A3): floors come only from `spec.gates`. `--guard-measurements`
  is a JSONL of measurements only - `{"name": "pii-recall-floor",
  "candidate": 0.97, "baseline": 0.96}`; a `floor` key in the file is a
  contract bypass and exits 2. `floor: baseline` in the contract resolves
  against the measured baseline value. A gate without a usable measurement
  is unevaluated (verdict INCONCLUSIVE); undeclared measurement names are
  reported but never veto.
- Constraints (E): `objective.constraints` declares economic conditions
  (`{name, metric, op gte|lte, bound: "baseline"|<number>, tolerance_pct}`);
  distinct from gates, which are safety vetoes. Verified sr-bench compare
  cost fields: `total_spend_usd`, `cache_neutral_cost_usd`, `cost_usd`,
  `evaluation_cost_usd` on the baseline_targets/candidate_targets entries.
  Unknown usage is null -> the constraint is INCONCLUSIVE (never zero).
  Constraint evidence lands in the record's `constraints` field. In the
  replay tier the same constraints are proxied onto `routed_cost_index`
  when the spec has `cost_model` (see `run`, E2).

## Reporting

- `recipe-lab ledger show|append|top`
- `recipe-lab report --ledger L --output R` - ledger-only replay with a
  consistency check. Live records render gate results, constraint evidence
  and `constraint_warning` notes; replay records with a cost proxy render
  `routed_cost_index` (E2).
- `recipe-lab next --spec S --ledger L [--diagnosis diag/diagnosis.json]
  --output C [--tail N]` - full agent context export (G3), byte-stable for
  the same inputs. Sections: untrusted-material header; contract summary
  (objective, replay_metric, constraints, gates, cost_model, split names +
  hashes, budget used/remaining); ledger tail; best replay candidate (ranked
  by `objective.replay_metric` through the same `best_replay` selection as
  `ledger top`, A2 - cost-proxy-violating candidates excluded, E2); last
  live verdict with reasons, gates, constraints and bench_refs; latest
  attribution summary read through the newest record's hash-verified
  `attribution_ref`; optional diagnosis highlights (dead/rare decisions,
  unreachable threshold sites); and a citable insight-ref list - paste
  those refs verbatim into `--insight-refs` when proposing (step 3). When
  cost constraints exist but `cost_model` does not, the summary states that
  cost is adjudicated only in the live tier.
- `recipe-lab provenance --ledger L --output M` - recipe-lab-run-v1 manifest.
- `recipe-lab record-draft --ledger L --output D` - promotion draft from the
  latest PASS live record (human checklist included).
- `recipe-lab bundle --spec S --ledger L --record rec-... --out DIR --config
  FINAL.yaml --skeleton SCAFFOLD.yaml --profile PROFILE.json` - C9 packaging
  of final-verify evidence. Only a live PASS citing `split: frozen_test` or
  a `role: transfer` workload, with every guard `pass` and a config that
  passes `<validate-bin> config validate`, is bundled; anything else exits 2
  without writing. Byte-stable artifacts: config.yaml, validate.txt,
  config.diff, report.md (verdict detail, campaign table, per-workload
  counts, evidence quality, assumptions, limitations, cache/tools note),
  ledger.jsonl (same-dataset-hash excerpt), provenance.json (hashes plus
  `--router-commit` / `--model-revision NAME=REV` / `--model-eval-script`,
  missing evidence recorded as null), PROMOTION.md (the record draft) and
  REVIEW.md (empty V5 template). All artifacts are scanned for private
  content (personal user paths, non-local IP URLs, `--private-pattern`
  regexes) before anything is written.

## Notes

- Replay tier needs no LLM backend: the router and signal model run on CPU.
- A store with prior sr-bench execution evidence refuses autostart; start it
  explicitly with `SR_BENCH_STORE=<path> vllm-sr benchmark serve`.
- The router's benchmark binding supports `X-SR-Bench-Max-Inference-Calls: 1`
  only (`pkg/extproc/benchmark_config_binding.go`).
- `verify` needs the sr-bench CLI (`--srbench-bin vllm-sr`) and a baseline
  run that reached `completed` with its manifest recoverable through
  `benchmark show`.
- The `mcq` bank grader imports `training.model_eval.mcq_judge` from the
  router repo: run recipe-lab with `src/` on `PYTHONPATH` for mcq cases.
