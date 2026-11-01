---
name: recipe-lab-loop
description: Use when improving routing accuracy through the recipe-lab evidence loop - reading misroute attribution, proposing recipe candidates, evaluating through the replay/live tiers, and recording keep/revert decisions in the ledger.
---

# Recipe-lab improvement loop

Drive the closed loop with `tools/recipe-lab/recipe_lab/cli.py` (installed as
`recipe-lab`): attribution -> proposal -> replay-tier search -> live verdict
-> ledger. sr-bench owns evaluation; this loop only orchestrates it. Read
`references/command-reference.md` for the command surface,
`references/evidence-interpretation.md` for reading outcomes, and
`references/red-lines.md` before proposing anything.

## Ground rules

- Never touch a stack you did not start; run `vllm-sr status` first.
- Credentials stay in environment variables, never in YAML, args, or logs.
- frozen-test is quarantined: proposals and agents see calib/dev only.
- A `RANKED` verdict is a ranking, not a conclusion. A `PASS` is a
  recommendation, not a promotion - promotion is human + Standard holdout.
- Change one variable per candidate; re-run the current step after it.
- INCONCLUSIVE and no-evidence are evidence problems: gather more evidence,
  never record a guess.

## Steps

1. Validate the experiment contract: `recipe-lab plan --spec spec.yaml`.
   Success: normalized spec JSON. Failure: fix the spec per the contract error.
2. Read the evidence: `recipe-lab next --spec spec.yaml --ledger ledger.jsonl
   --output context.md` (add `--diagnosis diag/diagnosis.json` when a
   diagnosis artifact exists). Read the untrusted-material header, then the
   contract summary, best replay candidate, last live verdict, attribution
   summary and the citable insight-ref list it exports (G3).
3. Propose one candidate: edit one recipe variable in a config revision and
   cite the motivating evidence with `--insight-refs`, using the exact refs
   from `next`'s citable list (`rec-<id>`, `bench:<run_id>`,
   `attr:<path>@sha256:<hex>`, `diag:<path>@sha256:<hex>#<anchor>`).
   When you authored the candidate config yourself (not a proposer output),
   evaluate it with `--proposer agent:<id>` in step 5; the id is lowercase
   `[a-z0-9._-]` (e.g. `agent:claude-loop`) and is recorded verbatim in the
   ledger provenance.
   For threshold-shaped fixes, ground the proposal in traffic first:
   `recipe-lab diagnose --records <collect-replays --unlabeled output>
   --config <recipe.yaml> --output-dir diag/` (no ground truth needed),
   then `recipe-lab propose-calibrated --diagnosis diag/diagnosis.json
   --base-config <recipe.yaml> --signal <signal> --quantiles ...`.
4. Validate the candidate: `vllm-sr config validate` on the revision.
   Failure: fix the config; never evaluate an invalid candidate.
5. Evaluate in the replay tier: `recipe-lab run --spec spec.yaml --split dev
   --cases <split.jsonl> --bank bank.jsonl --candidates <config>
   --proposer agent:<id> --ledger ledger.jsonl --router-bin ... --mock-backend
   ...` against the answer bank. Records land in the ledger.
6. For the final gate: run sr-bench preview/live for baseline and candidate,
   then `recipe-lab live-verdict --spec spec.yaml --baseline-run-id ...
   --candidate-run-id ... --guard-measurements measurements.jsonl
   --candidate cand-001.json --ledger ledger.jsonl` (add `--split frozen_test
   --final` only for the promotion verdict). Or drive the whole gate in one
   command: `recipe-lab verify --spec spec.yaml --baseline-run-id ...
   --candidate cand-001.json --ledger ledger.jsonl` runs the preview, the
   `--repetitions` AB/BA pairs, the compares and the verdict itself.
7. Read the outcome with `recipe-lab report --ledger ledger.jsonl --output
   report.md`; stop when the objective is met, the budget is exhausted, or
   evidence is repeatedly incomplete.

## Cold-start config generation

When no usable recipe exists, generate one instead of tuning. The steps below
are chained by `tools/recipe-lab/examples/cold-start/cold-start.sh` (set the
environment variables listed in its header to run end to end); drive them by
hand when you need to inspect between steps.

1. Freeze and split the dataset: `recipe-lab hash --file cases.jsonl`, then
   `recipe-lab split --cases cases.jsonl --output-dir splits/`. Input: a
   labeled MCQ jsonl. Output: calib/dev/frozen_test files and the content hash
   the spec must pin. Failure: a later hash mismatch means the dataset moved -
   restart from a frozen copy, never relabel in place.
2. Observe the router's own labels: `recipe-lab observe-labels --spec
   spec.yaml --split calib --out observe/ --router-bin ... --mock-backend
   ...`. Input: the spec, a router binary, its model backend. Output:
   labels-calib.jsonl (case_id/router_label/confidence). Failure: the router
   did not come up or mislabels everything - read the router log; never
   continue on guessed labels.
3. Profile every backend model: `recipe-lab profile --spec spec.yaml --labels
   observe/ --splits calib,dev --out profile/ --model-eval-dir ...`. Input:
   the observed labels plus OpenAI-compatible endpoints for each model.
   Output: profile.json (accuracy intervals, token costs, quality gates) and
   per-split judged output. Failure: an alias that cannot round-trip, or a
   quality-gate failure, is a hard stop - fix the spec aliases or the model
   output first.
4. Bootstrap candidates: `recipe-lab bootstrap --spec spec.yaml --profile
   profile/ --model-eval-dir ... --lambdas 0,0.1,0.2,0.3 --out cands/`.
   Input: profile.json. Output: one scaffold plus one completed candidate per
   lambda, each pre-validated with `config validate`. Failure: any
   result_to_config warning or failed validation aborts the batch - repair
   the profile, do not hand-edit candidates around the failure.
5. Rank by replay: `recipe-lab bank import --spec spec.yaml --profile-dir
   profile/ --split dev --output bank.jsonl`, then `recipe-lab run --spec
   spec.yaml --split dev --cases splits/dev.jsonl --bank bank.jsonl
   --candidates cands/ --proposer bootstrap --ledger ledger.jsonl ...`.
   Output: RANKED records. Failure: rejected candidates never enter the
   ledger; read the bootstrap provenance instead of guessing why.
6. Final verdict and bundle: pick a winner with `recipe-lab ledger top --file
   ledger.jsonl --metric-key capability.score`, author the candidate JSON,
   then `recipe-lab verify --spec spec.yaml --split frozen_test --final
   --candidate cand.json --ledger ledger.jsonl ...` against real endpoints,
   and finally `recipe-lab bundle --spec spec.yaml --ledger ledger.jsonl
   --record rec-... --out bundle/ --config FINAL.yaml --skeleton SCAFFOLD.yaml
   --profile profile/profile.json`. Failure: any non-PASS verdict stops the
   loop - frozen_test is never retried after a failed final.

Run `make check` after changing recipe-lab code; skill changes follow the
`tools/agent` structure checks and public-skill sync tests.
