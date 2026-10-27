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
2. Read the evidence: `recipe-lab next --ledger ledger.jsonl --output context.md`.
   Read the untrusted-material header and the ledger tail before proposing.
3. Propose one candidate: edit one recipe variable in a config revision and
   note which ledger rounds motivated it (future insight_refs).
4. Validate the candidate: `vllm-sr config validate` on the revision.
   Failure: fix the config; never evaluate an invalid candidate.
5. Evaluate in the replay tier: `recipe-lab run --tier replay --candidates <config>
   --proposer agent:<id>` against the answer bank. Records land in the ledger.
6. For the final gate: run sr-bench preview/live for baseline and candidate,
   then `recipe-lab live-verdict --baseline-run-id ... --candidate-run-id ...
   --guards guards.jsonl --ledger ledger.jsonl`.
7. Read the outcome with `recipe-lab report --ledger ledger.jsonl --output
   report.md`; stop when the objective is met, the budget is exhausted, or
   evidence is repeatedly incomplete.

Run `make check` after changing recipe-lab code; skill changes follow the
`tools/agent` structure checks and public-skill sync tests.
