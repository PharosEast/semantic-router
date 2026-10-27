#!/bin/bash
# Finish the live smoke verdict on the head node against a finished GPU job's
# store (compare is read-only; the service autostarts against that store).
set -u
D="$1"
RL="$HOME/jinghaoyu/venvs/recipe-lab-venv/bin"
ASSETS="$HOME/jinghaoyu/semantic-router-fork/tools/recipe-lab/examples/smoke"

printf '%s\n' '{"kind":"config","ref":"smoke-config@main","change_summary":"live smoke candidate","provenance":{"proposer":"threshold-search","insight_refs":[]}}' > /tmp/candidate.json

RUNS=$("$RL/vllm-sr" benchmark --store "$D/srbench-store" runs 2>/dev/null)
BASE=$(echo "$RUNS" | "$RL/python" -c "import json,sys; runs=json.load(sys.stdin).get('runs',[]); print(next((r['id'] for r in runs if r.get('target_id')=='single'),''))")
CAND=$(echo "$RUNS" | "$RL/python" -c "import json,sys; runs=json.load(sys.stdin).get('runs',[]); print(next((r['id'] for r in runs if r.get('target_id')=='mom'),''))")
echo "BASE=$BASE CAND=$CAND"
[ -n "$BASE" ] && [ -n "$CAND" ] || exit 1

"$RL/recipe-lab" live-verdict \
  --spec "$ASSETS/smoke-spec.yaml" \
  --baseline-run-id "$BASE" \
  --candidate-run-id "$CAND" \
  --ledger "$D/ledger.jsonl" \
  --candidate /tmp/candidate.json \
  --srbench-bin "$RL/vllm-sr" \
  --srbench-store "$D/srbench-store" 2>&1 | tail -20
echo ====LEDGER====
"$RL/recipe-lab" ledger show --file "$D/ledger.jsonl" --tier live --json 2>/dev/null | head -40
echo FINISH-DONE
