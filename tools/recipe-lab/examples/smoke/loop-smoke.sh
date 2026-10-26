#!/bin/bash
# recipe-lab loop smoke (Sub-issue 2 closed loop):
# answer bank -> threshold proposer -> replay-tier run over candidates (each
# candidate gets its own router lifecycle) -> RANKED records -> ranking.
#
# Run on a GPU compute node (from the head node):
#   nohup srun --partition=h20 --gres=gpu:1 --cpus-per-task=8 --time=00:50:00 \
#     bash ~/jinghaoyu/sr-build/scripts/loop-smoke.sh \
#     > ~/jinghaoyu/recipe-lab-workspace/logs/loop-smoke-launch.log 2>&1 &
set -u
source "$HOME/jinghaoyu/sr-build/env.sh"
REPO="$HOME/jinghaoyu/semantic-router-fork"
ASSETS="$REPO/tools/recipe-lab/examples/smoke"
RL="$HOME/jinghaoyu/venvs/recipe-lab-venv/bin"
WORK="$HOME/jinghaoyu/recipe-lab-workspace/runs/loop-smoke-$(date +%m%d%H%M%S)"
mkdir -p "$WORK"
echo "WORK=$WORK"

# 1. answer bank: mock backend up temporarily (direct queries, no router)
"$HOME/jinghaoyu/vllm028-env/bin/python" "$ASSETS/mock_backend.py" --port 8000 > "$WORK/mock.log" 2>&1 &
MOCK_PID=$!
sleep 2
"$RL/recipe-lab" bank build --cases "$ASSETS/cases.jsonl" \
  --models local/private-qwen,cloud/frontier-reasoning \
  --endpoint http://127.0.0.1:8000 --output "$WORK/answer-bank.jsonl" || exit 1
kill $MOCK_PID 2>/dev/null
wait $MOCK_PID 2>/dev/null

# 2. threshold proposer: candidate configs for the depth signal
"$RL/recipe-lab" propose-thresholds --base-config "$ASSETS/smoke-config.yaml" \
  --signal depth "--deltas=-0.1,-0.05,0.05,0.1" --output-dir "$WORK/candidates" || exit 1

# 3. replay-tier run: baseline anchors the deltas, then each candidate gets its
#    own router lifecycle; RANKED records land in the ledger
"$RL/recipe-lab" run --tier replay \
  --spec "$ASSETS/smoke-spec.yaml" \
  --cases "$ASSETS/cases.jsonl" \
  --bank "$WORK/answer-bank.jsonl" \
  --candidates "$WORK/candidates" \
  --baseline-config "$ASSETS/smoke-config.yaml" \
  --proposer threshold-search \
  --ledger "$WORK/ledger.jsonl" \
  --output-dir "$WORK/evals" \
  --router-bin "$SR_BUILD/bin/semantic-router" \
  --mock-backend "$ASSETS/mock_backend.py" \
  --python-bin "$HOME/jinghaoyu/vllm028-env/bin/python" \
  --hf-home "$HOME/jinghaoyu/hf-home" \
  --round 1 || exit 1

# 4. ranking
echo ====LEDGER-TOP====
"$RL/recipe-lab" ledger top --file "$WORK/ledger.jsonl" --metric-key capability.score
echo "LOOP-DONE $WORK"
