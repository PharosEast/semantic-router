#!/bin/bash
# recipe-lab loop over the frozen MMLU-Pro case list (24 questions, deep vs
# standard routing). The capability-aware mock makes the local lane fail hard
# questions, so threshold candidates move real accuracy.
#
# Run on a GPU compute node:
#   setsid nohup srun --partition=h20 --gres=gpu:1 --cpus-per-task=8 --time=00:50:00 \
#     bash ~/jinghaoyu/sr-build/scripts/loop-mmlu.sh \
#     > ~/jinghaoyu/recipe-lab-workspace/logs/loop-mmlu-launch.log 2>&1 < /dev/null &
set -u
source "$HOME/jinghaoyu/sr-build/env.sh"
REPO="$HOME/jinghaoyu/semantic-router-fork"
ASSETS="$REPO/tools/recipe-lab/examples/smoke"
RL="$HOME/jinghaoyu/venvs/recipe-lab-venv/bin"
WORK="$HOME/jinghaoyu/recipe-lab-workspace/runs/loop-mmlu-$(date +%m%d%H%M%S)"
mkdir -p "$WORK"
echo "WORK=$WORK"

# 1. build the frozen case list from the local MMLU-Pro snapshot
BUILD=$("$HOME/jinghaoyu/vllm028-env/bin/python" "$ASSETS/build_mmlu_cases.py" \
  --dataset-dir "$HOME/jinghaoyu/dataset/mmlu-pro" \
  --output "$WORK/cases-mmlu.jsonl" --per-deep 12 --per-standard 12 --seed 42) || exit 1
echo "$BUILD" | tee "$WORK/build.log"
HASH=$(echo "$BUILD" | grep -o "sha256:[0-9a-f]*" | head -1)

# 2. spec with the frozen dataset hash
sed "s/sha256:0000000000000000000000000000000000000000000000000000000000000000/$HASH/" \
  "$ASSETS/smoke-spec.yaml" > "$WORK/spec.yaml"

# 3. answer bank: capability-aware mock, direct queries per model
"$HOME/jinghaoyu/vllm028-env/bin/python" "$ASSETS/mock_backend.py" --port 8000 \
  --cases "$WORK/cases-mmlu.jsonl" > "$WORK/mock.log" 2>&1 &
MOCK_PID=$!
sleep 2
"$RL/recipe-lab" bank build --cases "$WORK/cases-mmlu.jsonl" \
  --models local/private-qwen,cloud/frontier-reasoning \
  --endpoint http://127.0.0.1:8000 --output "$WORK/answer-bank.jsonl" || exit 1
kill $MOCK_PID 2>/dev/null
wait $MOCK_PID 2>/dev/null

# 4. threshold candidates
"$RL/recipe-lab" propose-thresholds --base-config "$ASSETS/smoke-config.yaml" \
  --signal depth "--deltas=-0.1,-0.05,0.05,0.1" --output-dir "$WORK/candidates" || exit 1

# 5. replay-tier run: baseline anchors deltas, candidates -> RANKED ledger
"$RL/recipe-lab" run --tier replay \
  --spec "$WORK/spec.yaml" \
  --cases "$WORK/cases-mmlu.jsonl" \
  --bank "$WORK/answer-bank.jsonl" \
  --candidates "$WORK/candidates" \
  --baseline-config "$ASSETS/smoke-config.yaml" \
  --proposer threshold-search \
  --ledger "$WORK/ledger.jsonl" \
  --output-dir "$WORK/evals" \
  --router-bin "$SR_BUILD/bin/semantic-router" \
  --mock-backend "$ASSETS/mock_backend.py" \
  --mock-extra-args "--cases $WORK/cases-mmlu.jsonl" \
  --python-bin "$HOME/jinghaoyu/vllm028-env/bin/python" \
  --hf-home "$HOME/jinghaoyu/hf-home" \
  --round 1 || exit 1

# 6. ranking
echo ====LEDGER-TOP====
"$RL/recipe-lab" ledger top --file "$WORK/ledger.jsonl" --metric-key capability.score
echo "LOOP-DONE $WORK"
