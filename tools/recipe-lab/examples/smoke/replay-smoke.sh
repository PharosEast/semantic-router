#!/bin/bash
# recipe-lab replay smoke: router (d3-lite + router_replay) + mock backend
# + labeled requests + replay dump + join + attribution.
#
# Run on a GPU compute node (from the head node):
#   nohup srun --partition=h20 --gres=gpu:1 --cpus-per-task=8 --time=00:45:00 \
#     bash ~/jinghaoyu/sr-build/scripts/replay-smoke.sh \
#     > ~/jinghaoyu/recipe-lab-workspace/logs/replay-smoke-launch.log 2>&1 &
# Then poll the launch log for SMOKE-DONE / SMOKE-FAILED.
set -u
source "$HOME/jinghaoyu/sr-build/env.sh"
REPO="$HOME/jinghaoyu/semantic-router-fork"
ASSETS="$REPO/tools/recipe-lab/examples/smoke"
RL="$HOME/jinghaoyu/venvs/recipe-lab-venv/bin"
SMOKE_DIR="$HOME/jinghaoyu/recipe-lab-workspace/runs/replay-smoke-$(date +%m%d%H%M%S)"
mkdir -p "$SMOKE_DIR"
echo "SMOKE_DIR=$SMOKE_DIR"

cleanup() {
  for pid in ${ROUTER_PID:-} ${MOCK_PID:-}; do kill "$pid" 2>/dev/null; done
}
trap cleanup EXIT

# 1. mock OpenAI backend on :8000
"$HOME/jinghaoyu/vllm028-env/bin/python" "$ASSETS/mock_backend.py" --port 8000 > "$SMOKE_DIR/mock.log" 2>&1 &
MOCK_PID=$!
sleep 2
if ! curl -sf http://127.0.0.1:8000/v1/models > /dev/null; then echo SMOKE-FAILED mock; cat "$SMOKE_DIR/mock.log"; exit 1; fi
echo MOCK-OK

# 2. router with d3-lite (model_runtime) + router_replay on every decision
# -gateway standalone: the Router serves the OpenAI listeners itself (no Envoy)
"$SR_BUILD/bin/semantic-router" -config "$ASSETS/smoke-config.yaml" -gateway standalone > "$SMOKE_DIR/router.log" 2>&1 &
ROUTER_PID=$!

# 3. wait for the decision model to report ready via the management API
READY=0
for i in $(seq 1 60); do
  sleep 5
  if ! kill -0 $ROUTER_PID 2>/dev/null; then
    echo SMOKE-FAILED router-died; tail -40 "$SMOKE_DIR/router.log"; exit 1
  fi
  if curl -sf http://127.0.0.1:8080/api/v1/inventory/model-runtime 2>/dev/null | grep -q '"ready":true'; then
    READY=1; echo "ROUTER-READY after ~$((i*5))s"; break
  fi
done
if [ "$READY" != "1" ]; then echo SMOKE-FAILED not-ready; tail -40 "$SMOKE_DIR/router.log"; exit 1; fi

# 4. send the labeled cases through the router
"$RL/python" "$ASSETS/send_requests.py" --cases "$ASSETS/cases.jsonl" --router http://127.0.0.1:8899 2>&1 | tee "$SMOKE_DIR/requests.log"

# 5. dump replay records from the management API
curl -sf 'http://127.0.0.1:8080/api/v1/observability/replays?showDetails=true' > "$SMOKE_DIR/replays.json"
echo "REPLAYS-BYTES=$(wc -c < "$SMOKE_DIR/replays.json")"

# 6. join with labeled cases -> attribution records
"$RL/recipe-lab" collect-replays --url http://127.0.0.1:8080 \
  --cases "$ASSETS/cases.jsonl" \
  --output "$SMOKE_DIR/records.jsonl" --family d3 2>&1 | tee "$SMOKE_DIR/collect.log"

# 7. answer bank: query each case against each routing model directly, then
#    join correctness by (case_id, routed model) -> no_evidence becomes real outcomes
"$RL/recipe-lab" bank build --cases "$ASSETS/cases.jsonl" \
  --models local/private-qwen,cloud/frontier-reasoning \
  --endpoint http://127.0.0.1:8000 \
  --output "$SMOKE_DIR/answer-bank.jsonl" 2>&1 | tee "$SMOKE_DIR/bank-build.log"
"$RL/recipe-lab" bank apply --bank "$SMOKE_DIR/answer-bank.jsonl" \
  --records "$SMOKE_DIR/records.jsonl" --output "$SMOKE_DIR/records-graded.jsonl" 2>&1 | tee "$SMOKE_DIR/bank-apply.log"

# 8. attribution over graded records
"$RL/recipe-lab" attribute --spec "$ASSETS/smoke-spec.yaml" \
  --records "$SMOKE_DIR/records-graded.jsonl" --output-dir "$SMOKE_DIR" 2>&1 | tee "$SMOKE_DIR/attribute.log"

# 9. threshold proposer: candidate configs for the depth signal
# (--deltas= form: argparse rejects values that start with a dash otherwise)
"$RL/recipe-lab" propose-thresholds --base-config "$ASSETS/smoke-config.yaml" \
  --signal depth "--deltas=-0.1,-0.05,0.05,0.1" \
  --output-dir "$SMOKE_DIR/candidates" 2>&1 | tee "$SMOKE_DIR/propose.log"

echo "SMOKE-DONE $SMOKE_DIR"
