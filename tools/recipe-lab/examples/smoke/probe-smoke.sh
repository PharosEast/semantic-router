#!/bin/bash
# Probe the router's 400 with sr-bench-style streaming requests.
set -u
source "$HOME/jinghaoyu/sr-build/env.sh"
ASSETS="$HOME/jinghaoyu/semantic-router-fork/tools/recipe-lab/examples/smoke"
RL="$HOME/jinghaoyu/venvs/recipe-lab-venv/bin"
WORK="$HOME/jinghaoyu/recipe-lab-workspace/runs/probe-400-$(date +%H%M%S)"
mkdir -p "$WORK"
"$HOME/jinghaoyu/vllm028-env/bin/python" "$ASSETS/mock_backend.py" --port 8000 > "$WORK/mock.log" 2>&1 &
MOCK_PID=$!
sleep 1
"$SR_BUILD/bin/semantic-router" -config "$ASSETS/smoke-config.yaml" -gateway standalone > "$WORK/router.log" 2>&1 &
ROUTER_PID=$!
for i in $(seq 1 60); do
  sleep 5
  kill -0 $ROUTER_PID 2>/dev/null || { tail -20 "$WORK/router.log"; exit 1; }
  curl -sf http://127.0.0.1:8080/api/v1/inventory/model-runtime 2>/dev/null | grep -q '"ready":true' && break
done
"$RL/python" "$ASSETS/probe_400.py" --only-mom-headers --config-hash "$(curl -sf http://127.0.0.1:8080/api/v1/config/hash 2>/dev/null | grep -o "[0-9a-f]\{64\}" | head -1)" 2>&1
echo "PROBE-DONE $WORK"
kill $ROUTER_PID $MOCK_PID 2>/dev/null
