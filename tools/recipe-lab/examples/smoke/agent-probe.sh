#!/bin/bash
# Agentic-traffic probe (Sub-issue 6 groundwork): multi-turn conversation +
# tool-calling payloads through the router with session protection enabled.
# CPU-only; runs on the head node or a compute node.
set -u
source "$HOME/jinghaoyu/sr-build/env.sh"
REPO="$HOME/jinghaoyu/semantic-router-fork"
ASSETS="$REPO/tools/recipe-lab/examples/smoke"
RL="$HOME/jinghaoyu/venvs/recipe-lab-venv/bin"
VLLM_PY="$HOME/jinghaoyu/vllm028-env/bin/python"
W="$HOME/jinghaoyu/recipe-lab-workspace/runs/agent-probe-$(date +%m%d%H%M%S)"
mkdir -p "$W"
echo "PROBE-WORK=$W"

ROUTER_PID=""; MOCK_PID=""
cleanup() { for pid in ${ROUTER_PID:-} ${MOCK_PID:-}; do kill "$pid" 2>/dev/null; done; }
trap cleanup EXIT

"$VLLM_PY" "$ASSETS/mock_backend.py" --port 8000 > "$W/mock.log" 2>&1 &
MOCK_PID=$!
sleep 2
curl -sf http://127.0.0.1:8000/v1/models > /dev/null || { echo PROBE-FAILED mock; cat "$W/mock.log"; exit 1; }

"$SR_BUILD/bin/semantic-router" -config "$ASSETS/agent-probe-config.yaml" -gateway standalone > "$W/router.log" 2>&1 &
ROUTER_PID=$!
READY=0
for i in $(seq 1 60); do
  sleep 5
  kill -0 $ROUTER_PID 2>/dev/null || { echo PROBE-FAILED router-died; tail -30 "$W/router.log"; exit 1; }
  if curl -sf http://127.0.0.1:8080/api/v1/inventory/model-runtime 2>/dev/null | grep -q '"ready":true'; then
    READY=1; echo "ROUTER-READY after ~$((i*5))s"; break
  fi
done
[ "$READY" = "1" ] || { echo PROBE-FAILED not-ready; tail -30 "$W/router.log"; exit 1; }

"$RL/python" "$ASSETS/agentic_probe.py" --output-dir "$W" 2>&1 | tee "$W/probe-run.log"
echo "PROBE-DONE $W"
