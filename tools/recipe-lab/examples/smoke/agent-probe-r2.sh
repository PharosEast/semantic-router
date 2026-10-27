#!/bin/bash
# Round-2 agentic probe: tool-loop binding vs admitted-set conflict (E/F).
set -u
source "$HOME/jinghaoyu/sr-build/env.sh"
REPO="$HOME/jinghaoyu/semantic-router-fork"
ASSETS="$REPO/tools/recipe-lab/examples/smoke"
RL="$HOME/jinghaoyu/venvs/recipe-lab-venv/bin"
VLLM_PY="$HOME/jinghaoyu/vllm028-env/bin/python"
W="$HOME/jinghaoyu/recipe-lab-workspace/runs/agent-probe-r2-$(date +%m%d%H%M%S)"
mkdir -p "$W"
echo "PROBE2-WORK=$W"
ROUTER_PID=""; MOCK_PID=""
cleanup() { for pid in ${ROUTER_PID:-} ${MOCK_PID:-}; do kill "$pid" 2>/dev/null; done; }
trap cleanup EXIT
"$VLLM_PY" "$ASSETS/mock_backend.py" --port 8000 > "$W/mock.log" 2>&1 &
MOCK_PID=$!
sleep 2
"$SR_BUILD/bin/semantic-router" -config "$ASSETS/agent-probe-config.yaml" -gateway standalone > "$W/router.log" 2>&1 &
ROUTER_PID=$!
READY=0
for i in $(seq 1 60); do
  sleep 5
  kill -0 $ROUTER_PID 2>/dev/null || { echo PROBE2-FAILED router-died; exit 1; }
  if curl -sf http://127.0.0.1:8080/api/v1/inventory/model-runtime 2>/dev/null | grep -q '"ready":true'; then READY=1; echo READY; break; fi
done
[ "$READY" = "1" ] || { echo PROBE2-FAILED not-ready; exit 1; }
"$RL/python" "$ASSETS/agentic_probe_r2.py" --output "$W/r2-results.json" 2>&1 | tee "$W/probe-run.log"
echo "====SELECTION REJECTIONS===="
grep -o '"msg":"selection_rejected"[^}]*' "$W/router.log" | head -4
echo "PROBE2-DONE $W"
