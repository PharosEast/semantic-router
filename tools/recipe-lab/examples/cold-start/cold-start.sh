#!/bin/bash
# recipe-lab cold start: no existing recipe is assumed. From a labeled MCQ
# dataset and n backend models this walks the full generation chain:
#
#   MODE=template (default, sub-issue-8 §2.2):
#     split -> observe-lanes -> profile --lanes -> bindings -> bank import
#           -> run -> verify --split frozen_test --final --lanes -> bundle --bindings
#   MODE=legacy (the result_to_config domain scaffold, §2.5 compatibility):
#     split -> observe-labels -> profile --labels -> bootstrap --legacy-domain-template
#           -> bank import -> run -> verify --split frozen_test --final --labels
#           -> bundle --skeleton
#
# Every input comes from an environment variable, so the script never embeds
# private paths. Phases run greedily and the script stops at the first one
# whose inputs are missing (its error message says what to set next):
#   phase A needs CASES
#   phase B adds SPEC            (observe/profile/bindings|bootstrap/bank/run)
#   phase C adds the live final verdict + bundle; with DRIVER=model_eval
#            (the default) it also starts the stack the drivers talk to and
#            auto-picks the winner from `recipe-lab ledger top`
#
# Environment:
#   REPO            semantic-router checkout        (default: derived)
#   RECIPE_LAB_BIN  dir with recipe-lab CLI+python  (default: $REPO/.venv/bin)
#   PYTHON_BIN      python for helpers/mock backend (default: $RECIPE_LAB_BIN/python)
#   SR_BUILD        build prefix                    (default: $REPO/build)
#   ROUTER_BIN      router binary                   (default: $SR_BUILD/bin/semantic-router)
#   WORK_ROOT       scratch root for run dirs       (default: $REPO/.tmp-cold-start)
#   HF_HOME         HuggingFace cache               (default: $WORK_ROOT/hf-home)
#   CASES           labeled MCQ dataset jsonl       (required; phase A)
#   SPEC            experiment spec yaml pinning the CASES hash (required; phase B)
#                   MODE=template requires the spec's template/models with
#                   capabilities/providers_config/bootstrap sections
#   MODE            template (default) or legacy    (sub-issue-8 §2.5)
#   VLLM_SR_BIN     vllm-sr binary for recipe builtin list/init and config
#                   validate                        (default: $VALIDATE_BIN)
#   MODEL_NAME      public entrypoint name of the generated config (default:
#                   vllm-sr/auto; template mode)
#   MODEL_EVAL_DIR  model_eval checkout             (default: $REPO/src/training/model_eval)
#   MOCK_BACKEND    mock backend script             (default: examples/smoke/mock_backend.py)
#   LAMBDAS         bindings/bootstrap lambda grid  (default: 0,0.1,0.2,0.3)
#   BANK            1 to import the answer bank from the profile (default), 0 to skip
#   DRIVER          phase C verify driver: model_eval (default) or sr-bench
#   CLASSIFICATION_CONFIG  maintained source config the legacy C2 seed and every
#                   candidate take their classification sections from
#                   (default: $REPO/config/config.yaml; legacy mode only)
#   EXCLUDE_LANES   space-separated lanes for bindings --exclude-decision
#                   (template mode, e.g. "vision long_context")
#   VALIDATE_BIN    config validator binary invoked as
#                   '<bin> config validate --config <path>' by bindings/
#                   bootstrap and bundle (default: vllm-sr)
#   BACKEND_PORT    port for the runner-spawned mock backends (default: 8010)
#
# model_eval driver (DRIVER=model_eval, default) additionally:
#   ANSWER_PLAN     fixture JSON {served model: {category: correct?}} handed to
#                   the long-lived mock backend            (required)
#   MOCK_PORT       port for the long-lived mock backend (default: ephemeral);
#                   the SPEC's models[].endpoint must point at this port
#   ROUTER_MODEL    model name the verify arms send      (default: vllm-sr/auto)
#   VERIFY_REPETITIONS  verify AB/BA pairs               (default: 1)
#   VERIFY_CANDIDATE / BUNDLE_CONFIG / BUNDLE_SKELETON
#                   optional explicit winner (candidate JSON / config yaml /
#                   scaffold yaml); when all are unset the script picks the
#                   `ledger top` record's candidate and its sidecar scaffold
#   VERIFY_FLAGS    extra verify flags passed through
#
# sr-bench driver (DRIVER=sr-bench):
#   VERIFY_CANDIDATE candidate JSON for the ledger record        (required)
#   BUNDLE_CONFIG   winning candidate config yaml                 (required)
#   BUNDLE_SKELETON winning scaffold yaml from bootstrap          (required)
#   VERIFY_FLAGS    extra verify flags, e.g. --baseline-router-endpoint ...
#   SPLIT_FLAGS / OBSERVE_FLAGS / PROFILE_FLAGS / BOOTSTRAP_FLAGS / RUN_FLAGS
#                   extra flags passed through to the matching step
set -u
ASSETS="$(cd "$(dirname "$0")" && pwd)"
REPO="${REPO:-$(cd "$ASSETS/../../../.." && pwd)}"
RL="${RECIPE_LAB_BIN:-$REPO/.venv/bin}"
PYBIN="${PYTHON_BIN:-$RL/python}"
SR_BUILD="${SR_BUILD:-$REPO/build}"
ROUTER_BIN="${ROUTER_BIN:-$SR_BUILD/bin/semantic-router}"
WORK_ROOT="${WORK_ROOT:-$REPO/.tmp-cold-start}"
HF_HOME="${HF_HOME:-$WORK_ROOT/hf-home}"
MODEL_EVAL_DIR="${MODEL_EVAL_DIR:-$REPO/src/training/model_eval}"
MOCK_BACKEND="${MOCK_BACKEND:-$REPO/tools/recipe-lab/examples/smoke/mock_backend.py}"
LAMBDAS="${LAMBDAS:-0,0.1,0.2,0.3}"
DRIVER="${DRIVER:-model_eval}"
MODE="${MODE:-template}"
VALIDATE_BIN="${VALIDATE_BIN:-vllm-sr}"
VLLM_SR_BIN="${VLLM_SR_BIN:-$VALIDATE_BIN}"
MODEL_NAME="${MODEL_NAME:-vllm-sr/auto}"
CLASSIFICATION_CONFIG="${CLASSIFICATION_CONFIG:-$REPO/config/config.yaml}"
BACKEND_PORT="${BACKEND_PORT:-8010}"
ROUTER_MODEL="${ROUTER_MODEL:-vllm-sr/auto}"
VERIFY_REPETITIONS="${VERIFY_REPETITIONS:-1}"
WORK="$WORK_ROOT/cold-start-$(date +%m%d%H%M%S)"
mkdir -p "$WORK"
echo "WORK=$WORK"
PIDS=()
cleanup() {
  for pid in "${PIDS[@]:-}"; do
    kill "$pid" 2>/dev/null || true
  done
}
trap cleanup EXIT

free_port() {
  "$PYBIN" -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1",0)); print(s.getsockname()[1]); s.close()'
}

wait_router_ready() { # $1 = management base url, $2 = log path
  "$PYBIN" - "$1" "$2" <<'EOF'
import sys, time, urllib.request
url, log = sys.argv[1], sys.argv[2]
deadline = time.time() + 600
while time.time() < deadline:
    try:
        with urllib.request.urlopen(url + "/api/v1/inventory/model-runtime", timeout=2) as resp:
            body = resp.read().decode("utf-8", errors="replace").replace(" ", "")
        if '"ready":true' in body:
            sys.exit(0)
    except Exception:
        pass
    time.sleep(1)
sys.exit(f"router not ready within 600s (see {log})")
EOF
}

start_router() { # $1 = config path, $2 = log path, $3 = ports file (<listener> <mgmt>)
  local api_port listener_port
  api_port=$(free_port)
  listener_port=$("$PYBIN" -c 'import sys, yaml; c=yaml.safe_load(open(sys.argv[1])); print(c["listeners"][0]["port"])' "$1")
  "$ROUTER_BIN" -config "$1" -gateway standalone -api-port "$api_port" \
    > "$2" 2>&1 &
  PIDS+=($!)
  echo "$listener_port $api_port" > "$3"
  wait_router_ready "http://127.0.0.1:$api_port" "$2"
}

# ---- phase A: fingerprint the dataset, then split it -----------------------
# The hash is the frozen dataset identity (A4): the spec must pin it, and
# every later command verifies it before touching the data.
if [ -z "${CASES:-}" ]; then
  echo "cold-start: set CASES=<labeled MCQ jsonl> to start (see header)" >&2
  exit 1
fi
CASES_HASH=$("$RL/recipe-lab" hash --file "$CASES" | "$PYBIN" -c "import json,sys; print(json.load(sys.stdin)['content_hash'])")
echo "CASES_HASH=$CASES_HASH"
# shellcheck disable=SC2086  # SPLIT_FLAGS carries whole CLI flags
"$RL/recipe-lab" split --cases "$CASES" --output-dir "$WORK/splits" ${SPLIT_FLAGS:-} || exit 1

# ---- phase B prep: the long-lived model endpoint ---------------------------
# model_eval driver: the spec's models[].endpoint (direct-model evals in the
# profile, candidate backends in every config, and the live reference arms)
# all point at one long-lived mock backend on MOCK_PORT.
if [ "$DRIVER" = "model_eval" ]; then
  if [ -z "${ANSWER_PLAN:-}" ]; then
    echo "cold-start: DRIVER=model_eval needs ANSWER_PLAN=<fixture json>" \
      "{served model: {category: correct?}}" >&2
    exit 1
  fi
  MOCK_PORT="${MOCK_PORT:-$(free_port)}"
  echo "MOCK_PORT=$MOCK_PORT"
  "$PYBIN" "$MOCK_BACKEND" --port "$MOCK_PORT" --cases "$CASES" \
    --answer-plan "$ANSWER_PLAN" &
  PIDS+=($!)
  if ! "$PYBIN" - "$MOCK_PORT" <<'EOF'
import sys, time, urllib.request
deadline = time.time() + 30
while time.time() < deadline:
    try:
        urllib.request.urlopen(f"http://127.0.0.1:{sys.argv[1]}/v1/models", timeout=2)
        sys.exit(0)
    except Exception:
        time.sleep(0.2)
sys.exit("long-lived mock backend not ready")
EOF
  then
    exit 1
  fi
fi

# ---- phase B: observe, profile, generate candidates, bank, replay ranking --
# Requires SPEC. MODE=template observes lanes (sub-issue-8 §2.2) and
# generates candidates via `recipe builtin init`; MODE=legacy observes the
# domain labels and runs the result_to_config scaffold. Model accuracy,
# token costs and reasoning deltas come from model_eval; the answer bank is
# converted from the profile's dev judged output, so no extra endpoint runs.
if [ -z "${SPEC:-}" ]; then
  echo "cold-start: phase A done in $WORK; set SPEC=<spec.yaml pinning CASES_HASH> for phase B" >&2
  exit 1
fi
OBSERVE_SPLITS="calib dev"
if [ "$DRIVER" = "model_eval" ]; then
  # Y1: the verify driver rewrites the frozen split's task categories from
  # the C2 grouping source, so frozen_test is observed too.
  OBSERVE_SPLITS="calib dev frozen_test"
fi
if [ "$MODE" = "template" ]; then
  for split in $OBSERVE_SPLITS; do
    # shellcheck disable=SC2086  # OBSERVE_FLAGS carries whole CLI flags
    "$RL/recipe-lab" observe-lanes --spec "$SPEC" --split "$split" \
      --out "$WORK/observe-lanes" --vllm-sr-bin "$VLLM_SR_BIN" \
      --model-name "$MODEL_NAME" \
      --router-bin "$ROUTER_BIN" --mock-backend "$MOCK_BACKEND" \
      --python-bin "$PYBIN" --hf-home "$HF_HOME" --backend-port "$BACKEND_PORT" \
      ${OBSERVE_FLAGS:-} || exit 1
  done
  # shellcheck disable=SC2086  # PROFILE_FLAGS carries whole CLI flags
  "$RL/recipe-lab" profile --spec "$SPEC" --lanes "$WORK/observe-lanes" \
    --splits calib,dev --out "$WORK/profile" \
    --model-eval-dir "$MODEL_EVAL_DIR" --python-bin "$PYBIN" ${PROFILE_FLAGS:-} || exit 1
  EXCLUDE_ARGS=""
  for lane in ${EXCLUDE_LANES:-}; do
    EXCLUDE_ARGS="$EXCLUDE_ARGS --exclude-decision $lane"
  done
  # shellcheck disable=SC2086
  "$RL/recipe-lab" bindings --spec "$SPEC" --profile "$WORK/profile" \
    --lambdas "$LAMBDAS" $EXCLUDE_ARGS \
    --vllm-sr-bin "$VLLM_SR_BIN" --model-name "$MODEL_NAME" \
    --out "$WORK/cands" ${BINDINGS_FLAGS:-} || exit 1
  GROUPING_DIR="$WORK/observe-lanes"
  GROUPING_KIND="lanes"
  PROPOSER="bindings"
else
  for split in $OBSERVE_SPLITS; do
    # shellcheck disable=SC2086  # OBSERVE_FLAGS carries whole CLI flags
    "$RL/recipe-lab" observe-labels --spec "$SPEC" --split "$split" \
      --out "$WORK/observe" --classification-config "$CLASSIFICATION_CONFIG" \
      --router-bin "$ROUTER_BIN" --mock-backend "$MOCK_BACKEND" \
      --python-bin "$PYBIN" --hf-home "$HF_HOME" --backend-port "$BACKEND_PORT" \
      ${OBSERVE_FLAGS:-} || exit 1
  done
  # shellcheck disable=SC2086  # PROFILE_FLAGS carries whole CLI flags
  "$RL/recipe-lab" profile --spec "$SPEC" --labels "$WORK/observe" --splits calib,dev \
    --out "$WORK/profile" --model-eval-dir "$MODEL_EVAL_DIR" \
    --python-bin "$PYBIN" ${PROFILE_FLAGS:-} || exit 1
  # shellcheck disable=SC2086  # BOOTSTRAP_FLAGS carries whole CLI flags
  "$RL/recipe-lab" bootstrap --legacy-domain-template --spec "$SPEC" \
    --profile "$WORK/profile" \
    --model-eval-dir "$MODEL_EVAL_DIR" --classification-config "$CLASSIFICATION_CONFIG" \
    --router-bin "$ROUTER_BIN" --mock-backend "$MOCK_BACKEND" \
    --python-bin "$PYBIN" --backend-port "$BACKEND_PORT" --hf-home "$HF_HOME" \
    --validate-bin "$VALIDATE_BIN" --lambdas "$LAMBDAS" \
    --out "$WORK/cands" ${BOOTSTRAP_FLAGS:-} || exit 1
  GROUPING_DIR="$WORK/observe"
  GROUPING_KIND="labels"
  PROPOSER="bootstrap"
fi

if [ "${BANK:-1}" = "1" ]; then
  "$RL/recipe-lab" bank import --spec "$SPEC" --profile-dir "$WORK/profile" \
    --split dev --output "$WORK/bank.jsonl" || exit 1
else
  echo "cold-start: BANK=0, skipping bank import (provide --bank yourself in RUN_FLAGS)"
fi

# rank the bootstrap candidates only: the out dir also holds the scaffolds
# and the seed config, which are not candidates
RUN_CANDIDATES=""
for f in "$WORK"/cands/candidate-lambda-*.yaml; do
  [ -e "$f" ] || continue
  RUN_CANDIDATES="${RUN_CANDIDATES:+$RUN_CANDIDATES,}$f"
done
if [ -z "$RUN_CANDIDATES" ]; then
  echo "cold-start: bootstrap produced no candidate configs in $WORK/cands" >&2
  exit 1
fi
# shellcheck disable=SC2086  # RUN_FLAGS carries whole CLI flags
"$RL/recipe-lab" run --tier replay --spec "$SPEC" --split dev \
  --cases "$WORK/splits/dev.jsonl" \
  --bank "${BANK_FILE:-$WORK/bank.jsonl}" \
  --candidates "$RUN_CANDIDATES" --proposer "$PROPOSER" \
  --ledger "$WORK/ledger.jsonl" --output-dir "$WORK/evals" \
  --router-bin "$ROUTER_BIN" --mock-backend "$MOCK_BACKEND" \
  --python-bin "$PYBIN" --hf-home "$HF_HOME" --backend-port "$BACKEND_PORT" \
  --round 1 \
  ${RUN_FLAGS:-} || exit 1

# ---- phase C: live final verdict on frozen_test, then bundle ---------------
# model_eval driver: the script boots the baseline and candidate routers
# itself and auto-picks the winner from `recipe-lab ledger top` unless
# VERIFY_CANDIDATE and BUNDLE_CONFIG are given. The baseline arm serves
# MODE=template's BASELINE_CONFIG (e.g. the all-strongest reference config)
# or the legacy seed config. bundle refuses anything but a live --final PASS
# record whose --config/--bindings(--skeleton)/--profile hash-match the
# evidence.
if [ "$DRIVER" = "model_eval" ]; then
  VERIFY_DIR="$WORK/verify"
  mkdir -p "$VERIFY_DIR"
  if [ "$MODE" = "template" ]; then
    if [ -n "${VERIFY_CANDIDATE:-}" ] || [ -n "${BUNDLE_CONFIG:-}" ]; then
      if [ -z "${VERIFY_CANDIDATE:-}" ] || [ -z "${BUNDLE_CONFIG:-}" ]; then
        echo "cold-start: VERIFY_CANDIDATE and BUNDLE_CONFIG must be set together" >&2
        exit 1
      fi
    else
      "$RL/recipe-lab" ledger top --file "$WORK/ledger.jsonl" --spec "$SPEC" \
        > "$WORK/top.json" || exit 1
      BUNDLE_CONFIG=$("$PYBIN" -c 'import json,sys; print(json.load(open(sys.argv[1]))["candidate"]["ref"])' "$WORK/top.json") || exit 1
      [ -n "$BUNDLE_CONFIG" ] || { echo "cold-start: no ranked candidate to verify" >&2; exit 1; }
    fi
  else
    if [ -n "${VERIFY_CANDIDATE:-}" ] || [ -n "${BUNDLE_CONFIG:-}" ] || [ -n "${BUNDLE_SKELETON:-}" ]; then
      if [ -z "${VERIFY_CANDIDATE:-}" ] || [ -z "${BUNDLE_CONFIG:-}" ] || [ -z "${BUNDLE_SKELETON:-}" ]; then
        echo "cold-start: VERIFY_CANDIDATE, BUNDLE_CONFIG and BUNDLE_SKELETON must be set together" >&2
        exit 1
      fi
    else
      "$RL/recipe-lab" ledger top --file "$WORK/ledger.jsonl" --spec "$SPEC" \
        > "$WORK/top.json" || exit 1
      BUNDLE_CONFIG=$("$PYBIN" -c 'import json,sys; print(json.load(open(sys.argv[1]))["candidate"]["ref"])' "$WORK/top.json") || exit 1
      [ -n "$BUNDLE_CONFIG" ] || { echo "cold-start: no ranked candidate to verify" >&2; exit 1; }
    fi
  fi
  BUNDLE_SIDECAR="${BUNDLE_SIDECAR:-${BUNDLE_CONFIG%.yaml}.provenance.json}"
  if [ "$MODE" = "template" ]; then
    if [ -z "${BUNDLE_BINDINGS:-}" ]; then
      BUNDLE_BINDINGS=$("$PYBIN" -c 'import json,sys; print(json.load(open(sys.argv[1]))["bindings"]["path"])' "$BUNDLE_SIDECAR") || exit 1
    fi
  else
    if [ -z "${BUNDLE_SKELETON:-}" ]; then
      BUNDLE_SKELETON=$("$PYBIN" -c 'import json,sys; print(json.load(open(sys.argv[1]))["scaffold"]["path"])' "$BUNDLE_SIDECAR") || exit 1
    fi
  fi
  if [ -z "${VERIFY_CANDIDATE:-}" ]; then
    "$PYBIN" -c 'import json,sys; print(json.dumps(json.load(open(sys.argv[1]))["candidate"]))' \
      "$WORK/top.json" > "$VERIFY_DIR/candidate.json" || exit 1
    VERIFY_CANDIDATE="$VERIFY_DIR/candidate.json"
  fi

  # The verify arms serve patched copies: the baseline router runs the
  # baseline config with its backends re-pointed at the long-lived mock
  # (MODE=template: BASELINE_CONFIG, e.g. the all-strongest reference;
  # MODE=legacy: the seed config), and both routers get their own free
  # listener port. The candidate router must serve exactly the file
  # --candidate-config names (the Y4 management-API hash check), so the
  # bundle packages the patched candidate copy, not the original.
  if [ "$MODE" = "template" ]; then
    if [ -z "${BASELINE_CONFIG:-}" ]; then
      echo "cold-start: MODE=template with DRIVER=model_eval needs BASELINE_CONFIG=<baseline reference config (e.g. all-strongest)>" >&2
      exit 1
    fi
    BASELINE_SRC="$BASELINE_CONFIG"
  else
    BASELINE_SRC="$WORK/cands/seed-config.yaml"
  fi
  BASELINE_PORT=$(free_port)
  CANDIDATE_PORT=$(free_port)
  patch_listener() { # $1 = src, $2 = dst, $3 = listener port
    if ! "$PYBIN" - "$1" "$2" "$3" <<'EOF'
import sys, yaml
config = yaml.safe_load(open(sys.argv[1], encoding="utf-8"))
config["listeners"][0]["port"] = int(sys.argv[3])
with open(sys.argv[2], "w", encoding="utf-8") as fh:
    yaml.safe_dump(config, fh, sort_keys=False)
EOF
    then
      return 1
    fi
  }
  patch_listener "$BASELINE_SRC" "$VERIFY_DIR/baseline.yaml" \
    "$BASELINE_PORT" || exit 1
  if ! "$PYBIN" - "$VERIFY_DIR/baseline.yaml" "$MOCK_PORT" <<'EOF'
import sys, yaml
config = yaml.safe_load(open(sys.argv[1], encoding="utf-8"))
for model in config["providers"]["models"]:
    for ref in model["backend_refs"]:
        ref["endpoint"] = f"127.0.0.1:{int(sys.argv[2])}"
with open(sys.argv[1], "w", encoding="utf-8") as fh:
    yaml.safe_dump(config, fh, sort_keys=False)
EOF
  then
    exit 1
  fi
  patch_listener "$BUNDLE_CONFIG" "$VERIFY_DIR/candidate.yaml" \
    "$CANDIDATE_PORT" || exit 1
  BUNDLE_CONFIG="$VERIFY_DIR/candidate.yaml"

  start_router "$VERIFY_DIR/baseline.yaml" "$VERIFY_DIR/baseline-router.log" \
    "$VERIFY_DIR/baseline.ports" || exit 1
  read -r BASELINE_LISTENER BASELINE_MGMT < "$VERIFY_DIR/baseline.ports"
  start_router "$VERIFY_DIR/candidate.yaml" "$VERIFY_DIR/candidate-router.log" \
    "$VERIFY_DIR/candidate.ports" || exit 1
  read -r CANDIDATE_LISTENER CANDIDATE_MGMT < "$VERIFY_DIR/candidate.ports"
  echo "BASELINE_ROUTER=$BASELINE_LISTENER/$BASELINE_MGMT CANDIDATE_ROUTER=$CANDIDATE_LISTENER/$CANDIDATE_MGMT"

  # the frozen arms answer the contract-declared model_eval rendering
  "$RL/recipe-lab" cases to-model-eval --source "$WORK/splits/frozen_test.jsonl" \
    --out "$VERIFY_DIR/tasks-frozen_test.jsonl" || exit 1

  # shellcheck disable=SC2086  # VERIFY_FLAGS carries whole CLI flags
  "$RL/recipe-lab" verify --spec "$SPEC" --split frozen_test --final \
    --candidate "$VERIFY_CANDIDATE" --ledger "$WORK/ledger.jsonl" \
    --tasks "$VERIFY_DIR/tasks-frozen_test.jsonl" \
    "--$GROUPING_KIND" "$GROUPING_DIR/${GROUPING_KIND}-frozen_test.jsonl" \
    --candidate-config "$BUNDLE_CONFIG" \
    --baseline-router-endpoint "http://127.0.0.1:$BASELINE_LISTENER/v1" \
    --candidate-router-endpoint "http://127.0.0.1:$CANDIDATE_LISTENER/v1" \
    --candidate-management-endpoint "http://127.0.0.1:$CANDIDATE_MGMT" \
    --router-model "$ROUTER_MODEL" --model-eval-dir "$MODEL_EVAL_DIR" \
    --python-bin "$PYBIN" --repetitions "$VERIFY_REPETITIONS" \
    ${VERIFY_FLAGS:-} > "$VERIFY_DIR/result.json"
  VERIFY_RC=$?
  cat "$VERIFY_DIR/result.json"
  if [ "$VERIFY_RC" -ne 0 ]; then
    echo "cold-start: verify exited with $VERIFY_RC" >&2
    exit "$VERIFY_RC"
  fi
else
  if [ -z "${VERIFY_CANDIDATE:-}" ] || [ -z "${BUNDLE_CONFIG:-}" ]; then
    echo "cold-start: phase B done in $WORK; pick a winner with" \
      "'recipe-lab ledger top --file $WORK/ledger.jsonl --metric-key capability.score'," \
      "then set VERIFY_CANDIDATE, BUNDLE_CONFIG and VERIFY_FLAGS for phase C" >&2
    exit 1
  fi
  if [ "$MODE" = "template" ]; then
    if [ -z "${BUNDLE_BINDINGS:-}" ]; then
      BUNDLE_SIDECAR="${BUNDLE_SIDECAR:-${BUNDLE_CONFIG%.yaml}.provenance.json}"
      BUNDLE_BINDINGS=$("$PYBIN" -c 'import json,sys; print(json.load(open(sys.argv[1]))["bindings"]["path"])' "$BUNDLE_SIDECAR") || exit 1
    fi
  else
    if [ -z "${BUNDLE_SKELETON:-}" ]; then
      echo "cold-start: MODE=legacy needs BUNDLE_SKELETON=<winning scaffold yaml> for phase C" >&2
      exit 1
    fi
  fi
  # shellcheck disable=SC2086  # VERIFY_FLAGS carries whole CLI flags
  "$RL/recipe-lab" verify --spec "$SPEC" --split frozen_test --final \
    --candidate "$VERIFY_CANDIDATE" --ledger "$WORK/ledger.jsonl" ${VERIFY_FLAGS:-} || exit 1
fi

RECORD_ID=$(tail -n 1 "$WORK/ledger.jsonl" | "$PYBIN" -c "import json,sys; print(json.loads(sys.stdin.readline())['record_id'])")
echo "RECORD_ID=$RECORD_ID"
BUNDLE_SIDECAR="${BUNDLE_SIDECAR:-${BUNDLE_CONFIG%.yaml}.provenance.json}"
# the sidecar (bootstrap or bindings) sits beside the candidate config
BUNDLE_ARGS="--spec $SPEC --ledger $WORK/ledger.jsonl --record $RECORD_ID \
  --out $WORK/bundle --config $BUNDLE_CONFIG \
  --candidate-sidecar $BUNDLE_SIDECAR --validate-bin $VALIDATE_BIN \
  --profile $WORK/profile/profile.json"
if [ "$MODE" = "template" ]; then
  BUNDLE_ARGS="$BUNDLE_ARGS --bindings $BUNDLE_BINDINGS"
else
  BUNDLE_ARGS="$BUNDLE_ARGS --skeleton $BUNDLE_SKELETON"
fi
# shellcheck disable=SC2086
"$RL/recipe-lab" bundle $BUNDLE_ARGS || exit 1
echo "COLD-START-DONE $WORK/bundle"
