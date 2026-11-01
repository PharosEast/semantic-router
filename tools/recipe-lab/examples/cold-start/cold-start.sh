#!/bin/bash
# recipe-lab cold start: no existing recipe is assumed. From a labeled MCQ
# dataset and n backend models this walks the full generation chain:
#
#   split -> observe-labels -> profile -> bootstrap -> bank import -> run
#         -> verify --split frozen_test --final -> bundle
#
# Every input comes from an environment variable, so the script never embeds
# private paths. Phases run greedily and the script stops at the first one
# whose inputs are missing (its error message says what to set next):
#   phase A needs CASES
#   phase B adds SPEC            (observe/profile/bootstrap/bank/run)
#   phase C adds VERIFY_CANDIDATE, BUNDLE_CONFIG, BUNDLE_SKELETON and
#            VERIFY_FLAGS        (live final verdict + bundle)
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
#   MODEL_EVAL_DIR  model_eval checkout             (default: $REPO/src/training/model_eval)
#   MOCK_BACKEND    mock backend script             (default: examples/smoke/mock_backend.py)
#   LAMBDAS         bootstrap lambda grid           (default: 0,0.1,0.2,0.3)
#   BANK            1 to import the answer bank from the profile (default), 0 to skip
#   VERIFY_CANDIDATE candidate JSON for the ledger record        (required; phase C)
#   BUNDLE_CONFIG   winning candidate config yaml                 (required; phase C)
#   BUNDLE_SKELETON winning scaffold yaml from bootstrap          (required; phase C)
#   VERIFY_FLAGS    extra verify flags, e.g. --baseline-router-endpoint ...
#   SPLIT_FLAGS / OBSERVE_FLAGS / PROFILE_FLAGS / BOOTSTRAP_FLAGS / RUN_FLAGS
#                   extra flags passed through to the matching step
set -u
ASSETS="$(cd "$(dirname "$0")" && pwd)"
REPO="${REPO:-$(cd "$ASSETS/../../.." && pwd)}"
RL="${RECIPE_LAB_BIN:-$REPO/.venv/bin}"
PYBIN="${PYTHON_BIN:-$RL/python}"
SR_BUILD="${SR_BUILD:-$REPO/build}"
ROUTER_BIN="${ROUTER_BIN:-$SR_BUILD/bin/semantic-router}"
WORK_ROOT="${WORK_ROOT:-$REPO/.tmp-cold-start}"
HF_HOME="${HF_HOME:-$WORK_ROOT/hf-home}"
MODEL_EVAL_DIR="${MODEL_EVAL_DIR:-$REPO/src/training/model_eval}"
MOCK_BACKEND="${MOCK_BACKEND:-$REPO/tools/recipe-lab/examples/smoke/mock_backend.py}"
LAMBDAS="${LAMBDAS:-0,0.1,0.2,0.3}"
WORK="$WORK_ROOT/cold-start-$(date +%m%d%H%M%S)"
mkdir -p "$WORK"
echo "WORK=$WORK"

# ---- phase A: fingerprint the dataset, then split it -----------------------
# The hash is the frozen dataset identity (A4): the spec must pin it, and
# every later command verifies it before touching the data.
if [ -z "${CASES:-}" ]; then
  echo "cold-start: set CASES=<labeled MCQ jsonl> to start (see header)" >&2
  exit 1
fi
CASES_HASH=$("$RL/recipe-lab" hash --file "$CASES" | "$PYBIN" -c "import json,sys; print(json.load(sys.stdin)['content_hash'])")
echo "CASES_HASH=$CASES_HASH"
"$RL/recipe-lab" split --cases "$CASES" --output-dir "$WORK/splits" ${SPLIT_FLAGS:-} || exit 1

# ---- phase B: observe, profile, bootstrap, bank, replay ranking ------------
# Requires SPEC (calib labels come from the real router; model accuracy,
# token costs and reasoning deltas come from model_eval; the answer bank is
# converted from the profile's dev judged output, so no extra endpoint runs).
if [ -z "${SPEC:-}" ]; then
  echo "cold-start: phase A done in $WORK; set SPEC=<spec.yaml pinning CASES_HASH> for phase B" >&2
  exit 1
fi
"$RL/recipe-lab" observe-labels --spec "$SPEC" --split calib --out "$WORK/observe" \
  --router-bin "$ROUTER_BIN" --mock-backend "$MOCK_BACKEND" \
  --python-bin "$PYBIN" --hf-home "$HF_HOME" ${OBSERVE_FLAGS:-} || exit 1

"$RL/recipe-lab" profile --spec "$SPEC" --labels "$WORK/observe" --splits calib,dev \
  --out "$WORK/profile" --model-eval-dir "$MODEL_EVAL_DIR" \
  --python-bin "$PYBIN" ${PROFILE_FLAGS:-} || exit 1

"$RL/recipe-lab" bootstrap --spec "$SPEC" --profile "$WORK/profile" \
  --model-eval-dir "$MODEL_EVAL_DIR" --lambdas "$LAMBDAS" \
  --out "$WORK/cands" ${BOOTSTRAP_FLAGS:-} || exit 1

if [ "${BANK:-1}" = "1" ]; then
  "$RL/recipe-lab" bank import --spec "$SPEC" --profile-dir "$WORK/profile" \
    --split dev --output "$WORK/bank.jsonl" || exit 1
else
  echo "cold-start: BANK=0, skipping bank import (provide --bank yourself in RUN_FLAGS)"
fi

"$RL/recipe-lab" run --tier replay --spec "$SPEC" --split dev \
  --cases "$WORK/splits/dev.jsonl" \
  --bank "${BANK_FILE:-$WORK/bank.jsonl}" \
  --candidates "$WORK/cands" --proposer bootstrap \
  --ledger "$WORK/ledger.jsonl" --output-dir "$WORK/evals" \
  --router-bin "$ROUTER_BIN" --mock-backend "$MOCK_BACKEND" \
  --python-bin "$PYBIN" --hf-home "$HF_HOME" --round 1 \
  ${RUN_FLAGS:-} || exit 1

# ---- phase C: live final verdict on frozen_test, then bundle ---------------
# VERIFY_FLAGS must carry the real endpoints (e.g. --baseline-router-endpoint
# and --candidate-router-endpoint); the winner is usually picked from
# `recipe-lab ledger top` first. bundle refuses anything but a live PASS
# record backed by the final verify.
if [ -z "${VERIFY_CANDIDATE:-}" ] || [ -z "${BUNDLE_CONFIG:-}" ] || [ -z "${BUNDLE_SKELETON:-}" ]; then
  echo "cold-start: phase B done in $WORK; pick a winner with" \
    "'recipe-lab ledger top --file $WORK/ledger.jsonl --metric-key capability.score'," \
    "then set VERIFY_CANDIDATE, BUNDLE_CONFIG, BUNDLE_SKELETON and VERIFY_FLAGS for phase C" >&2
  exit 1
fi
"$RL/recipe-lab" verify --spec "$SPEC" --split frozen_test --final \
  --candidate "$VERIFY_CANDIDATE" --ledger "$WORK/ledger.jsonl" ${VERIFY_FLAGS:-} || exit 1

RECORD_ID=$(tail -n 1 "$WORK/ledger.jsonl" | "$PYBIN" -c "import json,sys; print(json.loads(sys.stdin.readline())['record_id'])")
echo "RECORD_ID=$RECORD_ID"
"$RL/recipe-lab" bundle --spec "$SPEC" --ledger "$WORK/ledger.jsonl" --record "$RECORD_ID" \
  --out "$WORK/bundle" --config "$BUNDLE_CONFIG" --skeleton "$BUNDLE_SKELETON" \
  --profile "$WORK/profile/profile.json" || exit 1
echo "COLD-START-DONE $WORK/bundle"
