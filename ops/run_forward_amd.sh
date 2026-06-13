#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

DATASET="${DATASET:-ca}"
MODEL_NAME="${MODEL_NAME:-Llama-3.1-8B-Instruct}"
PORT="${PORT:-7863}"
MODE="${MODE:-test}"
NUM_SAMPLES="${NUM_SAMPLES:-10}"
START_POINT="${START_POINT:-0}"
TOP_K="${TOP_K:-10}"
NUM_CANDIDATE="${NUM_CANDIDATE:-25}"
FORWARD_WORKERS="${FORWARD_WORKERS:-1}"
TEST_INTERVAL="${TEST_INTERVAL:-10}"
OP_STR="${OP_STR:-amd-forward-smoke}"
PROFILE_MAX_TOKENS="${PROFILE_MAX_TOKENS:-220}"
CANDIDATE_FUSION_STRATEGY="${CANDIDATE_FUSION_STRATEGY:-none}"
FUSED_CANDIDATE_TOP_K="${FUSED_CANDIDATE_TOP_K:-50}"
RRF_K="${RRF_K:-60}"
RRF_WEIGHTS="${RRF_WEIGHTS:-}"
HISTORY_CANDIDATE_K="${HISTORY_CANDIDATE_K:-30}"
GEO_CANDIDATE_K="${GEO_CANDIDATE_K:-50}"
CATEGORY_CANDIDATE_K="${CATEGORY_CANDIDATE_K:-50}"
POPULAR_CANDIDATE_K="${POPULAR_CANDIDATE_K:-50}"

BASE_API_NAME="${BASE_API_NAME:-llama3.1-8b}"
AGENT1_API="${AGENT1_API:-$BASE_API_NAME}"
AGENT2_API="${AGENT2_API:-$BASE_API_NAME}"
AGENT3_API="${AGENT3_API:-$BASE_API_NAME}"
WAIT_FOR_MODELS="${WAIT_FOR_MODELS:-1}"
WAIT_TIMEOUT="${WAIT_TIMEOUT:-300}"
WAIT_INTERVAL="${WAIT_INTERVAL:-5}"

TEMPERATURE="${TEMPERATURE:-0.0}"
TOP_P="${TOP_P:-1.0}"
AGENT1_MAX_TOKENS="${AGENT1_MAX_TOKENS:-256}"
AGENT2_MAX_TOKENS="${AGENT2_MAX_TOKENS:-256}"
AGENT3_MAX_TOKENS="${AGENT3_MAX_TOKENS:-256}"

cd "$PROJECT_ROOT"

echo "[forward] project_root=$PROJECT_ROOT"
echo "[forward] dataset=$DATASET"
echo "[forward] model=$MODEL_NAME"
echo "[forward] mode=$MODE"
echo "[forward] num_samples=$NUM_SAMPLES"
echo "[forward] forward_workers=$FORWARD_WORKERS"
echo "[forward] port=$PORT"
echo "[forward] agent1_api=$AGENT1_API"
echo "[forward] agent2_api=$AGENT2_API"
echo "[forward] agent3_api=$AGENT3_API"
echo "[forward] candidate_fusion_strategy=$CANDIDATE_FUSION_STRATEGY"
echo "[forward] fused_candidate_top_k=$FUSED_CANDIDATE_TOP_K"

if [[ "$WAIT_FOR_MODELS" == "1" ]]; then
  EXPECTED_MODELS="$(printf '%s\n' "$AGENT1_API" "$AGENT2_API" "$AGENT3_API" | awk '!seen[$0]++')" \
  WAIT_TIMEOUT="$WAIT_TIMEOUT" \
  WAIT_INTERVAL="$WAIT_INTERVAL" \
  PORT="$PORT" \
  "$PYTHON_BIN" - <<'PY'
import json
import os
import sys
import time
import urllib.request

port = os.environ["PORT"]
wait_timeout = int(os.environ["WAIT_TIMEOUT"])
wait_interval = int(os.environ["WAIT_INTERVAL"])
expected_models = [line.strip() for line in os.environ["EXPECTED_MODELS"].splitlines() if line.strip()]
url = f"http://127.0.0.1:{port}/v1/models"
deadline = time.time() + wait_timeout
last_error = None

while time.time() < deadline:
    try:
        with urllib.request.urlopen(url, timeout=5) as response:
            payload = json.loads(response.read().decode("utf-8"))
        model_ids = [item.get("id") for item in payload.get("data", [])]
        missing = [model for model in expected_models if model not in model_ids]
        if not missing:
            print(f"[forward] confirmed service models: {model_ids}")
            break
        last_error = f"missing models: {missing}; available: {model_ids}"
    except Exception as exc:
        last_error = str(exc)
    time.sleep(wait_interval)
else:
    print(f"[forward] service did not become ready at {url} within {wait_timeout}s: {last_error}", file=sys.stderr)
    sys.exit(1)
PY
fi

exec "$PYTHON_BIN" "$PROJECT_ROOT/inference_forward_new.py" \
  --dataset "$DATASET" \
  --model "$MODEL_NAME" \
  --mode "$MODE" \
  --num_samples "$NUM_SAMPLES" \
  --start_point "$START_POINT" \
  --top_k "$TOP_K" \
  --num_candidate "$NUM_CANDIDATE" \
  --batch_size "$FORWARD_WORKERS" \
  --test_interval "$TEST_INTERVAL" \
  --port "$PORT" \
  --agent1_api "$AGENT1_API" \
  --agent2_api "$AGENT2_API" \
  --agent3_api "$AGENT3_API" \
  --agent1_max_tokens "$AGENT1_MAX_TOKENS" \
  --agent2_max_tokens "$AGENT2_MAX_TOKENS" \
  --agent3_max_tokens "$AGENT3_MAX_TOKENS" \
  --profile_max_tokens "$PROFILE_MAX_TOKENS" \
  --candidate_fusion_strategy "$CANDIDATE_FUSION_STRATEGY" \
  --fused_candidate_top_k "$FUSED_CANDIDATE_TOP_K" \
  --rrf_k "$RRF_K" \
  --rrf_weights "$RRF_WEIGHTS" \
  --history_candidate_k "$HISTORY_CANDIDATE_K" \
  --geo_candidate_k "$GEO_CANDIDATE_K" \
  --category_candidate_k "$CATEGORY_CANDIDATE_K" \
  --popular_candidate_k "$POPULAR_CANDIDATE_K" \
  --temperature "$TEMPERATURE" \
  --top_p "$TOP_P" \
  --op_str "$OP_STR" \
  "$@"
