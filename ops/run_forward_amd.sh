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

BASE_API_NAME="${BASE_API_NAME:-llama3.1-8b}"
AGENT1_API="${AGENT1_API:-$BASE_API_NAME}"
AGENT2_API="${AGENT2_API:-$BASE_API_NAME}"
AGENT3_API="${AGENT3_API:-$BASE_API_NAME}"

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
  --temperature "$TEMPERATURE" \
  --top_p "$TOP_P" \
  --op_str "$OP_STR" \
  "$@"
