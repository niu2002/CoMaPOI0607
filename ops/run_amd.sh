#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
MODE="${1:-}"

if [ -z "$MODE" ]; then
  echo "Usage: $0 <train|smoke|eval> [extra args...]"
  exit 1
fi

shift || true

DATASET="${DATASET:-ca}"
MODEL_ROOT="${MODEL_ROOT:-$PROJECT_ROOT/models}"
MODEL_NAME="${MODEL_NAME:-Llama-3.1-8B-Instruct}"
BASE_MODEL_PATH="${BASE_MODEL_PATH:-$MODEL_ROOT/$MODEL_NAME}"
DATA_ROOT="${DATA_ROOT:-$PROJECT_ROOT/dataset_all}"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH:-$DATA_ROOT/${DATASET}_train.jsonl}"
TEST_DATA_PATH="${TEST_DATA_PATH:-$DATA_ROOT/${DATASET}_test.jsonl}"

OP_STR="${OP_STR:-amd-debug}"
SEQ_LENGTH="${SEQ_LENGTH:-2048}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACC="${GRAD_ACC:-4}"
MAX_STEPS="${MAX_STEPS:-100}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
NUM_WORKERS="${NUM_WORKERS:-0}"
TOP_K="${TOP_K:-5}"
NUM_BEAMS="${NUM_BEAMS:-5}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-64}"
TEMPERATURE="${TEMPERATURE:-0.0}"
TOP_P="${TOP_P:-1.0}"
SAMPLE_INDEX="${SAMPLE_INDEX:-0}"
START_INDEX="${START_INDEX:-0}"
NUM_SAMPLES="${NUM_SAMPLES:-20}"
SAVE_INTERVAL="${SAVE_INTERVAL:-10}"

SAVE_NAME="bs${BATCH_SIZE}-gas${GRAD_ACC}-ms${MAX_STEPS}-merged-lr${LEARNING_RATE}"
DEFAULT_ADAPTER_PATH="$PROJECT_ROOT/finetune/results/$OP_STR/sft-$DATASET/$SAVE_NAME"
ADAPTER_PATH="${ADAPTER_PATH:-$DEFAULT_ADAPTER_PATH}"

cd "$PROJECT_ROOT"

echo "[run] project_root=$PROJECT_ROOT"
echo "[run] mode=$MODE"
echo "[run] python=$PYTHON_BIN"
echo "[run] dataset=$DATASET"
echo "[run] base_model_path=$BASE_MODEL_PATH"

case "$MODE" in
  train)
    exec "$PYTHON_BIN" "$PROJECT_ROOT/finetune_sft_new.py" \
      --dataset "$DATASET" \
      --model "$MODEL_NAME" \
      --model_path "$MODEL_ROOT" \
      --data_path "$TRAIN_DATA_PATH" \
      --seq_length "$SEQ_LENGTH" \
      --batch_size "$BATCH_SIZE" \
      --gradient_accumulation_steps "$GRAD_ACC" \
      --num_workers "$NUM_WORKERS" \
      --max_steps "$MAX_STEPS" \
      --learning_rate "$LEARNING_RATE" \
      --type merged \
      --op_str "$OP_STR" \
      --gradient_checkpointing \
      --bf16 \
      "$@"
    ;;
  smoke)
    exec "$PYTHON_BIN" "$PROJECT_ROOT/lora_inference_smoke.py" \
      --dataset "$DATASET" \
      --data_path "$TEST_DATA_PATH" \
      --base_model_path "$BASE_MODEL_PATH" \
      --adapter_path "$ADAPTER_PATH" \
      --sample_index "$SAMPLE_INDEX" \
      --max_new_tokens "$MAX_NEW_TOKENS" \
      --temperature "$TEMPERATURE" \
      --top_p "$TOP_P" \
      "$@"
    ;;
  eval)
    exec "$PYTHON_BIN" "$PROJECT_ROOT/lora_batch_inference_eval.py" \
      --dataset "$DATASET" \
      --data_path "$TEST_DATA_PATH" \
      --base_model_path "$BASE_MODEL_PATH" \
      --adapter_path "$ADAPTER_PATH" \
      --start_index "$START_INDEX" \
      --num_samples "$NUM_SAMPLES" \
      --top_k "$TOP_K" \
      --num_beams "$NUM_BEAMS" \
      --max_new_tokens "$MAX_NEW_TOKENS" \
      --temperature "$TEMPERATURE" \
      --top_p "$TOP_P" \
      --save_interval "$SAVE_INTERVAL" \
      "$@"
    ;;
  *)
    echo "Unknown mode: $MODE"
    echo "Usage: $0 <train|smoke|eval> [extra args...]"
    exit 1
    ;;
esac
