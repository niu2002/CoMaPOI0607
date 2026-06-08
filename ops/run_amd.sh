#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
MODE="${1:-}"
DEVICE_MAP="${DEVICE_MAP:-cuda}"

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
BATCH_SIZE="${BATCH_SIZE:-40}"
GRAD_ACC="${GRAD_ACC:-1}"
MAX_STEPS="${MAX_STEPS:--1}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-1}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
NUM_WORKERS="${NUM_WORKERS:-0}"
TRAIN_NUM_SAMPLES="${TRAIN_NUM_SAMPLES:-0}"
LOG_FREQ="${LOG_FREQ:-5}"
SAVE_FREQ="${SAVE_FREQ:-10}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-3}"
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
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-}"
AGENT_TEST_SIZE="${AGENT_TEST_SIZE:-100}"

if [ "$RESUME_FROM_CHECKPOINT" = "latest" ]; then
  latest_checkpoint="$(find "$DEFAULT_ADAPTER_PATH" -maxdepth 1 -type d -name 'checkpoint-*' | sort -V | tail -n 1 || true)"
  if [ -z "$latest_checkpoint" ]; then
    echo "[run] resume requested but no checkpoint found under $DEFAULT_ADAPTER_PATH"
    exit 1
  fi
  RESUME_FROM_CHECKPOINT="$latest_checkpoint"
fi

cd "$PROJECT_ROOT"

echo "[run] project_root=$PROJECT_ROOT"
echo "[run] mode=$MODE"
echo "[run] python=$PYTHON_BIN"
echo "[run] dataset=$DATASET"
echo "[run] base_model_path=$BASE_MODEL_PATH"
echo "[run] device_map=$DEVICE_MAP"
echo "[run] batch_size=$BATCH_SIZE"
echo "[run] grad_acc=$GRAD_ACC"
echo "[run] max_steps=$MAX_STEPS"
echo "[run] num_train_epochs=$NUM_TRAIN_EPOCHS"
echo "[run] save_freq=$SAVE_FREQ"
echo "[run] resume_from_checkpoint=${RESUME_FROM_CHECKPOINT:-<none>}"

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
      --num_samples "$TRAIN_NUM_SAMPLES" \
      --max_steps "$MAX_STEPS" \
      --num_train_epochs "$NUM_TRAIN_EPOCHS" \
      --learning_rate "$LEARNING_RATE" \
      --log_freq "$LOG_FREQ" \
      --save_freq "$SAVE_FREQ" \
      --save_total_limit "$SAVE_TOTAL_LIMIT" \
      --type merged \
      --agent_test_size "$AGENT_TEST_SIZE" \
      --op_str "$OP_STR" \
      --device_map "$DEVICE_MAP" \
      --resume_from_checkpoint "$RESUME_FROM_CHECKPOINT" \
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
      --device_map "$DEVICE_MAP" \
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
      --device_map "$DEVICE_MAP" \
      "$@"
    ;;
  *)
    echo "Unknown mode: $MODE"
    echo "Usage: $0 <train|smoke|eval> [extra args...]"
    exit 1
    ;;
esac
