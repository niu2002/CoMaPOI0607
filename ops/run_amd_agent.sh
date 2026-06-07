#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

AGENT_TYPE="${AGENT_TYPE:-agent1}"
DATASET="${DATASET:-ca}"
MODEL_ROOT="${MODEL_ROOT:-$PROJECT_ROOT/models}"
MODEL_NAME="${MODEL_NAME:-Llama-3.1-8B-Instruct}"
DEVICE_MAP="${DEVICE_MAP:-cuda}"
OP_STR="${OP_STR:-amd-agent}"

SEQ_LENGTH="${SEQ_LENGTH:-2048}"
BATCH_SIZE="${BATCH_SIZE:-16}"
GRAD_ACC="${GRAD_ACC:-1}"
MAX_STEPS="${MAX_STEPS:--1}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-1}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
NUM_WORKERS="${NUM_WORKERS:-0}"
TRAIN_NUM_SAMPLES="${TRAIN_NUM_SAMPLES:-0}"
LOG_FREQ="${LOG_FREQ:-5}"
SAVE_FREQ="${SAVE_FREQ:-10}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-3}"
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-}"

if [[ "$AGENT_TYPE" != "agent1" && "$AGENT_TYPE" != "agent2" && "$AGENT_TYPE" != "agent3" ]]; then
  echo "[train-agent] AGENT_TYPE must be one of: agent1, agent2, agent3"
  exit 1
fi

cd "$PROJECT_ROOT"

echo "[train-agent] project_root=$PROJECT_ROOT"
echo "[train-agent] dataset=$DATASET"
echo "[train-agent] agent_type=$AGENT_TYPE"
echo "[train-agent] model=$MODEL_NAME"
echo "[train-agent] batch_size=$BATCH_SIZE"
echo "[train-agent] max_steps=$MAX_STEPS"

exec "$PYTHON_BIN" "$PROJECT_ROOT/finetune_sft_new.py" \
  --dataset "$DATASET" \
  --model "$MODEL_NAME" \
  --model_path "$MODEL_ROOT" \
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
  --type "$AGENT_TYPE" \
  --op_str "$OP_STR" \
  --device_map "$DEVICE_MAP" \
  --resume_from_checkpoint "$RESUME_FROM_CHECKPOINT" \
  --gradient_checkpointing \
  --bf16 \
  "$@"
