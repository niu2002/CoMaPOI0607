#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

export DATASET="${DATASET:-ca}"
export MODEL_ROOT="${MODEL_ROOT:-/mnt/workspace/comapoilatest/models}"
export MODEL_NAME="${MODEL_NAME:-Llama-3.1-8B-Instruct}"
export DEVICE_MAP="${DEVICE_MAP:-cuda}"
export OP_STR="${OP_STR:-amd-agent23-ca-v1}"
export SEQ_LENGTH="${SEQ_LENGTH:-2048}"
export BATCH_SIZE="${BATCH_SIZE:-16}"
export GRAD_ACC="${GRAD_ACC:-1}"
export MAX_STEPS="${MAX_STEPS:--1}"
export NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-1}"
export LEARNING_RATE="${LEARNING_RATE:-1e-4}"
export NUM_WORKERS="${NUM_WORKERS:-0}"
export TRAIN_NUM_SAMPLES="${TRAIN_NUM_SAMPLES:-0}"
export LOG_FREQ="${LOG_FREQ:-5}"
export SAVE_FREQ="${SAVE_FREQ:-50}"
export SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-3}"
export TRAIN_AGENT2="${TRAIN_AGENT2:-1}"
export TRAIN_AGENT3="${TRAIN_AGENT3:-1}"
export REQUIRE_AGENT_DATA="${REQUIRE_AGENT_DATA:-1}"
export KILL_VLLM_BEFORE_TRAIN="${KILL_VLLM_BEFORE_TRAIN:-1}"
export AGENT_TEST_SIZE="${AGENT_TEST_SIZE:-100}"
export FORCE_REPROCESS_AGENT_DATA="${FORCE_REPROCESS_AGENT_DATA:-0}"

PYTHON_BIN="${PYTHON_BIN:-python}"
PATH_ENV_FILE="${PATH_ENV_FILE:-$PROJECT_ROOT/finetune/results/$OP_STR/sft-$DATASET/agent23_paths.env}"

resolve_adapter_path() {
  local agent_type="$1"
  AGENT_TYPE_FOR_PATH="$agent_type" "$PYTHON_BIN" - <<'PY'
import os
from pathlib import Path

project_root = Path(os.environ["PROJECT_ROOT"]).resolve()
dataset = os.environ["DATASET"]
op_str = os.environ["OP_STR"]
agent_type = os.environ["AGENT_TYPE_FOR_PATH"]
batch_size = int(os.environ["BATCH_SIZE"])
grad_acc = int(os.environ["GRAD_ACC"])
max_steps = int(os.environ["MAX_STEPS"])
learning_rate = float(os.environ["LEARNING_RATE"])
save_name = f"bs{batch_size}-gas{grad_acc}-ms{max_steps}-{agent_type}-lr{learning_rate}"
print(project_root / "finetune" / "results" / op_str / f"sft-{dataset}" / save_name)
PY
}

latest_checkpoint() {
  local adapter_path="$1"
  if [[ ! -d "$adapter_path" ]]; then
    return 0
  fi
  find "$adapter_path" -maxdepth 1 -type d -name 'checkpoint-*' -printf '%f\n' \
    | sort -t- -k2,2n \
    | tail -n 1 \
    | sed "s#^#$adapter_path/#"
}

train_one_agent() {
  local agent_type="$1"
  local resume_env="$2"
  local adapter_path
  local resume_value

  adapter_path="$(resolve_adapter_path "$agent_type")"
  resume_value="${!resume_env:-}"

  if [[ "$resume_value" == "latest" ]]; then
    resume_value="$(latest_checkpoint "$adapter_path")"
    if [[ -z "$resume_value" ]]; then
      echo "[agent23-train] no checkpoint found for $agent_type; starting from scratch"
    else
      echo "[agent23-train] resolved latest checkpoint for $agent_type: $resume_value"
    fi
  fi

  echo "[agent23-train] training $agent_type"
  echo "[agent23-train] expected adapter path: $adapter_path"

  if ! AGENT_TYPE="$agent_type" RESUME_FROM_CHECKPOINT="$resume_value" "$SCRIPT_DIR/run_amd_agent.sh"; then
    echo "[agent23-train] $agent_type training failed; fix the error above, then rerun this script."
    exit 1
  fi

  if [[ ! -d "$adapter_path" ]]; then
    echo "[agent23-train] expected adapter path not found after training: $adapter_path"
    exit 1
  fi
}

export PROJECT_ROOT
cd "$PROJECT_ROOT"

echo "[agent23-train] project_root=$PROJECT_ROOT"
echo "[agent23-train] dataset=$DATASET"
echo "[agent23-train] op_str=$OP_STR"
echo "[agent23-train] batch_size=$BATCH_SIZE"
echo "[agent23-train] max_steps=$MAX_STEPS"
echo "[agent23-train] save_freq=$SAVE_FREQ"
echo "[agent23-train] agent_test_size=$AGENT_TEST_SIZE"

if [[ "$KILL_VLLM_BEFORE_TRAIN" == "1" ]]; then
  if pgrep -f "vllm.entrypoints.openai.api_server" >/dev/null 2>&1; then
    echo "[agent23-train] stopping vLLM before training to free GPU memory"
    pkill -f "vllm.entrypoints.openai.api_server" || true
    sleep 3
  fi
fi

if [[ "$REQUIRE_AGENT_DATA" == "1" ]]; then
  missing_agent_data=()
  for agent_type in agent2 agent3; do
    agent_data_file="$PROJECT_ROOT/finetune/data/$DATASET/${agent_type}_train_samples.jsonl"
    if [[ ! -s "$agent_data_file" ]]; then
      missing_agent_data+=("$agent_data_file")
    fi
  done
  if (( ${#missing_agent_data[@]} > 0 )); then
    echo "[agent23-train] missing agent training data:"
    printf '  %s\n' "${missing_agent_data[@]}"
    echo "[agent23-train] prepare them first with:"
    echo "  bash ./ops/prepare_agent_training_data_amd.sh"
    exit 1
  fi
fi

if [[ "$FORCE_REPROCESS_AGENT_DATA" == "1" ]]; then
  echo "[agent23-train] removing cached split files before reprocessing agent data"
  rm -f \
    "$PROJECT_ROOT/finetune/data/$DATASET/agent2_train_samples_train_ts${AGENT_TEST_SIZE}.jsonl" \
    "$PROJECT_ROOT/finetune/data/$DATASET/agent2_train_samples_holdout_ts${AGENT_TEST_SIZE}.jsonl" \
    "$PROJECT_ROOT/finetune/data/$DATASET/agent3_train_samples_train_ts${AGENT_TEST_SIZE}.jsonl" \
    "$PROJECT_ROOT/finetune/data/$DATASET/agent3_train_samples_holdout_ts${AGENT_TEST_SIZE}.jsonl"
fi

if [[ "$TRAIN_AGENT2" == "1" ]]; then
  train_one_agent "agent2" "RESUME_AGENT2_FROM_CHECKPOINT"
else
  echo "[agent23-train] skipping agent2 because TRAIN_AGENT2=$TRAIN_AGENT2"
fi

if [[ "$TRAIN_AGENT3" == "1" ]]; then
  train_one_agent "agent3" "RESUME_AGENT3_FROM_CHECKPOINT"
else
  echo "[agent23-train] skipping agent3 because TRAIN_AGENT3=$TRAIN_AGENT3"
fi

agent2_path="$(resolve_adapter_path agent2)"
agent3_path="$(resolve_adapter_path agent3)"
mkdir -p "$(dirname "$PATH_ENV_FILE")"
cat >"$PATH_ENV_FILE" <<EOF
export AGENT2_ADAPTER_PATH="$agent2_path"
export AGENT3_ADAPTER_PATH="$agent3_path"
export AGENT1_API="${SERVED_MODEL_NAME:-llama3.1-8b}"
export AGENT2_API="agent2"
export AGENT3_API="agent3"
export OP_STR_AGENT23="$OP_STR"
EOF

echo "[agent23-train] wrote adapter env file: $PATH_ENV_FILE"
echo "[agent23-train] next commands:"
echo "  source \"$PATH_ENV_FILE\""
echo "  bash ./ops/serve_agent2_agent3_amd.sh"
echo "  bash ./ops/run_forward_101_agent23_amd.sh"
