#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

export DATASET="${DATASET:-ca}"
export MODEL_ROOT="${MODEL_ROOT:-/mnt/workspace/comapoilatest/models}"
export MODEL_NAME="${MODEL_NAME:-Llama-3.1-8B-Instruct}"
export SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-llama3.1-8b}"
export PORT="${PORT:-7863}"
export OP_STR="${OP_STR:-${OP_STR_AGENT23:-amd-agent23-ca-v1}}"
export BATCH_SIZE="${BATCH_SIZE:-16}"
export GRAD_ACC="${GRAD_ACC:-1}"
export MAX_STEPS="${MAX_STEPS:--1}"
export LEARNING_RATE="${LEARNING_RATE:-1e-4}"
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"

resolve_adapter_path() {
  local agent_type="$1"
  AGENT_TYPE_FOR_PATH="$agent_type" PROJECT_ROOT="$PROJECT_ROOT" "$PYTHON_BIN" - <<'PY'
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

export AGENT1_ADAPTER_PATH="${AGENT1_ADAPTER_PATH:-}"
export AGENT2_ADAPTER_PATH="${AGENT2_ADAPTER_PATH:-$(resolve_adapter_path agent2)}"
export AGENT3_ADAPTER_PATH="${AGENT3_ADAPTER_PATH:-$(resolve_adapter_path agent3)}"

for pair in "agent2:$AGENT2_ADAPTER_PATH" "agent3:$AGENT3_ADAPTER_PATH"; do
  agent_name="${pair%%:*}"
  adapter_path="${pair#*:}"
  if [[ ! -d "$adapter_path" ]]; then
    echo "[agent23-serve] $agent_name adapter path not found: $adapter_path"
    echo "[agent23-serve] train first with: bash ./ops/train_agent2_agent3_amd.sh"
    exit 1
  fi
done

echo "[agent23-serve] agent1=base:$SERVED_MODEL_NAME"
echo "[agent23-serve] agent2=LoRA:$AGENT2_ADAPTER_PATH"
echo "[agent23-serve] agent3=LoRA:$AGENT3_ADAPTER_PATH"

"$SCRIPT_DIR/serve_agents_amd.sh"

echo "[agent23-serve] forward env:"
echo "  AGENT1_API=$SERVED_MODEL_NAME"
echo "  AGENT2_API=agent2"
echo "  AGENT3_API=agent3"
