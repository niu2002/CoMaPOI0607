#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PORT="${PORT:-7863}"
HOST="${HOST:-127.0.0.1}"
MODEL_ROOT="${MODEL_ROOT:-$PROJECT_ROOT/models}"
MODEL_NAME="${MODEL_NAME:-Llama-3.1-8B-Instruct}"
MODEL_PATH="${MODEL_PATH:-$MODEL_ROOT/$MODEL_NAME}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-llama3.1-8b}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
MAX_LORAS="${MAX_LORAS:-3}"
LOG_FILE="${LOG_FILE:-/tmp/comapoi-vllm-agents.log}"
BACKGROUND="${BACKGROUND:-1}"

AGENT1_ADAPTER_PATH="${AGENT1_ADAPTER_PATH:-}"
AGENT2_ADAPTER_PATH="${AGENT2_ADAPTER_PATH:-}"
AGENT3_ADAPTER_PATH="${AGENT3_ADAPTER_PATH:-}"

if ! command -v python >/dev/null 2>&1; then
  echo "[serve] python not found"
  exit 1
fi

PYTHON_BIN="${PYTHON_BIN:-python}"

base_cmd=(
  "$PYTHON_BIN" -m vllm.entrypoints.openai.api_server
  --host "$HOST"
  --port "$PORT"
  --model "$MODEL_PATH"
  --served-model-name "$SERVED_MODEL_NAME"
  --tensor-parallel-size 1
  --dtype auto
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
  --disable-log-stats
)

if [[ -n "$AGENT1_ADAPTER_PATH" && -n "$AGENT2_ADAPTER_PATH" && -n "$AGENT3_ADAPTER_PATH" ]]; then
  echo "[serve] enabling LoRA adapters for agent1/agent2/agent3"
  base_cmd+=(
    --enable-lora
    --max-loras "$MAX_LORAS"
    --lora-modules
    "agent1=$AGENT1_ADAPTER_PATH"
    "agent2=$AGENT2_ADAPTER_PATH"
    "agent3=$AGENT3_ADAPTER_PATH"
  )
  echo "[serve] forward run should use:"
  echo "  AGENT1_API=agent1"
  echo "  AGENT2_API=agent2"
  echo "  AGENT3_API=agent3"
else
  echo "[serve] adapter paths not fully provided, serving base model only"
  echo "[serve] forward smoke can still run with:"
  echo "  AGENT1_API=$SERVED_MODEL_NAME"
  echo "  AGENT2_API=$SERVED_MODEL_NAME"
  echo "  AGENT3_API=$SERVED_MODEL_NAME"
fi

echo "[serve] log_file=$LOG_FILE"
echo "[serve] model_path=$MODEL_PATH"
echo "[serve] port=$PORT"

if [[ "$BACKGROUND" == "1" ]]; then
  nohup "${base_cmd[@]}" >"$LOG_FILE" 2>&1 &
  echo "[serve] started background pid=$!"
  echo "[serve] tail -f $LOG_FILE"
else
  exec "${base_cmd[@]}"
fi
