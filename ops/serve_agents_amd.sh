#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PORT="${PORT:-7863}"
HOST="${HOST:-127.0.0.1}"
MODEL_ROOT="${MODEL_ROOT:-/mnt/workspace/comapoilatest/models}"
MODEL_NAME="${MODEL_NAME:-Llama-3.1-8B-Instruct}"
MODEL_PATH="${MODEL_PATH:-$MODEL_ROOT/$MODEL_NAME}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-llama3.1-8b}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
MAX_LORAS="${MAX_LORAS:-3}"
LOG_FILE="${LOG_FILE:-/tmp/comapoi-vllm-agents.log}"
BACKGROUND="${BACKGROUND:-1}"
WAIT_FOR_READY="${WAIT_FOR_READY:-1}"
WAIT_TIMEOUT="${WAIT_TIMEOUT:-300}"
WAIT_INTERVAL="${WAIT_INTERVAL:-5}"

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
  --max-num-seqs "${VLLM_MAX_NUM_SEQS:-16}"
  --max-model-len "${VLLM_MAX_MODEL_LEN:-8192}"
  --disable-log-stats
)

lora_modules=()
agent1_api="$SERVED_MODEL_NAME"
agent2_api="$SERVED_MODEL_NAME"
agent3_api="$SERVED_MODEL_NAME"

add_lora_module() {
  local agent_name="$1"
  local adapter_path="$2"
  if [[ -z "$adapter_path" ]]; then
    echo "[serve] $agent_name uses base model: $SERVED_MODEL_NAME"
    return
  fi
  if [[ ! -d "$adapter_path" ]]; then
    echo "[serve] adapter path for $agent_name not found: $adapter_path"
    exit 1
  fi
  echo "[serve] $agent_name uses LoRA adapter: $adapter_path"
  lora_modules+=("$agent_name=$adapter_path")
  case "$agent_name" in
    agent1) agent1_api="agent1" ;;
    agent2) agent2_api="agent2" ;;
    agent3) agent3_api="agent3" ;;
  esac
}

add_lora_module "agent1" "$AGENT1_ADAPTER_PATH"
add_lora_module "agent2" "$AGENT2_ADAPTER_PATH"
add_lora_module "agent3" "$AGENT3_ADAPTER_PATH"

expected_models=("$SERVED_MODEL_NAME")
if [[ "$agent1_api" != "$SERVED_MODEL_NAME" ]]; then
  expected_models+=("$agent1_api")
fi
if [[ "$agent2_api" != "$SERVED_MODEL_NAME" ]]; then
  expected_models+=("$agent2_api")
fi
if [[ "$agent3_api" != "$SERVED_MODEL_NAME" ]]; then
  expected_models+=("$agent3_api")
fi

if (( ${#lora_modules[@]} > 0 )); then
  echo "[serve] enabling LoRA adapters: ${lora_modules[*]}"
  base_cmd+=(
    --enable-lora
    --max-loras "$MAX_LORAS"
    --lora-modules
    "${lora_modules[@]}"
  )
else
  echo "[serve] no adapter paths provided, serving base model only"
fi

echo "[serve] forward run should use:"
echo "  AGENT1_API=$agent1_api"
echo "  AGENT2_API=$agent2_api"
echo "  AGENT3_API=$agent3_api"

echo "[serve] log_file=$LOG_FILE"
echo "[serve] model_path=$MODEL_PATH"
echo "[serve] port=$PORT"

wait_for_models() {
  EXPECTED_MODELS="$(printf '%s\n' "${expected_models[@]}")" \
  WAIT_TIMEOUT="$WAIT_TIMEOUT" \
  WAIT_INTERVAL="$WAIT_INTERVAL" \
  HOST="$HOST" \
  PORT="$PORT" \
  "$PYTHON_BIN" - <<'PY'
import json
import os
import sys
import time
import urllib.request

host = os.environ["HOST"]
port = os.environ["PORT"]
wait_timeout = int(os.environ["WAIT_TIMEOUT"])
wait_interval = int(os.environ["WAIT_INTERVAL"])
expected_models = [line.strip() for line in os.environ["EXPECTED_MODELS"].splitlines() if line.strip()]
url = f"http://{host}:{port}/v1/models"
deadline = time.time() + wait_timeout
last_error = None

while time.time() < deadline:
    try:
        with urllib.request.urlopen(url, timeout=5) as response:
            payload = json.loads(response.read().decode("utf-8"))
        model_ids = [item.get("id") for item in payload.get("data", [])]
        missing = [model for model in expected_models if model not in model_ids]
        if not missing:
            print(f"[serve] ready models: {model_ids}")
            sys.exit(0)
        last_error = f"missing models: {missing}; available: {model_ids}"
    except Exception as exc:
        last_error = str(exc)
    time.sleep(wait_interval)

print(f"[serve] service did not become ready at {url} within {wait_timeout}s: {last_error}", file=sys.stderr)
sys.exit(1)
PY
}

if [[ "$BACKGROUND" == "1" ]]; then
  nohup "${base_cmd[@]}" >"$LOG_FILE" 2>&1 &
  echo "[serve] started background pid=$!"
  echo "[serve] tail -f $LOG_FILE"
  if [[ "$WAIT_FOR_READY" == "1" ]]; then
    echo "[serve] waiting for models: ${expected_models[*]}"
    wait_for_models
  fi
else
  exec "${base_cmd[@]}"
fi
