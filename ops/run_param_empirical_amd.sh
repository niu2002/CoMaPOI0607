#!/usr/bin/env bash
# ==============================================================================
# One-Click Real Empirical Parameter Sensitivity Suite on AMD ROCm GPU
# ==============================================================================
# 1. Starts vLLM with fine-tuned LoRA adapters (Agent 3 LoRA)
# 2. Runs high-concurrency (8 workers) parameter sweep on 100 genuine test samples
# 3. Flushes real empirical CSVs for K, mu, Kc, and L_prof across CA, NYC, TKY
# 4. Automatically plots Figure 2 (fig_param_sensitivity_v2.png / .pdf)
# ==============================================================================

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

PORT="${PORT:-7863}"
HOST="${HOST:-127.0.0.1}"
MODEL_ROOT="${MODEL_ROOT:-/mnt/workspace/models}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-8B}"
MODEL_PATH="${MODEL_PATH:-$MODEL_ROOT/$MODEL_NAME}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3-8b}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.78}"
NUM_SAMPLES="${NUM_SAMPLES:-100}"
CONCURRENCY="${CONCURRENCY:-8}"
PARAM="${PARAM:-kc,lprof}"
DATASET="${DATASET:-all}"

echo "================================================================================"
echo "  Starting Authentic Empirical Parameter Evaluation on AMD GPU"
echo "================================================================================"
echo "  Project Root: $PROJECT_ROOT"
echo "  Base Model:   $MODEL_PATH"
echo "  Test Samples: $NUM_SAMPLES per dataset (Paired Benchmark)"
echo "  Concurrency:  $CONCURRENCY threads"
echo "================================================================================"

# 1. Stop any stale vLLM processes
echo ">>> Checking and terminating stale vLLM processes..."
pkill -f "vllm.entrypoints.openai.api_server" || true
sleep 2

# 2. Locate LoRA Adapter Checkpoints
CA_LORA="$PROJECT_ROOT/finetune/results/amd-agent3-ca-lora-v1/sft-ca/bs16-gas1-ms-1-agent3-lr0.0001"
NYC_LORA="$PROJECT_ROOT/finetune/results/amd-agent3-nyc-lora-v1/sft-nyc/bs4-gas2-ms-1-agent3-lr0.0001"

# Check alternative directory if needed
if [[ ! -d "$CA_LORA" ]]; then
  FOUND_CA=$(find "$PROJECT_ROOT/finetune/results" -type d -path "*agent3*ca*" | head -n 1 || true)
  if [[ -n "$FOUND_CA" && -d "$FOUND_CA" ]]; then
    CA_LORA="$FOUND_CA"
  fi
fi

if [[ ! -d "$NYC_LORA" ]]; then
  FOUND_NYC=$(find "$PROJECT_ROOT/finetune/results" -type d -path "*agent3*nyc*" | head -n 1 || true)
  if [[ -n "$FOUND_NYC" && -d "$FOUND_NYC" ]]; then
    NYC_LORA="$FOUND_NYC"
  fi
fi

LORA_MODULES=()
if [[ -d "$CA_LORA" ]]; then
  echo ">>> [LORA FOUND] CA Adapter:  $CA_LORA"
  LORA_MODULES+=("agent3-ca=$CA_LORA" "agent3=$CA_LORA")
else
  echo ">>> [WARN] CA LoRA adapter not found at $CA_LORA, running base model"
fi

if [[ -d "$NYC_LORA" ]]; then
  echo ">>> [LORA FOUND] NYC Adapter: $NYC_LORA"
  LORA_MODULES+=("agent3-nyc=$NYC_LORA")
fi

LORA_ARGS=()
if (( ${#LORA_MODULES[@]} > 0 )); then
  LORA_ARGS=(--enable-lora --max-loras 4 --lora-modules "${LORA_MODULES[@]}")
fi

# 3. Start vLLM OpenAI-Compatible Server
LOG_FILE="/tmp/comapoi-vllm-agents.log"
echo ">>> Launching vLLM server on port $PORT (log: $LOG_FILE)..."
python -m vllm.entrypoints.openai.api_server \
  --host "$HOST" \
  --port "$PORT" \
  --model "$MODEL_PATH" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --tensor-parallel-size 1 \
  --dtype auto \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --max-model-len 4096 \
  "${LORA_ARGS[@]}" \
  > "$LOG_FILE" 2>&1 &

VLLM_PID=$!
echo ">>> vLLM launched with PID: $VLLM_PID. Waiting for endpoint readiness..."

# Wait for readiness
READY=0
for i in {1..60}; do
  if curl -s "http://$HOST:$PORT/v1/models" >/dev/null 2>&1; then
    READY=1
    break
  fi
  sleep 4
  echo -n "."
done
echo ""

if [[ "$READY" -ne 1 ]]; then
  echo "[ERROR] vLLM failed to start within timeout. Check log: $LOG_FILE"
  tail -n 40 "$LOG_FILE"
  exit 1
fi

echo ">>> vLLM is READY! Registered models:"
curl -s "http://$HOST:$PORT/v1/models" | python -m json.tool || true

# 4. Run High-Concurrency Empirical Parameter Evaluation
echo ""
echo "================================================================================"
echo "  Running Empirical Parameter Sweeps (Samples: $NUM_SAMPLES, Concurrency: $CONCURRENCY)"
echo "================================================================================"

# Ensure scikit-learn is installed for clustering
python -m pip install scikit-learn -q || true

python eval_param_empirical_all.py \
  --dataset "$DATASET" \
  --param "$PARAM" \
  --num_samples "$NUM_SAMPLES" \
  --concurrency "$CONCURRENCY" \
  --vllm_url "http://$HOST:$PORT/v1" \
  --plot

echo ""
echo "================================================================================"
echo "  [ALL DONE] Empirical Parameter Evaluation Completed Successfully!"
echo "================================================================================"
echo "  Output CSVs in:      $PROJECT_ROOT/experiments/param_sensitivity/"
echo "  Publication Figures: $PROJECT_ROOT/figures/fig_param_sensitivity_v2.png"
echo "                       $PROJECT_ROOT/figures/fig_param_sensitivity.pdf"
echo "================================================================================"
