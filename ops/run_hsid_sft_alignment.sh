#!/usr/bin/env bash
# Automate the 4-phase plan to align HSID SFT dataset and evaluate Agent 3

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

echo "========================================================================="
echo "=== Phase 1: Generating Agent 3 Training Data (HSID + paper_label_first_fused) ==="
echo "========================================================================="
export INVERSE_RRF_STYLE=paper_label_first_fused
export FORCE_AGENT_DATA=1
export INVERSE_WORKERS=1
export INVERSE_NUM_SAMPLES=300
bash ops/prepare_agent_training_data_amd.sh --use_hsid

echo "========================================================================="
echo "=== Phase 2: Fine-Tuning Agent 3 LoRA Model ==="
echo "========================================================================="
export OP_STR=amd-agent3-full-hsid
export BATCH_SIZE=16
export FORCE_REPROCESS_AGENT_DATA=1
bash ops/train_agent3_amd.sh

echo "========================================================================="
echo "=== Phase 3: Loading Environment and Restarting vLLM Server ==="
echo "========================================================================="
# Kill old vLLM to release GPU memory
if pgrep -f "vllm.entrypoints.openai.api_server" >/dev/null 2>&1; then
  echo "Stopping running vLLM API server..."
  pkill -f "vllm.entrypoints.openai.api_server" || true
  sleep 3
fi

# Load path environment and launch server
source finetune/results/amd-agent3-full-hsid/sft-ca/agent3_path.env
export GPU_MEMORY_UTILIZATION=0.78
bash ops/serve_agent3_amd.sh

echo "========================================================================="
echo "=== Phase 4: Running Forward Evaluation (101 samples) ==="
echo "========================================================================="
export FORWARD_WORKERS=1
export NUM_SAMPLES=101
bash ops/run_forward_101_agent3_amd.sh --use_hsid

echo "========================================================================="
echo "=== Execution Complete! ==="
echo "========================================================================="
