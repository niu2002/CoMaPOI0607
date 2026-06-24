#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

export DATASET="${DATASET:-ca}"
export MODEL_ROOT="${MODEL_ROOT:-/mnt/workspace/models}"
export MODEL_NAME="${MODEL_NAME:-Llama-3.1-8B-Instruct}"
export SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-llama3.1-8b}"
export PORT="${PORT:-7863}"
export MODE="${MODE:-train}"
export RAG_TOP_K="${RAG_TOP_K:-100}"
export EMBEDDING_MODEL_PATH="${EMBEDDING_MODEL_PATH:-$MODEL_ROOT/Qwen3-Embedding-4B}"
export EMBEDDING_BATCH_SIZE="${EMBEDDING_BATCH_SIZE:-4}"
export INVERSE_NUM_SAMPLES="${INVERSE_NUM_SAMPLES:-0}"
export INVERSE_START_POINT="${INVERSE_START_POINT:-0}"
export INVERSE_WORKERS="${INVERSE_WORKERS:-1}"
export NUM_CANDIDATE="${NUM_CANDIDATE:-25}"
export PROFILE_MAX_TOKENS="${PROFILE_MAX_TOKENS:-220}"
export INVERSE_RRF_STYLE="${INVERSE_RRF_STYLE:-clean}"
export INVERSE_FUSION_STRATEGY="${INVERSE_FUSION_STRATEGY:-rrf}"
export FUSED_CANDIDATE_TOP_K="${FUSED_CANDIDATE_TOP_K:-50}"
export RRF_K="${RRF_K:-60}"
export RRF_WEIGHTS="${RRF_WEIGHTS:-}"
export HISTORY_CANDIDATE_K="${HISTORY_CANDIDATE_K:-30}"
export GEO_CANDIDATE_K="${GEO_CANDIDATE_K:-50}"
export CATEGORY_CANDIDATE_K="${CATEGORY_CANDIDATE_K:-50}"
export POPULAR_CANDIDATE_K="${POPULAR_CANDIDATE_K:-50}"
export FORCE_CANDIDATES="${FORCE_CANDIDATES:-0}"
export FORCE_AGENT_DATA="${FORCE_AGENT_DATA:-0}"

cd "$PROJECT_ROOT"

split_file="$PROJECT_ROOT/dataset_all/$DATASET/$MODE/${DATASET}_${MODE}.jsonl"
candidate_file="$PROJECT_ROOT/dataset_all/$DATASET/$MODE/${DATASET}_${MODE}_candidates.jsonl"
agent1_file="$PROJECT_ROOT/finetune/data/$DATASET/agent1_train_samples.jsonl"
agent2_file="$PROJECT_ROOT/finetune/data/$DATASET/agent2_train_samples.jsonl"
agent3_file="$PROJECT_ROOT/finetune/data/$DATASET/agent3_train_samples.jsonl"

count_lines() {
  local file_path="$1"
  FILE_PATH="$file_path" "$PYTHON_BIN" - <<'PY'
import os
from pathlib import Path

path = Path(os.environ["FILE_PATH"])
with path.open("r", encoding="utf-8") as f:
    print(sum(1 for _ in f))
PY
}

check_api() {
  "$PYTHON_BIN" - <<'PY'
import json
import os
import sys
import urllib.request

port = os.environ["PORT"]
url = f"http://127.0.0.1:{port}/v1/models"
try:
    with urllib.request.urlopen(url, timeout=5) as response:
        payload = json.loads(response.read().decode("utf-8"))
except Exception as exc:
    print(f"[prepare-agent-data] OpenAI-compatible service is not ready at {url}: {exc}")
    sys.exit(1)
model_ids = [item.get("id") for item in payload.get("data", [])]
print(f"[prepare-agent-data] service models: {model_ids}")
PY
}

files_ready() {
  [[ -s "$agent1_file" && -s "$agent2_file" && -s "$agent3_file" ]]
}

echo "[prepare-agent-data] project_root=$PROJECT_ROOT"
echo "[prepare-agent-data] dataset=$DATASET"
echo "[prepare-agent-data] mode=$MODE"
echo "[prepare-agent-data] port=$PORT"
echo "[prepare-agent-data] inverse_rrf_style=$INVERSE_RRF_STYLE"

"$PYTHON_BIN" "$SCRIPT_DIR/prepare_multiagent_assets.py" --dataset "$DATASET"

if [[ ! -f "$split_file" ]]; then
  echo "[prepare-agent-data] split file not found: $split_file"
  exit 1
fi

split_count="$(count_lines "$split_file")"
candidate_num_samples="$INVERSE_NUM_SAMPLES"
if [[ "$candidate_num_samples" == "0" ]]; then
  candidate_num_samples="$split_count"
fi

if [[ "$FORCE_CANDIDATES" == "1" || ! -s "$candidate_file" ]]; then
  echo "[prepare-agent-data] building RAG candidates for $candidate_num_samples samples"
  "$PYTHON_BIN" "$SCRIPT_DIR/build_candidates_amd.py" \
    --dataset "$DATASET" \
    --mode "$MODE" \
    --num_samples "$candidate_num_samples" \
    --top_k "$RAG_TOP_K" \
    --embedding_model_path "$EMBEDDING_MODEL_PATH" \
    --embedding_batch_size "$EMBEDDING_BATCH_SIZE"
else
  echo "[prepare-agent-data] keep existing candidates: $candidate_file"
fi

if [[ "$FORCE_AGENT_DATA" != "1" ]] && files_ready; then
  echo "[prepare-agent-data] agent training files already exist:"
  echo "  $agent1_file"
  echo "  $agent2_file"
  echo "  $agent3_file"
  exit 0
fi

check_api

echo "[prepare-agent-data] generating agent training data with inverse inference"
echo "[prepare-agent-data] inverse_num_samples=$INVERSE_NUM_SAMPLES start=$INVERSE_START_POINT workers=$INVERSE_WORKERS"
"$PYTHON_BIN" "$PROJECT_ROOT/inference_inverse_new.py" \
  --dataset "$DATASET" \
  --mode "$MODE" \
  --api_type "$SERVED_MODEL_NAME" \
  --port "$PORT" \
  --num_samples "$INVERSE_NUM_SAMPLES" \
  --start_point "$INVERSE_START_POINT" \
  --batch_size "$INVERSE_WORKERS" \
  --num_candidate "$NUM_CANDIDATE" \
  --profile_max_tokens "$PROFILE_MAX_TOKENS" \
  --inverse_rrf_style "$INVERSE_RRF_STYLE" \
  --inverse_fusion_strategy "$INVERSE_FUSION_STRATEGY" \
  --fused_candidate_top_k "$FUSED_CANDIDATE_TOP_K" \
  --rrf_k "$RRF_K" \
  --rrf_weights "$RRF_WEIGHTS" \
  --history_candidate_k "$HISTORY_CANDIDATE_K" \
  --geo_candidate_k "$GEO_CANDIDATE_K" \
  --category_candidate_k "$CATEGORY_CANDIDATE_K" \
  --popular_candidate_k "$POPULAR_CANDIDATE_K"

for output_file in "$agent1_file" "$agent2_file" "$agent3_file"; do
  if [[ ! -s "$output_file" ]]; then
    echo "[prepare-agent-data] expected non-empty output file not found: $output_file"
    exit 1
  fi
  echo "[prepare-agent-data] ready: $output_file ($(count_lines "$output_file") lines)"
done
