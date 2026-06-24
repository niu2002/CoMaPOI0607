#!/usr/bin/env bash
set -euo pipefail

MODEL_ID="${MODEL_ID:-Qwen/Qwen3-Embedding-4B}"
TARGET_DIR="${TARGET_DIR:-/mnt/workspace/models/Qwen3-Embedding-4B}"
PYTHON_BIN="${PYTHON_BIN:-python}"
export MODEL_ID TARGET_DIR

echo "[download] model_id=$MODEL_ID"
echo "[download] target_dir=$TARGET_DIR"

exec "$PYTHON_BIN" - <<'PY'
import os
from modelscope import snapshot_download

model_id = os.environ.get("MODEL_ID", "Qwen/Qwen3-Embedding-4B")
target_dir = os.environ.get("TARGET_DIR", "/mnt/workspace/models/Qwen3-Embedding-4B")

kwargs = {"model_id": model_id, "local_dir": target_dir}
try:
    snapshot_download(local_dir_use_symlinks=False, **kwargs)
except TypeError:
    print("[download] modelscope version does not support local_dir_use_symlinks, retrying without it")
    snapshot_download(**kwargs)
print(f"[download] completed: {target_dir}")
PY
