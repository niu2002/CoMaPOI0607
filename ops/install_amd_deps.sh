#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
PIP_INDEX_URL="${PIP_INDEX_URL:-https://mirrors.aliyun.com/pypi/simple/}"
PIP_TRUSTED_HOST="${PIP_TRUSTED_HOST:-mirrors.aliyun.com}"

echo "[deps] python=$("$PYTHON_BIN" -V)"
echo "[deps] index=$PIP_INDEX_URL"
echo "[deps] installing project deps without reinstalling torch/ROCm or bitsandbytes"

"$PYTHON_BIN" -m pip install \
  -i "$PIP_INDEX_URL" \
  --trusted-host "$PIP_TRUSTED_HOST" \
  --root-user-action=ignore \
  "transformers>=4.30.0" \
  "peft>=0.4.0" \
  "trl>=0.7.1" \
  "datasets>=2.14.0" \
  "agentscope==0.1.6" \
  "tqdm>=4.65.0" \
  "pandas>=1.5.3" \
  "numpy>=1.24.3" \
  "faiss-cpu>=1.7.4" \
  "jsonlines>=3.1.0" \
  "loguru==0.6.0" \
  "openai>=1.1.0" \
  "accelerate>=0.20.3" \
  "scipy>=1.10.1" \
  "scikit-learn>=1.2.2"

"$PYTHON_BIN" - <<'PY'
import agentscope
import accelerate
import datasets
import jsonlines
import openai
import peft
import sklearn
import transformers
import trl

print("deps_ok")
print("trl", getattr(trl, "__version__", "unknown"))
print("agentscope", getattr(agentscope, "__version__", "unknown"))
PY
