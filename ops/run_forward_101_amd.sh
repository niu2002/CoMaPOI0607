#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export DATASET="${DATASET:-ca}"
export MODEL_NAME="${MODEL_NAME:-Llama-3.1-8B-Instruct}"
export PORT="${PORT:-7863}"
export NUM_SAMPLES="${NUM_SAMPLES:-101}"
export START_POINT="${START_POINT:-0}"
export TOP_K="${TOP_K:-10}"
export NUM_CANDIDATE="${NUM_CANDIDATE:-25}"
export FORWARD_WORKERS="${FORWARD_WORKERS:-1}"
export TEST_INTERVAL="${TEST_INTERVAL:-101}"
export PROFILE_MAX_TOKENS="${PROFILE_MAX_TOKENS:-220}"
export AGENT1_MAX_TOKENS="${AGENT1_MAX_TOKENS:-384}"
export AGENT2_MAX_TOKENS="${AGENT2_MAX_TOKENS:-384}"
export AGENT3_MAX_TOKENS="${AGENT3_MAX_TOKENS:-384}"
export TEMPERATURE="${TEMPERATURE:-0.0}"
export TOP_P="${TOP_P:-1.0}"
export OP_STR="${OP_STR:-amd-forward-101-top10}"

exec "$SCRIPT_DIR/run_forward_amd.sh" "$@"
