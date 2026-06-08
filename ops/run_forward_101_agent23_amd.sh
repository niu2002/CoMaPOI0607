#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export BASE_API_NAME="${BASE_API_NAME:-llama3.1-8b}"
export AGENT1_API="${AGENT1_API:-$BASE_API_NAME}"
export AGENT2_API="${AGENT2_API:-agent2}"
export AGENT3_API="${AGENT3_API:-agent3}"
export OP_STR="${OP_STR:-amd-forward-101-top10-agent23}"
export NUM_SAMPLES="${NUM_SAMPLES:-101}"
export TOP_K="${TOP_K:-10}"
export NUM_CANDIDATE="${NUM_CANDIDATE:-25}"
export TEST_INTERVAL="${TEST_INTERVAL:-101}"
export FORWARD_WORKERS="${FORWARD_WORKERS:-1}"
export PROFILE_MAX_TOKENS="${PROFILE_MAX_TOKENS:-220}"
export AGENT1_MAX_TOKENS="${AGENT1_MAX_TOKENS:-384}"
export AGENT2_MAX_TOKENS="${AGENT2_MAX_TOKENS:-384}"
export AGENT3_MAX_TOKENS="${AGENT3_MAX_TOKENS:-384}"

exec "$SCRIPT_DIR/run_forward_101_amd.sh" "$@"
