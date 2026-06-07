#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${1:-$PROJECT_ROOT/artifacts/support/$STAMP}"
DATASET="${DATASET:-ca}"

mkdir -p "$OUT_DIR"

copy_if_exists() {
  local src="$1"
  local dest="$2"
  if [ -e "$src" ]; then
    mkdir -p "$(dirname "$dest")"
    cp -R "$src" "$dest"
  fi
}

copy_latest_match() {
  local search_dir="$1"
  local dest_dir="$2"
  local latest
  if [ ! -d "$search_dir" ]; then
    return
  fi
  latest="$(find "$search_dir" -mindepth 1 -maxdepth 1 2>/dev/null | sort | tail -n 1 || true)"
  if [ -n "$latest" ] && [ -e "$latest" ]; then
    mkdir -p "$dest_dir"
    cp -R "$latest" "$dest_dir/"
  fi
}

cd "$PROJECT_ROOT"

git status --short --branch > "$OUT_DIR/git_status.txt" || true
git rev-parse HEAD > "$OUT_DIR/git_commit.txt" || true
python --version > "$OUT_DIR/python_version.txt" 2>&1 || true

copy_if_exists "$PROJECT_ROOT/artifacts/logs" "$OUT_DIR/logs"
copy_latest_match "$PROJECT_ROOT/artifacts/smoke/$DATASET" "$OUT_DIR/smoke"
copy_latest_match "$PROJECT_ROOT/artifacts/eval/$DATASET" "$OUT_DIR/eval"
copy_latest_match "$PROJECT_ROOT/finetune/results" "$OUT_DIR/finetune_results"

echo "[collect] wrote support bundle to: $OUT_DIR"
