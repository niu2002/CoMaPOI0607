#!/usr/bin/env bash
set -euo pipefail

BRANCH="${1:-amd}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd "$PROJECT_ROOT"

echo "[pull] repo: $PROJECT_ROOT"
echo "[pull] branch: $BRANCH"

git fetch origin --prune

if ! git show-ref --verify --quiet "refs/remotes/origin/$BRANCH"; then
  echo "[pull] remote branch origin/$BRANCH not found"
  git branch -a
  exit 1
fi

if [ -n "$(git status --porcelain)" ]; then
  echo "[pull] working tree is dirty; refusing to pull"
  git status --short
  exit 2
fi

if git show-ref --verify --quiet "refs/heads/$BRANCH"; then
  git checkout "$BRANCH"
else
  git checkout -b "$BRANCH" "origin/$BRANCH"
fi

git pull --ff-only origin "$BRANCH"
git status --short --branch
git rev-parse HEAD
