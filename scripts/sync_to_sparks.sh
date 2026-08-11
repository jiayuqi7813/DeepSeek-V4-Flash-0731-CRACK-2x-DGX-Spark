#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REMOTE_USER="${REMOTE_USER:-$(id -un)}"
REMOTE_HOME="${REMOTE_HOME:-/home/$REMOTE_USER}"
REMOTE_DIR="${REMOTE_DIR:-$REMOTE_HOME/DeepSeek-V4-Flash-0731-CRACK-2x-DGX-Spark}"
HEAD_ALIAS="${HEAD_ALIAS:-dgx-spark-1}"
WORKER_ALIAS="${WORKER_ALIAS:-dgx-spark-2}"

sync_one() {
  local target="$1"
  echo "Syncing project to $target:$REMOTE_DIR"
  COPYFILE_DISABLE=1 tar -C "$PROJECT_DIR" \
    --no-xattrs \
    --exclude='.git' \
    --exclude='.venv' \
    --exclude='__pycache__' \
    --exclude='artifacts/captures*' \
    --exclude='artifacts/candidates' \
    --exclude='artifacts/evals' \
    -cf - . | ssh "$target" "mkdir -p '$REMOTE_DIR' && tar -C '$REMOTE_DIR' --no-overwrite-dir -xf -"
}

sync_one "$HEAD_ALIAS"
sync_one "$WORKER_ALIAS"
