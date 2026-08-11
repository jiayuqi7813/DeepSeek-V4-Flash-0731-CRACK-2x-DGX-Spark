#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
HEAD_ALIAS="${HEAD_ALIAS:-dgx-spark-1}"
WORKER_ALIAS="${WORKER_ALIAS:-dgx-spark-2}"
REMOTE_USER="${REMOTE_USER:-$(id -un)}"
REMOTE_HOME="${REMOTE_HOME:-/home/$REMOTE_USER}"
REMOTE_PROJECT="${REMOTE_PROJECT:-$REMOTE_HOME/DeepSeek-V4-Flash-0731-CRACK-2x-DGX-Spark}"
REMOTE_RUNTIME="${REMOTE_RUNTIME:-$REMOTE_HOME/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark}"
PROFILE_ENV_BASENAME="${PROFILE_ENV_BASENAME:-.env.production.local}"
PROJECT_NAME="${PROJECT_NAME:-deepseek-v4-0731-crack}"
API_HOST="${API_HOST:-192.168.31.200}"
API_PORT="${API_PORT:-8888}"

if [[ "$(basename "$PROFILE_ENV_BASENAME")" != "$PROFILE_ENV_BASENAME" ]]; then
  echo "PROFILE_ENV_BASENAME must be a file name within deploy/: $PROFILE_ENV_BASENAME" >&2
  exit 2
fi
if [[ ! -f "$PROJECT_DIR/deploy/$PROFILE_ENV_BASENAME" ]]; then
  echo "Missing local profile: $PROJECT_DIR/deploy/$PROFILE_ENV_BASENAME" >&2
  exit 2
fi

cleanup_failed_start() {
  local status="${1:-1}"
  trap - ERR INT TERM
  echo "Production start failed; stopping worker first, then head..." >&2
  PROFILE_ENV_BASENAME="$PROFILE_ENV_BASENAME" PROJECT_NAME="$PROJECT_NAME" \
    "$PROJECT_DIR/scripts/stop_production_cluster.sh" || true
  exit "$status"
}

trap 'cleanup_failed_start $?' ERR
trap 'cleanup_failed_start 130' INT
trap 'cleanup_failed_start 143' TERM

REMOTE_DIR="$REMOTE_PROJECT" HEAD_ALIAS="$HEAD_ALIAS" WORKER_ALIAS="$WORKER_ALIAS" \
  "$PROJECT_DIR/scripts/sync_to_sparks.sh"
ssh "$HEAD_ALIAS" "cd '$REMOTE_RUNTIME' && \
  ENV_FILE='$REMOTE_PROJECT/deploy/$PROFILE_ENV_BASENAME' \
  PROJECT_NAME='$PROJECT_NAME' \
  ./start-deepseek-v4-flash-dspark.sh --host '$API_HOST' --port '$API_PORT'"

trap - ERR INT TERM
