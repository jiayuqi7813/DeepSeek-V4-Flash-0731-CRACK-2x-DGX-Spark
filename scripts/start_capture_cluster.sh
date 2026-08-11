#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
HEAD_ALIAS="${HEAD_ALIAS:-dgx-spark-1}"
WORKER_ALIAS="${WORKER_ALIAS:-dgx-spark-2}"
REMOTE_USER="${REMOTE_USER:-$(id -un)}"
REMOTE_HOME="${REMOTE_HOME:-/home/$REMOTE_USER}"
REMOTE_PROJECT="${REMOTE_PROJECT:-$REMOTE_HOME/DeepSeek-V4-Flash-0731-CRACK-2x-DGX-Spark}"
REMOTE_BASE="${REMOTE_BASE:-$REMOTE_HOME/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark}"
PROJECT_NAME="${PROJECT_NAME:-deepseek-v4-0731-crack-capture}"
PROFILE_ENV_BASENAME="${PROFILE_ENV_BASENAME:-.env.capture.local}"

if [ "$(basename "$PROFILE_ENV_BASENAME")" != "$PROFILE_ENV_BASENAME" ]; then
  echo "PROFILE_ENV_BASENAME must be a file name within deploy/: $PROFILE_ENV_BASENAME" >&2
  exit 2
fi
if [ ! -f "$PROJECT_DIR/deploy/$PROFILE_ENV_BASENAME" ]; then
  echo "Missing local profile: $PROJECT_DIR/deploy/$PROFILE_ENV_BASENAME" >&2
  exit 2
fi

REMOTE_DIR="$REMOTE_PROJECT" HEAD_ALIAS="$HEAD_ALIAS" WORKER_ALIAS="$WORKER_ALIAS" \
  "$PROJECT_DIR/scripts/sync_to_sparks.sh"

ssh "$HEAD_ALIAS" "mkdir -p '$REMOTE_PROJECT/artifacts/captures/samples' && cd '$REMOTE_BASE' && \
  ENV_FILE='$REMOTE_PROJECT/deploy/$PROFILE_ENV_BASENAME' \
  COMPOSE_FILE='$REMOTE_PROJECT/deploy/docker-compose.capture.yml' \
  PROJECT_NAME='$PROJECT_NAME' \
  WAIT_ATTEMPTS='120' WAIT_SECONDS='15' \
  ./start-deepseek-v4-flash-dspark.sh --host 127.0.0.1 --port 8890"
