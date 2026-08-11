#!/usr/bin/env bash
set -euo pipefail

HEAD_ALIAS="${HEAD_ALIAS:-dgx-spark-1}"
WORKER_ALIAS="${WORKER_ALIAS:-dgx-spark-2}"
REMOTE_USER="${REMOTE_USER:-$(id -un)}"
REMOTE_HOME="${REMOTE_HOME:-/home/$REMOTE_USER}"
REMOTE_PROJECT="${REMOTE_PROJECT:-$REMOTE_HOME/DeepSeek-V4-Flash-0731-CRACK-2x-DGX-Spark}"
PROJECT_NAME="${PROJECT_NAME:-deepseek-v4-0731-crack-capture}"
PROFILE_ENV_BASENAME="${PROFILE_ENV_BASENAME:-.env.capture.local}"

if [ "$(basename "$PROFILE_ENV_BASENAME")" != "$PROFILE_ENV_BASENAME" ]; then
  echo "PROFILE_ENV_BASENAME must be a file name within deploy/: $PROFILE_ENV_BASENAME" >&2
  exit 2
fi

echo "Stopping capture worker first..."
ssh "$WORKER_ALIAS" "cd '$REMOTE_PROJECT/deploy' && docker compose -p '$PROJECT_NAME' --env-file .env.dspark -f docker-compose.dspark.yml down"

echo "Stopping capture head..."
ssh "$HEAD_ALIAS" "cd '$REMOTE_PROJECT/deploy' && docker compose -p '$PROJECT_NAME' --env-file '$PROFILE_ENV_BASENAME' -f docker-compose.capture.yml down"

echo "Capture cluster stopped."
