#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ENV_FILE="${ENV_FILE:-$SCRIPT_DIR/.env.dspark}"
COMPOSE_FILE="${COMPOSE_FILE:-$SCRIPT_DIR/docker-compose.dspark.yml}"
PROJECT_NAME="${PROJECT_NAME:-deepseek-v4-flash}"
LEGACY_PROJECT_NAME="${LEGACY_PROJECT_NAME:-$(basename "$SCRIPT_DIR" | tr '[:upper:]' '[:lower:]')}"

if [ -f "$ENV_FILE" ]; then
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
fi

: "${WORKER_HOST:?WORKER_HOST must be set in $ENV_FILE or environment}"

cd "$SCRIPT_DIR"

WORKER_DIR="${WORKER_SCRIPT_DIR:-${WORKER_DIR:-$SCRIPT_DIR}}"
WORKER_HF_CACHE="${WORKER_HF_CACHE:-${HF_CACHE:-}}"
WORKER_VLLM_HOST_IP="${WORKER_VLLM_HOST_IP:-}"

local_project_has_resources() {
  local project="$1"
  {
    docker ps -aq --filter "label=com.docker.compose.project=$project"
    docker network ls -q --filter "label=com.docker.compose.project=$project"
    docker volume ls -q --filter "label=com.docker.compose.project=$project"
  } | grep -q .
}

stop_project() {
  local project="$1"

  if local_project_has_resources "$project"; then
    echo "Stopping DSpark head project ${project}..."
    COMPOSE_DISABLE_ENV_FILE=1 docker compose -p "$project" --env-file "$ENV_FILE" -f "$COMPOSE_FILE" down || true
  else
    echo "No DSpark head resources for project ${project}; skipping."
  fi

  ssh "$WORKER_HOST" "
    cd '$WORKER_DIR' || exit 1
    if {
      docker ps -aq --filter 'label=com.docker.compose.project=$project'
      docker network ls -q --filter 'label=com.docker.compose.project=$project'
      docker volume ls -q --filter 'label=com.docker.compose.project=$project'
    } | grep -q .; then
      echo 'Stopping DSpark worker project $project on $WORKER_HOST...'
      env -u MASTER_ADDR -u MASTER_PORT -u NODE_RANK -u HEADLESS \
        COMPOSE_DISABLE_ENV_FILE=1 HF_CACHE='$WORKER_HF_CACHE' \
        VLLM_HOST_IP='$WORKER_VLLM_HOST_IP' \
        docker compose -p '$project' --env-file .env.dspark \
          -f docker-compose.dspark.yml down
    else
      echo 'No DSpark worker resources for project $project on $WORKER_HOST; skipping.'
    fi
  " || true
}

stop_project "$PROJECT_NAME"
if [ "$LEGACY_PROJECT_NAME" != "$PROJECT_NAME" ]; then
  stop_project "$LEGACY_PROJECT_NAME"
fi

echo "DeepSeek V4 Flash DSpark stopped."
