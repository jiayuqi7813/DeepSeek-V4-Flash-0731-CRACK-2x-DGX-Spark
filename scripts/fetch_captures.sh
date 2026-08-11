#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
HEAD_ALIAS="${HEAD_ALIAS:-dgx-spark-1}"
REMOTE_USER="${REMOTE_USER:-$(id -un)}"
REMOTE_HOME="${REMOTE_HOME:-/home/$REMOTE_USER}"
REMOTE_PROJECT="${REMOTE_PROJECT:-$REMOTE_HOME/DeepSeek-V4-Flash-0731-CRACK-2x-DGX-Spark}"

mkdir -p "$PROJECT_DIR/artifacts/captures"
rsync -a --stats "$HEAD_ALIAS:$REMOTE_PROJECT/artifacts/captures/" "$PROJECT_DIR/artifacts/captures/"
