#!/usr/bin/env bash
set -euo pipefail

DGX_USER="${DGX_USER:-$(id -un)}"
DGX_HOME="${DGX_HOME:-/home/$DGX_USER}"
PROJECT_DIR="${PROJECT_DIR:-$DGX_HOME/DeepSeek-V4-Flash-0731-CRACK-2x-DGX-Spark}"
SOURCE_MODEL_DIR="${SOURCE_MODEL_DIR:-$DGX_HOME/models/DeepSeek-V4-Flash-0731}"
RUNTIME_IMAGE="${RUNTIME_IMAGE:-ghcr.io/anemll/dspark-vllm-gx10:0.1.1}"

docker run --rm \
  --entrypoint /usr/bin/python3 \
  -e PYTHONPATH=/workspace/src \
  -v "$PROJECT_DIR:/workspace:ro" \
  -v "$SOURCE_MODEL_DIR:/model:ro" \
  "$RUNTIME_IMAGE" \
  -m dspark_crack.identity \
  /model \
  --lock /workspace/model-lock.json \
  "$@"
