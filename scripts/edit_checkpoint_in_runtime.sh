#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 2 ]; then
  echo "Usage: $0 DIRECTION_FILE OUTPUT_MODEL_DIR [editor arguments...]" >&2
  exit 2
fi

DIRECTION_FILE="$1"
OUTPUT_MODEL_DIR="$2"
shift 2

DGX_USER="${DGX_USER:-$(id -un)}"
DGX_HOME="${DGX_HOME:-/home/$DGX_USER}"
PROJECT_DIR="${PROJECT_DIR:-$DGX_HOME/DeepSeek-V4-Flash-0731-CRACK-2x-DGX-Spark}"
SOURCE_MODEL_DIR="${SOURCE_MODEL_DIR:-$DGX_HOME/models/DeepSeek-V4-Flash-0731}"
RUNTIME_IMAGE="${RUNTIME_IMAGE:-ghcr.io/anemll/dspark-vllm-gx10:0.1.1}"
SOURCE_PARENT="$(dirname "$SOURCE_MODEL_DIR")"
OUTPUT_PARENT="$(dirname "$OUTPUT_MODEL_DIR")"

mkdir -p "$(dirname "$OUTPUT_MODEL_DIR")"

if [ "$SOURCE_PARENT" = "$OUTPUT_PARENT" ]; then
  # One bind mount keeps source and candidate on the same filesystem, allowing
  # unchanged shards to remain hard-linked instead of copying ~167 GB.
  docker run --rm --gpus all \
    --user "$(id -u):$(id -g)" \
    --entrypoint /usr/bin/python3 \
    -e PYTHONPATH=/workspace/src \
    -v "$PROJECT_DIR:/workspace:ro" \
    -v "$SOURCE_PARENT:/models" \
    -v "$(dirname "$DIRECTION_FILE"):/directions:ro" \
    "$RUNTIME_IMAGE" \
    -m dspark_crack.edit_checkpoint \
    --source "/models/$(basename "$SOURCE_MODEL_DIR")" \
    --output "/models/$(basename "$OUTPUT_MODEL_DIR")" \
    --directions "/directions/$(basename "$DIRECTION_FILE")" \
    --lock /workspace/model-lock.json \
    "$@"
else
  docker run --rm --gpus all \
    --user "$(id -u):$(id -g)" \
    --entrypoint /usr/bin/python3 \
    -e PYTHONPATH=/workspace/src \
    -v "$PROJECT_DIR:/workspace:ro" \
    -v "$SOURCE_MODEL_DIR:/source:ro" \
    -v "$(dirname "$DIRECTION_FILE"):/directions:ro" \
    -v "$OUTPUT_PARENT:/output" \
    "$RUNTIME_IMAGE" \
    -m dspark_crack.edit_checkpoint \
    --source /source \
    --output "/output/$(basename "$OUTPUT_MODEL_DIR")" \
    --directions "/directions/$(basename "$DIRECTION_FILE")" \
    --lock /workspace/model-lock.json \
    "$@"
fi
