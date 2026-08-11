#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 1 ]; then
  echo "Usage: $0 CANDIDATE_MODEL_DIR" >&2
  exit 2
fi

CANDIDATE_MODEL_DIR="$1"
DGX_USER="${DGX_USER:-$(id -un)}"
DGX_HOME="${DGX_HOME:-/home/$DGX_USER}"
PROJECT_DIR="${PROJECT_DIR:-$DGX_HOME/DeepSeek-V4-Flash-0731-CRACK-2x-DGX-Spark}"
SOURCE_MODEL_DIR="${SOURCE_MODEL_DIR:-$DGX_HOME/models/DeepSeek-V4-Flash-0731}"
RUNTIME_IMAGE="${RUNTIME_IMAGE:-ghcr.io/anemll/dspark-vllm-gx10:0.1.1}"
SOURCE_PARENT="$(dirname "$SOURCE_MODEL_DIR")"
CANDIDATE_PARENT="$(dirname "$CANDIDATE_MODEL_DIR")"

if [ "$SOURCE_PARENT" != "$CANDIDATE_PARENT" ]; then
  echo "Source and candidate must share a parent directory for this validator." >&2
  exit 2
fi

docker run --rm \
  --entrypoint /usr/bin/python3 \
  -e PYTHONPATH=/workspace/src \
  -v "$PROJECT_DIR:/workspace:ro" \
  -v "$SOURCE_PARENT:/models" \
  "$RUNTIME_IMAGE" \
  -m dspark_crack.validate_candidate \
  --source "/models/$(basename "$SOURCE_MODEL_DIR")" \
  --candidate "/models/$(basename "$CANDIDATE_MODEL_DIR")" \
  --lock /workspace/model-lock.json \
  --output "/models/$(basename "$CANDIDATE_MODEL_DIR")/CRACK_VALIDATION.json"
