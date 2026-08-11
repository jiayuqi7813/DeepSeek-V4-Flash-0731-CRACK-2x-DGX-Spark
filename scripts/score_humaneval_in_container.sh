#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
IMAGE="${DSPARK_VLLM_IMAGE:-ghcr.io/anemll/dspark-vllm-gx10:0.1.1}"

if [[ $# -ne 3 ]]; then
  echo "Usage: $0 DATASET COMPLETIONS OUTPUT" >&2
  exit 2
fi

resolve_in_project() {
  python3 - "$PROJECT_DIR" "$1" <<'PY'
from pathlib import Path
import sys

root = Path(sys.argv[1]).resolve()
path = Path(sys.argv[2])
path = path.resolve() if path.is_absolute() else (root / path).resolve()
try:
    path.relative_to(root)
except ValueError as exc:
    raise SystemExit(f"path must be inside {root}: {path}") from exc
print(path)
PY
}

DATASET="$(resolve_in_project "$1")"
COMPLETIONS="$(resolve_in_project "$2")"
OUTPUT="$(resolve_in_project "$3")"
mkdir -p "$(dirname "$OUTPUT")"

dataset_rel="${DATASET#"$PROJECT_DIR"/}"
completions_rel="${COMPLETIONS#"$PROJECT_DIR"/}"
output_rel="${OUTPUT#"$PROJECT_DIR"/}"

docker run --rm \
  --user "$(id -u):$(id -g)" \
  --network none \
  --cap-drop ALL \
  --security-opt no-new-privileges \
  --pids-limit 64 \
  --memory 1g \
  --cpus 2 \
  --read-only \
  --tmpfs /tmp:rw,nosuid,nodev,size=512m \
  -v "$PROJECT_DIR:/project:ro" \
  -v "$(dirname "$OUTPUT"):/output:rw" \
  --entrypoint /usr/bin/python3 \
  "$IMAGE" \
  /project/scripts/score_humaneval.py \
    --dataset "/project/$dataset_rel" \
    --completions "/project/$completions_rel" \
    --output "/output/$(basename "$output_rel")"
