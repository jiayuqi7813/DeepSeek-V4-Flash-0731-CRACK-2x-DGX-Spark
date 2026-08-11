#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
HEAD_ALIAS="${HEAD_ALIAS:-dgx-spark-1}"
WORKER_ALIAS="${WORKER_ALIAS:-dgx-spark-2}"
REMOTE_USER="${REMOTE_USER:-$(id -un)}"
REMOTE_HOME="${REMOTE_HOME:-/home/$REMOTE_USER}"
REMOTE_PROJECT="${REMOTE_PROJECT:-$REMOTE_HOME/DeepSeek-V4-Flash-0731-CRACK-2x-DGX-Spark}"
OUTPUT_MODEL_NAME="${OUTPUT_MODEL_NAME:-DeepSeek-V4-Flash-0731-CRACK}"
DIRECTION_REL="${DIRECTION_REL:-artifacts/directions/attn-out-sra-r8.safetensors}"
VERIFY_SOURCE_FULL_HASH="${VERIFY_SOURCE_FULL_HASH:-0}"

if [[ "$OUTPUT_MODEL_NAME" == */* || -z "$OUTPUT_MODEL_NAME" || \
      "$OUTPUT_MODEL_NAME" == "." || "$OUTPUT_MODEL_NAME" == ".." ]]; then
  echo "OUTPUT_MODEL_NAME must be a single non-empty directory name." >&2
  exit 2
fi
if [[ "$HEAD_ALIAS" == "$WORKER_ALIAS" ]]; then
  echo "HEAD_ALIAS and WORKER_ALIAS must identify different nodes." >&2
  exit 2
fi
if [[ "$DIRECTION_REL" = /* || "$DIRECTION_REL" == *".."* ]]; then
  echo "DIRECTION_REL must be a repository-relative path without '..'." >&2
  exit 2
fi
if [[ ! -f "$PROJECT_DIR/$DIRECTION_REL" ]]; then
  echo "Direction artifact does not exist: $PROJECT_DIR/$DIRECTION_REL" >&2
  exit 2
fi
if [[ "$VERIFY_SOURCE_FULL_HASH" != 0 && "$VERIFY_SOURCE_FULL_HASH" != 1 ]]; then
  echo "VERIFY_SOURCE_FULL_HASH must be 0 or 1." >&2
  exit 2
fi

REMOTE_OUTPUT="$REMOTE_HOME/models/$OUTPUT_MODEL_NAME"
REMOTE_DIRECTION="$REMOTE_PROJECT/$DIRECTION_REL"
EXPECTED_DIRECTION_SHA="$(shasum -a 256 "$PROJECT_DIR/$DIRECTION_REL" | awk '{print $1}')"

REMOTE_DIR="$REMOTE_PROJECT" HEAD_ALIAS="$HEAD_ALIAS" WORKER_ALIAS="$WORKER_ALIAS" \
  "$SCRIPT_DIR/sync_to_sparks.sh"

for host in "$HEAD_ALIAS" "$WORKER_ALIAS"; do
  if ssh "$host" "test -e '$REMOTE_OUTPUT'"; then
    echo "Refusing to overwrite existing candidate on $host: $REMOTE_OUTPUT" >&2
    exit 2
  fi
done

run_pair() {
  local description="$1"
  local remote_command="$2"
  local head_status worker_status

  echo "$description on $HEAD_ALIAS and $WORKER_ALIAS..."
  ssh "$HEAD_ALIAS" "$remote_command" &
  local head_pid=$!
  ssh "$WORKER_ALIAS" "$remote_command" &
  local worker_pid=$!

  set +e
  wait "$head_pid"
  head_status=$?
  wait "$worker_pid"
  worker_status=$?
  set -e
  if (( head_status != 0 || worker_status != 0 )); then
    echo "$description failed: head=$head_status worker=$worker_status" >&2
    return 1
  fi
}

if [[ "$VERIFY_SOURCE_FULL_HASH" == 1 ]]; then
  run_pair "Verifying every source LFS shard" \
    "cd '$REMOTE_PROJECT' && ./scripts/verify_source_in_runtime.sh --full-hash"
else
  run_pair "Verifying source identity and target headers" \
    "cd '$REMOTE_PROJECT' && ./scripts/verify_source_in_runtime.sh"
fi

run_pair "Building candidate independently" \
  "cd '$REMOTE_PROJECT' && ./scripts/edit_checkpoint_in_runtime.sh '$REMOTE_DIRECTION' '$REMOTE_OUTPUT' --layers 10-42 --direction-mode layer.sra --strength 2.0 --no-preserve-row-norm --mtp stock --scale-policy fixed"

run_pair "Installing model card and release notices" \
  "test ! -e '$REMOTE_OUTPUT/UPSTREAM_README.md' && cp '$REMOTE_OUTPUT/README.md' '$REMOTE_OUTPUT/UPSTREAM_README.md' && cp '$REMOTE_OUTPUT/.hf-manifest.json' '$REMOTE_OUTPUT/SOURCE_HF_MANIFEST.json' && cp '$REMOTE_PROJECT/MODEL_CARD.md' '$REMOTE_OUTPUT/README.md' && cp '$REMOTE_PROJECT/MODEL_CARD.md' '$REMOTE_OUTPUT/CRACK_MODEL_CARD.md' && cp '$REMOTE_PROJECT/THIRD_PARTY_NOTICES.md' '$REMOTE_OUTPUT/THIRD_PARTY_NOTICES.md'"

run_pair "Validating every candidate tensor" \
  "cd '$REMOTE_PROJECT' && ./scripts/validate_candidate_in_runtime.sh '$REMOTE_OUTPUT'"

PAIR_TMP_DIR="$(mktemp -d -t dspark-crack-pair.XXXXXX)"
trap 'rm -rf "$PAIR_TMP_DIR"' EXIT
scp -q "$HEAD_ALIAS:$REMOTE_OUTPUT/CRACK_EDIT_MANIFEST.json" "$PAIR_TMP_DIR/head-manifest.json"
scp -q "$WORKER_ALIAS:$REMOTE_OUTPUT/CRACK_EDIT_MANIFEST.json" "$PAIR_TMP_DIR/worker-manifest.json"
scp -q "$HEAD_ALIAS:$REMOTE_OUTPUT/CRACK_EDIT_REPORT.json" "$PAIR_TMP_DIR/head-report.json"
scp -q "$WORKER_ALIAS:$REMOTE_OUTPUT/CRACK_EDIT_REPORT.json" "$PAIR_TMP_DIR/worker-report.json"
scp -q "$HEAD_ALIAS:$REMOTE_OUTPUT/CRACK_VALIDATION.json" "$PAIR_TMP_DIR/head-validation.json"
scp -q "$WORKER_ALIAS:$REMOTE_OUTPUT/CRACK_VALIDATION.json" "$PAIR_TMP_DIR/worker-validation.json"
scp -q "$HEAD_ALIAS:$REMOTE_OUTPUT/README.md" "$PAIR_TMP_DIR/head-model-card.md"
scp -q "$WORKER_ALIAS:$REMOTE_OUTPUT/README.md" "$PAIR_TMP_DIR/worker-model-card.md"
scp -q "$HEAD_ALIAS:$REMOTE_OUTPUT/THIRD_PARTY_NOTICES.md" "$PAIR_TMP_DIR/head-notices.md"
scp -q "$WORKER_ALIAS:$REMOTE_OUTPUT/THIRD_PARTY_NOTICES.md" "$PAIR_TMP_DIR/worker-notices.md"

python3 - "$PAIR_TMP_DIR" "$EXPECTED_DIRECTION_SHA" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
expected_direction_sha = sys.argv[2]


def load(name: str) -> dict:
    return json.loads((root / name).read_text(encoding="utf-8"))


head_manifest = load("head-manifest.json")
worker_manifest = load("worker-manifest.json")
head_report = load("head-report.json")
worker_report = load("worker-report.json")
head_validation = load("head-validation.json")
worker_validation = load("worker-validation.json")

failures = []
if head_manifest != worker_manifest:
    failures.append("compact edit manifests differ across nodes")
if head_manifest.get("direction_sha256") != expected_direction_sha:
    failures.append("candidate direction SHA-256 does not match the local release artifact")
for node, report, validation in (
    ("head", head_report, head_validation),
    ("worker", worker_report, worker_validation),
):
    if report.get("status") != "complete":
        failures.append(f"{node} edit report is not complete")
    if not validation.get("ok") or validation.get("failures"):
        failures.append(f"{node} candidate validation did not pass cleanly")


def shard_hashes(report: dict) -> dict[str, str]:
    return {item["file"]: item["sha256"] for item in report.get("edited_shards", [])}


head_hashes = shard_hashes(head_report)
worker_hashes = shard_hashes(worker_report)
if not head_hashes:
    failures.append("edit reports contain no edited shard hashes")
if head_hashes != worker_hashes:
    failures.append("edited shard SHA-256 values differ across nodes")
for filename in ("model-card.md", "notices.md"):
    if (root / f"head-{filename}").read_bytes() != (root / f"worker-{filename}").read_bytes():
        failures.append(f"{filename} differs across nodes")

if failures:
    raise SystemExit("Pair validation failed:\n- " + "\n- ".join(failures))

print(
    json.dumps(
        {
            "ok": True,
            "format": head_manifest.get("format"),
            "direction_sha256": expected_direction_sha,
            "edited_target_count": head_validation.get("edited_target_count"),
            "edited_shard_count": len(head_hashes),
            "cross_node_edited_shards_identical": True,
        },
        indent=2,
    )
)
PY

echo "Release candidate is independently built and validated on both nodes: $REMOTE_OUTPUT"
