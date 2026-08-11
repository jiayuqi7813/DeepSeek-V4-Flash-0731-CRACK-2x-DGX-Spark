#!/usr/bin/env python3
"""Build a deterministic paired calibration/holdout manifest from a pinned dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path

DATASET_ID = "heretic-org/Semantic-Harmless"
DATASET_REVISION = "7e9f2b01272da85f2be7a3437f31ac46698e8735"
DATASET_FILE = "metadata/matched_pairs.json"
DATASET_SHA256 = "d066ddd4a8005271d21f269aacc4cc30bfb5d2f1d7004f3f37004e42f63f5225"


def _split(pair_id: str, holdout_percent: int) -> str:
    bucket = int.from_bytes(hashlib.sha256(pair_id.encode()).digest()[:4], "big") % 100
    return "holdout" if bucket < holdout_percent else "train"


def build(output: Path, raw_cache: Path, *, holdout_percent: int, limit_pairs: int | None) -> dict:
    url = f"https://huggingface.co/datasets/{DATASET_ID}/resolve/{DATASET_REVISION}/{DATASET_FILE}"
    raw_cache.parent.mkdir(parents=True, exist_ok=True)
    if not raw_cache.exists():
        with urllib.request.urlopen(url, timeout=60) as response:
            payload = response.read()
        raw_cache.write_bytes(payload)
    payload = raw_cache.read_bytes()
    actual_hash = hashlib.sha256(payload).hexdigest()
    if actual_hash != DATASET_SHA256:
        raise RuntimeError(f"dataset hash mismatch: expected {DATASET_SHA256}, got {actual_hash}")
    source = json.loads(payload)
    pairs = source["pairs"]
    if limit_pairs is not None:
        pairs = pairs[:limit_pairs]

    output.parent.mkdir(parents=True, exist_ok=True)
    counts = {"train": {"harmful": 0, "harmless": 0}, "holdout": {"harmful": 0, "harmless": 0}}
    with output.open("w", encoding="utf-8") as handle:
        for index, pair in enumerate(pairs):
            pair_id = f"semantic-{index:04d}"
            split = _split(pair_id, holdout_percent)
            for label_name, label_value, field in (
                ("harmful", 1, "harmful"),
                ("harmless", 0, "harmless"),
            ):
                sample = {
                    "id": f"{pair_id}-{label_name}",
                    "pair_id": pair_id,
                    "split": split,
                    "label": label_value,
                    "label_name": label_name,
                    "language": "en",
                    "mode": "base",
                    "text": pair[field],
                    "messages": [{"role": "user", "content": pair[field]}],
                    "semantic_similarity": pair.get("score"),
                    "source": {
                        "dataset": DATASET_ID,
                        "revision": DATASET_REVISION,
                        "row": index,
                        "license": "CC-BY-4.0",
                    },
                }
                handle.write(json.dumps(sample, ensure_ascii=False) + "\n")
                counts[split][label_name] += 1

    manifest = {
        "dataset": DATASET_ID,
        "revision": DATASET_REVISION,
        "file": DATASET_FILE,
        "sha256": DATASET_SHA256,
        "license": "CC-BY-4.0",
        "pair_count": len(pairs),
        "sample_count": len(pairs) * 2,
        "holdout_percent": holdout_percent,
        "counts": counts,
        "output": str(output.resolve()),
    }
    manifest_path = output.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/generated/semantic-pairs.jsonl"),
    )
    parser.add_argument(
        "--raw-cache",
        type=Path,
        default=Path("data/generated/upstream-matched_pairs.json"),
    )
    parser.add_argument("--holdout-percent", type=int, default=20)
    parser.add_argument("--limit-pairs", type=int)
    args = parser.parse_args()
    if not 5 <= args.holdout_percent <= 50:
        parser.error("--holdout-percent must be between 5 and 50")
    result = build(
        args.output,
        args.raw_cache,
        holdout_percent=args.holdout_percent,
        limit_pairs=args.limit_pairs,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
