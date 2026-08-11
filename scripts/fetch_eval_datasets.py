#!/usr/bin/env python3
"""Fetch small, pinned public evaluation datasets and verify their hashes."""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path

DATASETS = {
    "gsm8k": {
        "revision": "3101c7d5072418e28b9008a6636bde82a006892c",
        "url": (
            "https://raw.githubusercontent.com/openai/grade-school-math/"
            "3101c7d5072418e28b9008a6636bde82a006892c/"
            "grade_school_math/data/test.jsonl"
        ),
        "filename": "gsm8k-test.jsonl",
        "sha256": "3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14",
    },
    "humaneval": {
        "revision": "6d43fb980f9fee3c892a914eda09951f772ad10d",
        "url": (
            "https://raw.githubusercontent.com/openai/human-eval/"
            "6d43fb980f9fee3c892a914eda09951f772ad10d/data/HumanEval.jsonl.gz"
        ),
        "filename": "HumanEval.jsonl.gz",
        "sha256": "b796127e635a67f93fb35c04f4cb03cf06f38c8072ee7cee8833d7bee06979ef",
    },
    "mmlu_pro": {
        "revision": "b189ec765aa7ed75c8acfea42df31fdae71f97be",
        "url": (
            "https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro/resolve/"
            "b189ec765aa7ed75c8acfea42df31fdae71f97be/"
            "data/test-00000-of-00001.parquet"
        ),
        "filename": "mmlu-pro-test.parquet",
        "sha256": "0e24a191921c2f453518a537a8b2117bd137e7714d4ef1565e9ba06c1ecb9ad8",
    },
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset", action="append", choices=sorted(DATASETS))
    args = parser.parse_args()
    selected = args.dataset or sorted(DATASETS)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("format") != "dspark-crack-eval-data-v1":
            raise RuntimeError(f"unexpected existing manifest format: {manifest_path}")
    else:
        manifest = {"format": "dspark-crack-eval-data-v1", "datasets": {}}
    for name in selected:
        spec = DATASETS[name]
        with urllib.request.urlopen(spec["url"], timeout=60) as response:
            payload = response.read()
        digest = hashlib.sha256(payload).hexdigest()
        if digest != spec["sha256"]:
            raise RuntimeError(f"{name} SHA-256 mismatch: {digest} != {spec['sha256']}")
        destination = args.output_dir / spec["filename"]
        destination.write_bytes(payload)
        manifest["datasets"][name] = {
            "revision": spec["revision"],
            "url": spec["url"],
            "filename": spec["filename"],
            "sha256": digest,
            "bytes": len(payload),
        }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
