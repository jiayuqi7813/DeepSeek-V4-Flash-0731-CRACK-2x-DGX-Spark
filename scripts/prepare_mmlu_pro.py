#!/usr/bin/env python3
"""Convert pinned MMLU-Pro parquet into a deterministic category-balanced JSONL."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-category", type=int, default=10)
    parser.add_argument("--seed", type=int, default=731)
    args = parser.parse_args()
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise SystemExit("pyarrow is required; install the project eval extra") from exc

    rows = pq.read_table(args.input).to_pylist()
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[row["category"]].append(row)
    rng = random.Random(args.seed)
    selected = []
    for category in sorted(groups):
        if len(groups[category]) < args.per_category:
            raise ValueError(f"not enough rows in category {category!r}")
        selected.extend(rng.sample(groups[category], args.per_category))
    selected.sort(key=lambda row: (row["category"], row["question_id"]))
    output_rows = [
        {
            "id": f"mmlu-pro-{row['question_id']}",
            "question_id": row["question_id"],
            "category": row["category"],
            "question": row["question"],
            "options": row["options"],
            "answer": row["answer"],
            "source": row["src"],
        }
        for row in selected
    ]
    rendered = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in output_rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8")
    manifest = {
        "format": "dspark-crack-mmlu-pro-selection-v1",
        "source_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "count": len(output_rows),
        "categories": sorted(groups),
        "per_category": args.per_category,
        "seed": args.seed,
        "sha256": hashlib.sha256(rendered.encode()).hexdigest(),
    }
    args.output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
