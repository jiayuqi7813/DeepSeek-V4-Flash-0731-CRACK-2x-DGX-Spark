from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path


def test_bilingual_behavior_probe_is_paired_and_locked() -> None:
    path = Path(__file__).resolve().parents[1] / "data/eval/bilingual-behavior-v1.jsonl"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == (
        "ba1587a255168d48e4048caac84ab733ec35510ab7fcbdce7df2f823ba89a966"
    )
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    assert len(rows) == 20
    assert len({row["id"] for row in rows}) == 20
    assert {row["language"] for row in rows} == {"en", "zh"}
    assert sum(row["language"] == "en" for row in rows) == 10
    assert sum(row["language"] == "zh" for row in rows) == 10

    pairs: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        assert row["split"] == "holdout"
        assert row["label_name"] in {"harmful", "harmless"}
        assert row["label"] == int(row["label_name"] == "harmful")
        assert row["messages"] and row["messages"][-1]["role"] == "user"
        pairs[row["pair_id"]].append(row)
    assert len(pairs) == 10
    assert all(
        {row["label_name"] for row in pair} == {"harmful", "harmless"}
        for pair in pairs.values()
    )
