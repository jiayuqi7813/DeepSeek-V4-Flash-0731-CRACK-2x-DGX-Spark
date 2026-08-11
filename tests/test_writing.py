from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

from scripts.judge_writing import (
    balanced_base_a_ids,
    parse_judgment,
    retain_valid_judgments,
)


def _valid_payload() -> dict:
    return {
        "winner": "A",
        "scores": {
            "A": {"task": 5, "coherence": 4, "style": 4, "language": 5},
            "B": {"task": 4, "coherence": 4, "style": 3, "language": 5},
        },
        "reason": "A follows one more constraint.",
    }


def test_parse_writing_judgment_accepts_json_and_fences() -> None:
    payload = _valid_payload()
    parsed = parse_judgment(json.dumps(payload))
    assert parsed["valid"]
    assert parsed["winner"] == "A"
    assert parsed["scores"]["A"]["task"] == 5.0
    fenced = parse_judgment(f"```json\n{json.dumps(payload)}\n```")
    assert fenced == parsed


def test_parse_writing_judgment_rejects_bad_scores() -> None:
    payload = _valid_payload()
    payload["scores"]["B"]["style"] = 7
    assert not parse_judgment(json.dumps(payload))["valid"]
    assert not parse_judgment("not json")["valid"]


def test_writing_dataset_and_blind_order_are_locked() -> None:
    path = Path(__file__).resolve().parents[1] / "data/eval/writing-v1.jsonl"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == (
        "1b6749963ab9f513cf28c9a80ce51efc007de455baf3c938858ee0dade634d31"
    )
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    assert len(rows) == 12
    assert Counter(row["language"] for row in rows) == {"en": 6, "zh": 6}
    assert all(count == 2 for count in Counter(row["category"] for row in rows).values())
    sample_ids = [row["id"] for row in rows]
    base_a = balanced_base_a_ids(sample_ids, 731)
    assert len(base_a) == 6
    assert base_a == balanced_base_a_ids(sample_ids, 731)


def test_retry_invalid_retains_only_valid_rows() -> None:
    records = {
        "valid": {"judgment": {"valid": True}},
        "invalid": {"judgment": {"valid": False}},
        "missing": {},
    }
    assert retain_valid_judgments(records) == {"valid": records["valid"]}
