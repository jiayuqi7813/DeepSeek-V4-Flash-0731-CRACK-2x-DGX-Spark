from __future__ import annotations

import json
from pathlib import Path

from scripts.compare_capability import compare


def _write(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_compare_mixed_record_formats(tmp_path: Path) -> None:
    base = tmp_path / "base.jsonl"
    candidate = tmp_path / "candidate.jsonl"
    _write(
        base,
        [
            {"id": "a", "passed": True},
            {"id": "b", "passed": True},
            {"id": "c", "score": {"passed": False}},
        ],
    )
    _write(
        candidate,
        [
            {"id": "a", "score": {"passed": True}},
            {"id": "b", "score": {"passed": False}},
            {"id": "c", "passed": True},
        ],
    )
    result = compare(base, candidate)
    assert result["paired_count"] == 3
    assert result["accuracy_delta"] == 0
    assert result["transitions"] == {
        "pass_to_pass": 1,
        "pass_to_fail": 1,
        "fail_to_pass": 1,
        "fail_to_fail": 0,
    }
