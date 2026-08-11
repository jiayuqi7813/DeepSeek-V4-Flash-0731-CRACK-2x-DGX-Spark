from __future__ import annotations

import json
from pathlib import Path

from scripts.compare_judgments import compare


def _write(path: Path, labels: list[str]) -> None:
    path.write_text(
        "".join(
            json.dumps({"id": f"sample-{index}", "label": label}) + "\n"
            for index, label in enumerate(labels)
        ),
        encoding="utf-8",
    )


def test_compare_judgments(tmp_path: Path) -> None:
    base = tmp_path / "base.jsonl"
    candidate = tmp_path / "candidate.jsonl"
    _write(base, ["REFUSE", "REFUSE", "PARTIAL", "COMPLY"])
    _write(candidate, ["COMPLY", "PARTIAL", "REFUSE", "COMPLY"])

    result = compare(base, candidate)

    assert result["paired_count"] == 4
    assert result["strict_compliance"]["base_rate"] == 0.25
    assert result["strict_compliance"]["candidate_rate"] == 0.5
    assert result["material_compliance"]["base_rate"] == 0.5
    assert result["material_compliance"]["candidate_rate"] == 0.75
    assert result["label_transition_matrix"]["REFUSE"] == {
        "REFUSE": 0,
        "PARTIAL": 1,
        "COMPLY": 1,
    }
