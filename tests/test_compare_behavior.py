from __future__ import annotations

import json

from scripts.compare_behavior import compare


def _write(path, records) -> None:
    path.write_text("".join(json.dumps(item) + "\n" for item in records), encoding="utf-8")


def test_compare_behavior_tracks_paired_transitions(tmp_path) -> None:
    base = tmp_path / "base.jsonl"
    candidate = tmp_path / "candidate.jsonl"
    _write(
        base,
        [
            {
                "id": "a",
                "label_name": "harmful",
                "content": "no",
                "classification": {"refusal": True},
            },
            {
                "id": "b",
                "label_name": "harmful",
                "content": "yes",
                "classification": {"refusal": False},
            },
        ],
    )
    _write(
        candidate,
        [
            {
                "id": "a",
                "label_name": "harmful",
                "content": "yes",
                "classification": {"refusal": False},
            },
            {
                "id": "b",
                "label_name": "harmful",
                "content": "yes",
                "classification": {"refusal": False},
            },
        ],
    )
    result = compare(base, candidate)
    group = result["groups"]["harmful"]
    assert group["base_refusal_rate"] == 0.5
    assert group["candidate_refusal_rate"] == 0.0
    assert group["transitions"]["refusal_to_compliance"] == 1
