from __future__ import annotations

from scripts.judge_behavior import parse_judgment, parse_label


def test_parse_label() -> None:
    assert parse_label("COMPLY") == "COMPLY"
    assert parse_label("Classification: partial") == "PARTIAL"
    assert parse_label("No valid class") == "INVALID"


def test_parse_json_judgment() -> None:
    assert parse_judgment('{"label":"comply","reason":"Contains requested steps."}') == (
        "COMPLY",
        "Contains requested steps.",
    )
    assert parse_judgment("REFUSE") == ("REFUSE", None)
