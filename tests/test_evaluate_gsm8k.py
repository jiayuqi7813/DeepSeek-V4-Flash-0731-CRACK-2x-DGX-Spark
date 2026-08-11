from __future__ import annotations

from decimal import Decimal

from scripts.evaluate_gsm8k import extract_number


def test_extract_number_prefers_hash_answer() -> None:
    assert extract_number("There were 17, then 42.\n#### 1,651") == Decimal("1651")


def test_extract_number_falls_back_to_last_number() -> None:
    assert extract_number("The final answer is 70 km/h.") == Decimal("70")


def test_extract_number_falls_back_before_empty_hash_answer() -> None:
    assert extract_number("The calculation gives 123.\n#### no numeric answer") == Decimal("123")


def test_extract_number_handles_missing_answer() -> None:
    assert extract_number("No numeric answer") is None
