from __future__ import annotations

from scripts.evaluate_mmlu_pro import extract_answer


def test_extract_answer_prefers_explicit_answer() -> None:
    assert extract_answer("A is tempting, but the answer is (I).") == "I"


def test_extract_answer_falls_back_to_last_letter() -> None:
    assert extract_answer("I considered B and C. Final: D") == "D"


def test_extract_answer_handles_missing_letter() -> None:
    assert extract_answer("No selection was made") is None
