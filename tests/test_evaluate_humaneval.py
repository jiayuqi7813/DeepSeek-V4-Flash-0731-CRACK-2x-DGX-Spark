from __future__ import annotations

from scripts.evaluate_humaneval import extract_code


def test_extract_code_from_python_fence() -> None:
    text = "Here you go:\n```python\ndef add(a, b):\n    return a + b\n```"
    assert extract_code(text) == "def add(a, b):\n    return a + b"


def test_extract_code_without_fence() -> None:
    assert extract_code("def f():\n    return 1\n") == "def f():\n    return 1"
