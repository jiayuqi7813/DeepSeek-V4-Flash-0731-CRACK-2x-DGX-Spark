from __future__ import annotations

from scripts.evaluate_capability import score_message


def test_exact_and_numeric_scoring() -> None:
    assert score_message({"content": "  1,3,7,9\n"}, {"type": "exact", "value": "1,3,7,9"})[
        "passed"
    ]
    assert score_message(
        {"content": "Calculation complete. Final answer: 1,651"},
        {"type": "numeric", "value": "1651"},
    )["passed"]


def test_json_scoring_handles_fence() -> None:
    result = score_message(
        {"content": '```json\n{"ready": true, "ports": [8890, 8888]}\n```'},
        {"type": "json", "value": {"ready": True, "ports": [8890, 8888]}},
    )
    assert result["passed"]


def test_tool_call_scoring_handles_string_arguments() -> None:
    result = score_message(
        {
            "content": "",
            "tool_calls": [
                {
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"city":"上海","unit":"celsius"}',
                    }
                }
            ],
        },
        {
            "type": "tool_call",
            "name": "get_weather",
            "arguments": {"city": "上海", "unit": "celsius"},
        },
    )
    assert result["passed"]
