#!/usr/bin/env python3
"""Build a deterministic, synthetic capability-regression suite."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any


def _row(
    sample_id: str,
    category: str,
    prompt: str,
    expected: dict[str, Any],
    *,
    tools: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "id": sample_id,
        "category": category,
        "messages": [{"role": "user", "content": prompt}],
        "expected": expected,
    }
    if tools is not None:
        result["tools"] = tools
        result["tool_choice"] = "auto"
    return result


def _tool(name: str, description: str, properties: dict[str, Any], required: list[str]) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": False,
            },
        },
    }


def build_suite() -> list[dict[str, Any]]:
    rows = [
        _row(
            "exact-zh-sort",
            "instruction",
            "只输出这些整数从小到大的结果，用英文逗号连接，不要解释：9, 1, 7, 3",
            {"type": "exact", "value": "1,3,7,9"},
        ),
        _row(
            "exact-zh-reverse",
            "instruction",
            "只输出字符串 DGX-Spark-0731 的逐字符逆序结果，不要解释。",
            {"type": "exact", "value": "1370-krapS-XGD"},
        ),
        _row(
            "exact-en-filter",
            "instruction",
            "Return only the even integers, preserving order, comma-separated: 7 4 9 12 3 8",
            {"type": "exact", "value": "4,12,8"},
        ),
        _row(
            "exact-en-acrostic",
            "instruction",
            "Return only the first letter of each word, with no separators: Delta Gamma Xray Spark",
            {"type": "exact", "value": "DGXS"},
        ),
        _row(
            "math-01",
            "math",
            "Compute 37 * 48 - 125. Return only the number.",
            {"type": "numeric", "value": "1651"},
        ),
        _row(
            "math-02",
            "math",
            "A box has 18 rows of 24 bolts. 57 bolts are removed. "
            "How many remain? Return only the number.",
            {"type": "numeric", "value": "375"},
        ),
        _row(
            "math-03",
            "math",
            "What is 15% of 860? Return only the number.",
            {"type": "numeric", "value": "129"},
        ),
        _row(
            "math-04",
            "math",
            "The mean of 14, 18, 21, 27, and x is 22. What is x? Return only the number.",
            {"type": "numeric", "value": "30"},
        ),
        _row(
            "math-05",
            "math",
            "A train travels 315 km in 4.5 hours. What is its average speed in km/h? "
            "Return only the number.",
            {"type": "numeric", "value": "70"},
        ),
        _row(
            "math-06",
            "math",
            "Solve 7x - 19 = 100. Return only x.",
            {"type": "numeric", "value": "17"},
        ),
        _row(
            "json-01",
            "json",
            '只输出合法 JSON，不要代码块：键为 "city" 和 "count"，值分别为 "北京" 和 3。',
            {"type": "json", "value": {"city": "北京", "count": 3}},
        ),
        _row(
            "json-02",
            "json",
            'Output only JSON: an object with "ready" set to true and "ports" set to [8890, 8888].',
            {"type": "json", "value": {"ready": True, "ports": [8890, 8888]}},
        ),
        _row(
            "json-03",
            "json",
            '只输出 JSON 数组，依次包含字符串 "甲"、null、数字 7、布尔值 false。',
            {"type": "json", "value": ["甲", None, 7, False]},
        ),
    ]

    weather = _tool(
        "get_weather",
        "Get weather for one city.",
        {
            "city": {"type": "string"},
            "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
        },
        ["city", "unit"],
    )
    stock = _tool(
        "lookup_stock",
        "Look up a stock quote.",
        {"ticker": {"type": "string"}, "exchange": {"type": "string"}},
        ["ticker", "exchange"],
    )
    rows.extend(
        [
            _row(
                "tool-weather-zh",
                "tool_call",
                "请查询上海的摄氏温度。",
                {
                    "type": "tool_call",
                    "name": "get_weather",
                    "arguments": {"city": "上海", "unit": "celsius"},
                },
                tools=[weather],
            ),
            _row(
                "tool-weather-en",
                "tool_call",
                "Check the weather in Boston using Fahrenheit.",
                {
                    "type": "tool_call",
                    "name": "get_weather",
                    "arguments": {"city": "Boston", "unit": "fahrenheit"},
                },
                tools=[weather],
            ),
            _row(
                "tool-stock",
                "tool_call",
                "Look up NVDA on NASDAQ.",
                {
                    "type": "tool_call",
                    "name": "lookup_stock",
                    "arguments": {"ticker": "NVDA", "exchange": "NASDAQ"},
                },
                tools=[weather, stock],
            ),
        ]
    )

    rng = random.Random(731)
    positions = (0, 99, 199)
    for case, position in enumerate(positions, 1):
        needle = f"SPARK-NEEDLE-{case}-7F3A"
        records = []
        for index in range(200):
            payload = "".join(rng.choice("ABCDEFGHJKLMNPQRSTUVWXYZ23456789") for _ in range(32))
            value = needle if index == position else payload
            records.append(f"record-{index:03d}: {value}")
        prompt = (
            "Below is an immutable diagnostic log. Return only the value on the requested "
            f"line record-{position:03d}, with no explanation.\n\n" + "\n".join(records)
        )
        rows.append(
            _row(
                f"needle-{case}",
                "long_context",
                prompt,
                {"type": "exact", "value": needle},
            )
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = build_suite()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rendered = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    args.output.write_text(rendered, encoding="utf-8")
    manifest = {
        "format": "dspark-crack-capability-suite-v1",
        "count": len(rows),
        "categories": sorted({row["category"] for row in rows}),
        "sha256": hashlib.sha256(rendered.encode()).hexdigest(),
        "seed": 731,
    }
    args.output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
