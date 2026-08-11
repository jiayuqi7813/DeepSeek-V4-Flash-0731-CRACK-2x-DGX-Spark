#!/usr/bin/env python3
"""Evaluate deterministic capability records through an OpenAI-compatible API."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.request
from collections import defaultdict
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

NUMBER_RE = re.compile(r"[-+]?(?:\d[\d,]*)(?:\.\d+)?")


def _strip_fence(text: str) -> str:
    value = text.strip()
    if value.startswith("```") and value.endswith("```"):
        lines = value.splitlines()
        if len(lines) >= 3:
            value = "\n".join(lines[1:-1]).strip()
    return value


def _parse_json(text: str) -> Any:
    value = _strip_fence(text)
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        starts = [index for index in (value.find("{"), value.find("[")) if index >= 0]
        if not starts:
            raise
        start = min(starts)
        end = max(value.rfind("}"), value.rfind("]"))
        if end < start:
            raise
        return json.loads(value[start : end + 1])


def _last_number(text: str) -> Decimal | None:
    matches = NUMBER_RE.findall(text)
    if not matches:
        return None
    try:
        return Decimal(matches[-1].replace(",", ""))
    except InvalidOperation:
        return None


def score_message(message: dict[str, Any], expected: dict[str, Any]) -> dict[str, Any]:
    kind = expected["type"]
    content = message.get("content") or ""
    detail: dict[str, Any] = {"type": kind}
    passed = False
    if kind == "exact":
        actual = _strip_fence(content)
        detail["actual"] = actual
        passed = actual == expected["value"]
    elif kind == "numeric":
        actual_number = _last_number(content)
        wanted = Decimal(str(expected["value"]))
        detail["actual"] = None if actual_number is None else str(actual_number)
        passed = actual_number == wanted
    elif kind == "json":
        try:
            actual_json = _parse_json(content)
            detail["actual"] = actual_json
            passed = actual_json == expected["value"]
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            detail["error"] = str(exc)
    elif kind == "tool_call":
        calls = message.get("tool_calls") or []
        parsed_calls = []
        for call in calls:
            function = call.get("function", {})
            arguments = function.get("arguments", {})
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    pass
            parsed_calls.append({"name": function.get("name"), "arguments": arguments})
        detail["actual"] = parsed_calls
        passed = any(
            call["name"] == expected["name"] and call["arguments"] == expected["arguments"]
            for call in parsed_calls
        )
    else:
        raise ValueError(f"unsupported evaluator type: {kind}")
    detail["passed"] = passed
    return detail


def _post_chat(
    endpoint: str,
    api_key: str,
    model: str,
    row: dict[str, Any],
    max_tokens: int,
    timeout: float,
) -> tuple[dict[str, Any], float]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": row["messages"],
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "stream": False,
        "chat_template_kwargs": {"thinking": False},
    }
    if "tools" in row:
        payload["tools"] = row["tools"]
        payload["tool_choice"] = row.get("tool_choice", "auto")
    request = urllib.request.Request(
        endpoint.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode(),
        method="POST",
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            result = json.load(response)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "replace")
        raise RuntimeError(f"HTTP {exc.code}: {body}") from exc
    return result, time.perf_counter() - started


def _load_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _load_existing(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    rows = _load_rows(path)
    return {row["id"]: row for row in rows}


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[record["category"]].append(record)
    categories = {}
    for name, items in sorted(groups.items()):
        passed = sum(bool(item["score"]["passed"]) for item in items)
        categories[name] = {
            "count": len(items),
            "passed": passed,
            "accuracy": passed / len(items),
            "mean_latency_seconds": sum(item["latency_seconds"] for item in items) / len(items),
        }
    passed = sum(bool(item["score"]["passed"]) for item in records)
    return {
        "count": len(records),
        "passed": passed,
        "accuracy": passed / len(records),
        "categories": categories,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--endpoint", default="http://127.0.0.1:8890")
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-key", default=os.environ.get("VLLM_API_KEY", ""))
    parser.add_argument("--api-key-file", type=Path)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--progress-every", type=int, default=5)
    args = parser.parse_args()

    api_key = args.api_key_file.read_text().strip() if args.api_key_file else args.api_key
    if not api_key:
        parser.error("API key is required")
    rows = _load_rows(args.dataset)
    existing = _load_existing(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    generated = 0
    started = time.time()
    with args.output.open("a" if args.output.exists() else "w", encoding="utf-8") as handle:
        for index, row in enumerate(rows, 1):
            if row["id"] in existing:
                continue
            response, latency = _post_chat(
                args.endpoint, api_key, args.model, row, args.max_tokens, args.timeout
            )
            choice = response["choices"][0]
            message = choice["message"]
            record = {
                "id": row["id"],
                "category": row["category"],
                "prompt_sha256": hashlib.sha256(
                    json.dumps(row["messages"], ensure_ascii=False, sort_keys=True).encode()
                ).hexdigest(),
                "content": message.get("content"),
                "reasoning_content": message.get("reasoning_content"),
                "tool_calls": message.get("tool_calls"),
                "score": score_message(message, row["expected"]),
                "finish_reason": choice.get("finish_reason"),
                "usage": response.get("usage", {}),
                "latency_seconds": latency,
                "model": response.get("model", args.model),
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            existing[row["id"]] = record
            generated += 1
            if index % args.progress_every == 0 or index == len(rows):
                print(
                    json.dumps(
                        {
                            "processed": index,
                            "total": len(rows),
                            "generated": generated,
                            "elapsed_seconds": round(time.time() - started, 2),
                        }
                    ),
                    flush=True,
                )
    ordered = [existing[row["id"]] for row in rows if row["id"] in existing]
    result = {
        "format": "dspark-crack-capability-eval-v1",
        "dataset": str(args.dataset.resolve()),
        "dataset_sha256": hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
        "model": args.model,
        "summary": summarize(ordered),
        "generated_this_run": generated,
        "output": str(args.output.resolve()),
    }
    args.output.with_suffix(".summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
