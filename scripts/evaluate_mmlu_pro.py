#!/usr/bin/env python3
"""Run a deterministic MMLU-Pro selection through an OpenAI-compatible API."""

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
from pathlib import Path
from typing import Any

ANSWER_RE = re.compile(r"(?:answer|答案)\s*(?:is|是)?\s*[:：]?\s*\(?([A-J])\)?", re.I)
LETTER_RE = re.compile(r"\b([A-J])\b")


def extract_answer(text: str) -> str | None:
    matches = ANSWER_RE.findall(text)
    if matches:
        return matches[-1].upper()
    letters = LETTER_RE.findall(text.upper())
    return letters[-1] if letters else None


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _load_existing(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    return {row["id"]: row for row in _load_jsonl(path)}


def _post_chat(
    endpoint: str,
    api_key: str,
    model: str,
    row: dict[str, Any],
    max_tokens: int,
    timeout: float,
) -> tuple[dict[str, Any], float]:
    choices = "\n".join(
        f"{chr(65 + index)}. {option}" for index, option in enumerate(row["options"])
    )
    prompt = (
        "Answer this multiple-choice question. Output only `Answer: <letter>` and do not "
        f"show reasoning or explanation.\n\n{row['question']}\n\n{choices}"
    )
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "stream": False,
        "chat_template_kwargs": {"thinking": False},
    }
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--endpoint", default="http://127.0.0.1:8890")
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-key", default=os.environ.get("VLLM_API_KEY", ""))
    parser.add_argument("--api-key-file", type=Path)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--progress-every", type=int, default=10)
    args = parser.parse_args()

    api_key = args.api_key_file.read_text().strip() if args.api_key_file else args.api_key
    if not api_key:
        parser.error("API key is required")
    rows = _load_jsonl(args.dataset)
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
            content = message.get("content") or ""
            actual = extract_answer(content)
            record = {
                "id": row["id"],
                "category": row["category"],
                "question_sha256": hashlib.sha256(row["question"].encode()).hexdigest(),
                "content": content,
                "reasoning_content": message.get("reasoning_content"),
                "expected": row["answer"],
                "actual": actual,
                "passed": actual == row["answer"],
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
    ordered = [existing[row["id"]] for row in rows]
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in ordered:
        groups[record["category"]].append(record)
    categories = {
        name: {
            "count": len(items),
            "passed": sum(bool(item["passed"]) for item in items),
            "accuracy": sum(bool(item["passed"]) for item in items) / len(items),
        }
        for name, items in sorted(groups.items())
    }
    passed = sum(bool(item["passed"]) for item in ordered)
    summary = {
        "format": "dspark-crack-mmlu-pro-eval-v1",
        "dataset": str(args.dataset.resolve()),
        "dataset_sha256": hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
        "model": args.model,
        "count": len(ordered),
        "passed": passed,
        "accuracy": passed / len(ordered),
        "categories": categories,
        "generated_this_run": generated,
    }
    args.output.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
