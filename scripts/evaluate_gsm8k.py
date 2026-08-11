#!/usr/bin/env python3
"""Run a deterministic GSM8K subset through an OpenAI-compatible chat API."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import time
import urllib.error
import urllib.request
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

NUMBER_RE = re.compile(r"[-+]?(?:\d[\d,]*)(?:\.\d+)?")


def extract_number(text: str) -> Decimal | None:
    tail = text.rsplit("####", 1)[-1] if "####" in text else text
    matches = NUMBER_RE.findall(tail)
    if not matches and "####" in text:
        matches = NUMBER_RE.findall(text)
    if not matches:
        return None
    try:
        return Decimal(matches[-1].replace(",", ""))
    except InvalidOperation:
        return None


def _post_chat(
    endpoint: str,
    api_key: str,
    model: str,
    question: str,
    max_tokens: int,
    timeout: float,
) -> tuple[dict[str, Any], float]:
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": (
                    "Solve the following grade-school math problem. Use concise reasoning and end "
                    f"with a line in the exact form `#### <number>`.\n\n{question}"
                ),
            }
        ],
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "stream": False,
        "chat_template_kwargs": {"thinking": False},
    }
    request = urllib.request.Request(
        endpoint.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(payload).encode(),
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


def _load_existing(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    records = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line:
            item = json.loads(line)
            records[item["id"]] = item
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--endpoint", default="http://127.0.0.1:8890")
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-key", default=os.environ.get("VLLM_API_KEY", ""))
    parser.add_argument("--api-key-file", type=Path)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--seed", type=int, default=731)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--progress-every", type=int, default=10)
    args = parser.parse_args()

    api_key = args.api_key_file.read_text().strip() if args.api_key_file else args.api_key
    if not api_key:
        parser.error("API key is required")
    all_rows = [
        json.loads(line) for line in args.dataset.read_text(encoding="utf-8").splitlines() if line
    ]
    if args.limit <= 0 or args.limit > len(all_rows):
        parser.error(f"--limit must be between 1 and {len(all_rows)}")
    selected_indices = sorted(random.Random(args.seed).sample(range(len(all_rows)), args.limit))
    rows = [(index, all_rows[index]) for index in selected_indices]
    selection_sha256 = hashlib.sha256(
        json.dumps(selected_indices, separators=(",", ":")).encode()
    ).hexdigest()
    existing = _load_existing(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    generated = 0
    started = time.time()
    with args.output.open("a" if args.output.exists() else "w", encoding="utf-8") as handle:
        for position, (source_index, row) in enumerate(rows, 1):
            sample_id = f"gsm8k-{source_index:04d}"
            if sample_id in existing:
                continue
            response, latency = _post_chat(
                args.endpoint,
                api_key,
                args.model,
                row["question"],
                args.max_tokens,
                args.timeout,
            )
            choice = response["choices"][0]
            message = choice["message"]
            content = message.get("content") or ""
            expected = extract_number(row["answer"].rsplit("####", 1)[-1])
            actual = extract_number(content)
            record = {
                "id": sample_id,
                "source_index": source_index,
                "question_sha256": hashlib.sha256(row["question"].encode()).hexdigest(),
                "content": content,
                "reasoning_content": message.get("reasoning_content"),
                "expected": None if expected is None else str(expected),
                "actual": None if actual is None else str(actual),
                "passed": actual is not None and actual == expected,
                "finish_reason": choice.get("finish_reason"),
                "usage": response.get("usage", {}),
                "latency_seconds": latency,
                "model": response.get("model", args.model),
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            existing[sample_id] = record
            generated += 1
            if position % args.progress_every == 0 or position == len(rows):
                print(
                    json.dumps(
                        {
                            "processed": position,
                            "total": len(rows),
                            "generated": generated,
                            "elapsed_seconds": round(time.time() - started, 2),
                        }
                    ),
                    flush=True,
                )
    ordered = [existing[f"gsm8k-{index:04d}"] for index in selected_indices]
    passed = sum(bool(record["passed"]) for record in ordered)
    result = {
        "format": "dspark-crack-gsm8k-eval-v1",
        "dataset": str(args.dataset.resolve()),
        "dataset_sha256": hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
        "model": args.model,
        "seed": args.seed,
        "selection_sha256": selection_sha256,
        "count": len(ordered),
        "passed": passed,
        "accuracy": passed / len(ordered),
        "mean_latency_seconds": sum(row["latency_seconds"] for row in ordered) / len(ordered),
        "generated_this_run": generated,
        "output": str(args.output.resolve()),
    }
    args.output.with_suffix(".summary.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
