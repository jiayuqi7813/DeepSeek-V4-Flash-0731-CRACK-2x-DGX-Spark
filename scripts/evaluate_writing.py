#!/usr/bin/env python3
"""Generate deterministic open-ended writing responses for paired comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _post_chat(
    endpoint: str,
    api_key: str,
    model: str,
    messages: list[dict[str, str]],
    max_tokens: int,
    timeout: float,
) -> tuple[dict[str, Any], float]:
    payload = {
        "model": model,
        "messages": messages,
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


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[record["category"]].append(record)

    def group_summary(items: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "count": len(items),
            "mean_completion_tokens": sum(
                item.get("usage", {}).get("completion_tokens", 0) for item in items
            )
            / len(items),
            "mean_character_count": sum(len(item.get("content") or "") for item in items)
            / len(items),
            "length_finish_count": sum(item.get("finish_reason") == "length" for item in items),
            "mean_latency_seconds": sum(item["latency_seconds"] for item in items) / len(items),
        }

    return {
        "count": len(records),
        "overall": group_summary(records),
        "categories": {
            category: group_summary(items) for category, items in sorted(groups.items())
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--endpoint", default="http://127.0.0.1:8890")
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-key", default=os.environ.get("VLLM_API_KEY", ""))
    parser.add_argument("--api-key-file", type=Path)
    parser.add_argument("--max-tokens", type=int, default=384)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--progress-every", type=int, default=4)
    args = parser.parse_args()

    api_key = args.api_key_file.read_text().strip() if args.api_key_file else args.api_key
    if not api_key:
        parser.error("API key is required")
    rows = _load_jsonl(args.dataset)
    existing = {row["id"]: row for row in _load_jsonl(args.output)} if args.output.exists() else {}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    generated = 0
    started = time.time()
    with args.output.open("a" if args.output.exists() else "w", encoding="utf-8") as handle:
        for index, row in enumerate(rows, 1):
            if row["id"] in existing:
                continue
            response, latency = _post_chat(
                args.endpoint,
                api_key,
                args.model,
                row["messages"],
                args.max_tokens,
                args.timeout,
            )
            choice = response["choices"][0]
            message = choice["message"]
            record = {
                "id": row["id"],
                "category": row["category"],
                "language": row["language"],
                "prompt_sha256": hashlib.sha256(
                    json.dumps(row["messages"], ensure_ascii=False, sort_keys=True).encode()
                ).hexdigest(),
                "content": message.get("content") or "",
                "reasoning_content": message.get("reasoning_content"),
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
    summary = {
        "format": "dspark-crack-writing-eval-v1",
        "dataset": str(args.dataset.resolve()),
        "dataset_sha256": hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
        "model": args.model,
        "max_tokens": args.max_tokens,
        "summary": summarize(ordered),
        "generated_this_run": generated,
    }
    args.output.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
