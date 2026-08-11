#!/usr/bin/env python3
"""Generate a deterministic HumanEval subset through an OpenAI-compatible API."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import random
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

FENCE_RE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_code(text: str) -> str:
    matches = FENCE_RE.findall(text)
    if matches:
        return max(matches, key=len).strip()
    return text.strip()


def _load_dataset(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _load_existing(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    records = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line:
            item = json.loads(line)
            records[item["task_id"]] = item
    return records


def _post_chat(
    endpoint: str,
    api_key: str,
    model: str,
    prompt: str,
    max_tokens: int,
    timeout: float,
) -> tuple[dict[str, Any], float]:
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": (
                    "Complete the Python function below. Return only the complete implementation "
                    "in one Python code block. Preserve the function name and signature. Do not "
                    f"include tests or explanations.\n\n```python\n{prompt}\n```"
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--endpoint", default="http://127.0.0.1:8890")
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-key", default=os.environ.get("VLLM_API_KEY", ""))
    parser.add_argument("--api-key-file", type=Path)
    parser.add_argument("--limit", type=int, default=32)
    parser.add_argument("--seed", type=int, default=731)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--progress-every", type=int, default=5)
    args = parser.parse_args()

    api_key = args.api_key_file.read_text().strip() if args.api_key_file else args.api_key
    if not api_key:
        parser.error("API key is required")
    all_rows = _load_dataset(args.dataset)
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
            task_id = row["task_id"]
            if task_id in existing:
                continue
            response, latency = _post_chat(
                args.endpoint,
                api_key,
                args.model,
                row["prompt"],
                args.max_tokens,
                args.timeout,
            )
            choice = response["choices"][0]
            message = choice["message"]
            content = message.get("content") or ""
            record = {
                "task_id": task_id,
                "source_index": source_index,
                "prompt_sha256": hashlib.sha256(row["prompt"].encode()).hexdigest(),
                "content": content,
                "code": extract_code(content),
                "reasoning_content": message.get("reasoning_content"),
                "finish_reason": choice.get("finish_reason"),
                "usage": response.get("usage", {}),
                "latency_seconds": latency,
                "model": response.get("model", args.model),
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            existing[task_id] = record
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
    result = {
        "format": "dspark-crack-humaneval-generations-v1",
        "dataset": str(args.dataset.resolve()),
        "dataset_sha256": hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
        "model": args.model,
        "seed": args.seed,
        "selection_sha256": selection_sha256,
        "count": len(rows),
        "generated_this_run": generated,
        "output": str(args.output.resolve()),
    }
    args.output.with_suffix(".generation-summary.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
