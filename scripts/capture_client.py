#!/usr/bin/env python3
"""Drive single-request capture over an OpenAI-compatible vLLM endpoint."""

from __future__ import annotations

import argparse
import json
import os
import secrets
import struct
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


def _load_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{secrets.token_hex(6)}.tmp")
    temporary.write_text(json.dumps(value), encoding="utf-8")
    os.replace(temporary, path)


def _safetensors_header(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        size = struct.unpack("<Q", handle.read(8))[0]
        return json.loads(handle.read(size))


def _validate_capture(path: Path, sample_id: str) -> None:
    header = _safetensors_header(path)
    metadata = header.get("__metadata__", {})
    if metadata.get("sample_id") != sample_id:
        raise RuntimeError(f"capture metadata id mismatch in {path}")
    for name in ("attn_in", "attn_out", "ffn_out"):
        entry = header.get(name)
        if entry is None:
            raise RuntimeError(f"{path} missing {name}")
        if entry.get("shape") != [43, 4096] or entry.get("dtype") != "F16":
            raise RuntimeError(f"unexpected {name} metadata: {entry}")


def _post_chat(
    endpoint: str,
    api_key: str,
    model: str,
    messages: list[dict[str, str]],
    timeout: float,
) -> dict[str, Any]:
    payload = json.dumps(
        {
            "model": model,
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": 1,
            "stream": False,
            "chat_template_kwargs": {"thinking": False},
        }
    ).encode()
    request = urllib.request.Request(
        endpoint.rstrip("/") + "/v1/chat/completions",
        data=payload,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "replace")
        raise RuntimeError(f"HTTP {exc.code}: {body}") from exc


def run(args: argparse.Namespace) -> dict[str, Any]:
    rows = _load_rows(args.dataset)
    if args.limit is not None:
        rows = rows[: args.limit]
    api_key = args.api_key
    if args.api_key_file:
        api_key = args.api_key_file.read_text(encoding="utf-8").strip()
    if not api_key:
        raise RuntimeError("API key is required")

    output_dir = args.capture_root / "samples"
    output_dir.mkdir(parents=True, exist_ok=True)
    control_path = args.capture_root / "control.json"
    started = time.time()
    captured = 0
    skipped = 0
    failures: list[dict[str, str]] = []

    for index, row in enumerate(rows, 1):
        sample_id = row["id"]
        target = output_dir / f"{sample_id}.safetensors"
        if target.exists() and args.resume:
            _validate_capture(target, sample_id)
            skipped += 1
            continue
        nonce = secrets.token_hex(12)
        _atomic_json(
            control_path,
            {
                "format": "dspark-crack-control-v1",
                "armed": True,
                "sample_id": sample_id,
                "nonce": nonce,
                "prompt_sha256": __import__("hashlib")
                .sha256(json.dumps(row["messages"], ensure_ascii=False, sort_keys=True).encode())
                .hexdigest(),
                "written_at": time.time(),
            },
        )
        try:
            _post_chat(args.endpoint, api_key, args.model, row["messages"], args.request_timeout)
            deadline = time.monotonic() + args.capture_timeout
            while not target.exists() and time.monotonic() < deadline:
                time.sleep(0.05)
            if not target.exists():
                raise TimeoutError(f"capture file did not appear within {args.capture_timeout}s")
            _validate_capture(target, sample_id)
            captured += 1
        except Exception as exc:  # noqa: BLE001 - failures are recorded and then stop by default
            failures.append({"id": sample_id, "error": repr(exc)})
            if not args.continue_on_error:
                break
        if index % args.progress_every == 0 or index == len(rows):
            elapsed = max(time.time() - started, 1e-6)
            print(
                json.dumps(
                    {
                        "processed": index,
                        "total": len(rows),
                        "captured": captured,
                        "skipped": skipped,
                        "failures": len(failures),
                        "samples_per_second": round((captured + skipped) / elapsed, 3),
                    }
                ),
                flush=True,
            )

    _atomic_json(control_path, {"format": "dspark-crack-control-v1", "armed": False})
    result = {
        "format": "dspark-crack-capture-run-v1",
        "dataset": str(args.dataset.resolve()),
        "endpoint": args.endpoint,
        "model": args.model,
        "total": len(rows),
        "captured": captured,
        "skipped": skipped,
        "failures": failures,
        "elapsed_seconds": time.time() - started,
        "capture_root": str(args.capture_root.resolve()),
    }
    run_path = args.capture_root / "capture-run.json"
    run_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--capture-root", type=Path, required=True)
    parser.add_argument("--endpoint", default="http://127.0.0.1:8890")
    parser.add_argument("--model", default="deepseek-v4-flash-0731")
    parser.add_argument("--api-key", default=os.environ.get("VLLM_API_KEY", ""))
    parser.add_argument("--api-key-file", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--request-timeout", type=float, default=180.0)
    parser.add_argument("--capture-timeout", type=float, default=30.0)
    parser.add_argument("--progress-every", type=int, default=10)
    args = parser.parse_args()
    result = run(args)
    print(json.dumps(result, indent=2))
    return 1 if result["failures"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
