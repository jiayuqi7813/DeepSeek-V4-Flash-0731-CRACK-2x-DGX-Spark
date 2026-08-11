#!/usr/bin/env python3
"""Score HumanEval generations inside a resource-limited, networkless container."""

from __future__ import annotations

import argparse
import gzip
import json
import os
import resource
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


def _load_gzip_jsonl(path: Path) -> dict[str, dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return {item["task_id"]: item for line in handle if (item := json.loads(line))}


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _limit_child() -> None:
    resource.setrlimit(resource.RLIMIT_CPU, (5, 5))
    resource.setrlimit(resource.RLIMIT_FSIZE, (1_048_576, 1_048_576))
    resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
    resource.setrlimit(resource.RLIMIT_NPROC, (32, 32))


def _program(problem: dict[str, Any], code: str) -> str:
    if f"def {problem['entry_point']}" in code:
        implementation = code
    else:
        implementation = problem["prompt"] + "\n" + code
    return (
        implementation
        + "\n\n"
        + problem["test"]
        + f"\n\ncheck({problem['entry_point']})\n"
    )


def _run_one(problem: dict[str, Any], code: str, timeout: float) -> dict[str, Any]:
    program = _program(problem, code)
    with tempfile.TemporaryDirectory(prefix="humaneval-") as directory:
        script = Path(directory) / "candidate.py"
        script.write_text(program, encoding="utf-8")
        try:
            completed = subprocess.run(
                [sys.executable, "-I", str(script)],
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                timeout=timeout,
                env={"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "PYTHONHASHSEED": "0"},
                preexec_fn=_limit_child,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return {"passed": False, "status": "timeout", "stderr": ""}
    stderr = completed.stderr[-4000:]
    return {
        "passed": completed.returncode == 0,
        "status": "passed" if completed.returncode == 0 else "failed",
        "returncode": completed.returncode,
        "stderr": stderr,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--completions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=6.0)
    args = parser.parse_args()
    problems = _load_gzip_jsonl(args.dataset)
    completions = _load_jsonl(args.completions)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    results = []
    with args.output.open("w", encoding="utf-8") as handle:
        for index, completion in enumerate(completions, 1):
            task_id = completion["task_id"]
            if task_id not in problems:
                raise KeyError(f"unknown task_id: {task_id}")
            score = _run_one(problems[task_id], completion["code"], args.timeout)
            record = {"task_id": task_id, **score}
            results.append(record)
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            print(json.dumps({"processed": index, "total": len(completions)}), flush=True)
    passed = sum(bool(item["passed"]) for item in results)
    summary = {
        "format": "dspark-crack-humaneval-score-v1",
        "count": len(results),
        "passed": passed,
        "pass_at_1": passed / len(results),
        "output": str(args.output),
    }
    args.output.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
