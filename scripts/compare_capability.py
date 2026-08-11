#!/usr/bin/env python3
"""Compare paired pass/fail records from capability, GSM8K, or HumanEval runs."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, dict[str, Any]]:
    records = {}
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line:
            continue
        item = json.loads(line)
        sample_id = item.get("id", item.get("task_id"))
        if not sample_id:
            raise ValueError(f"record without id in {path}:{line_no}")
        if sample_id in records:
            raise ValueError(f"duplicate id {sample_id!r} in {path}:{line_no}")
        records[sample_id] = item
    return records


def _passed(item: dict[str, Any]) -> bool:
    if "passed" in item:
        return bool(item["passed"])
    if isinstance(item.get("score"), dict) and "passed" in item["score"]:
        return bool(item["score"]["passed"])
    raise ValueError("record has no pass/fail field")


def _mcnemar_exact(base_only: int, candidate_only: int) -> float:
    discordant = base_only + candidate_only
    if discordant == 0:
        return 1.0
    tail = min(base_only, candidate_only)
    probability = sum(math.comb(discordant, k) for k in range(tail + 1)) / (2**discordant)
    return min(1.0, 2.0 * probability)


def compare(base_path: Path, candidate_path: Path) -> dict[str, Any]:
    base = _load(base_path)
    candidate = _load(candidate_path)
    common = sorted(set(base) & set(candidate))
    if not common:
        raise ValueError("no common record ids")
    transitions = {
        "pass_to_pass": 0,
        "pass_to_fail": 0,
        "fail_to_pass": 0,
        "fail_to_fail": 0,
    }
    for sample_id in common:
        before = _passed(base[sample_id])
        after = _passed(candidate[sample_id])
        if before and after:
            transitions["pass_to_pass"] += 1
        elif before:
            transitions["pass_to_fail"] += 1
        elif after:
            transitions["fail_to_pass"] += 1
        else:
            transitions["fail_to_fail"] += 1
    base_passed = transitions["pass_to_pass"] + transitions["pass_to_fail"]
    candidate_passed = transitions["pass_to_pass"] + transitions["fail_to_pass"]
    return {
        "format": "dspark-crack-capability-comparison-v1",
        "base": str(base_path.resolve()),
        "candidate": str(candidate_path.resolve()),
        "base_count": len(base),
        "candidate_count": len(candidate),
        "paired_count": len(common),
        "base_accuracy": base_passed / len(common),
        "candidate_accuracy": candidate_passed / len(common),
        "accuracy_delta": (candidate_passed - base_passed) / len(common),
        "transitions": transitions,
        "mcnemar_exact_p": _mcnemar_exact(
            transitions["pass_to_fail"], transitions["fail_to_pass"]
        ),
        "missing_from_candidate": sorted(set(base) - set(candidate)),
        "extra_in_candidate": sorted(set(candidate) - set(base)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = compare(args.base, args.candidate)
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
