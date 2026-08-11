#!/usr/bin/env python3
"""Compare paired base/candidate behavior-evaluation records."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line:
            continue
        item = json.loads(line)
        sample_id = item["id"]
        if sample_id in records:
            raise ValueError(f"duplicate id {sample_id!r} in {path} at line {line_no}")
        records[sample_id] = item
    return records


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
    groups: dict[str, list[str]] = defaultdict(list)
    for sample_id in common:
        base_label = base[sample_id]["label_name"]
        if candidate[sample_id]["label_name"] != base_label:
            raise ValueError(f"label mismatch for {sample_id}")
        groups[base_label].append(sample_id)

    summaries = {}
    for label, sample_ids in sorted(groups.items()):
        transitions = {
            "refusal_to_refusal": 0,
            "refusal_to_compliance": 0,
            "compliance_to_refusal": 0,
            "compliance_to_compliance": 0,
        }
        changed_responses = 0
        base_refusals = 0
        candidate_refusals = 0
        for sample_id in sample_ids:
            before = bool(base[sample_id]["classification"]["refusal"])
            after = bool(candidate[sample_id]["classification"]["refusal"])
            base_refusals += before
            candidate_refusals += after
            if before and after:
                transitions["refusal_to_refusal"] += 1
            elif before:
                transitions["refusal_to_compliance"] += 1
            elif after:
                transitions["compliance_to_refusal"] += 1
            else:
                transitions["compliance_to_compliance"] += 1
            changed_responses += base[sample_id].get("content") != candidate[sample_id].get(
                "content"
            )
        summaries[label] = {
            "count": len(sample_ids),
            "base_refusal_rate": base_refusals / len(sample_ids),
            "candidate_refusal_rate": candidate_refusals / len(sample_ids),
            "refusal_rate_delta": (candidate_refusals - base_refusals) / len(sample_ids),
            "transitions": transitions,
            "mcnemar_exact_p": _mcnemar_exact(
                transitions["refusal_to_compliance"],
                transitions["compliance_to_refusal"],
            ),
            "changed_response_count": changed_responses,
        }

    return {
        "format": "dspark-crack-behavior-comparison-v1",
        "base": str(base_path.resolve()),
        "candidate": str(candidate_path.resolve()),
        "base_count": len(base),
        "candidate_count": len(candidate),
        "paired_count": len(common),
        "missing_from_candidate": sorted(set(base) - set(candidate)),
        "extra_in_candidate": sorted(set(candidate) - set(base)),
        "groups": summaries,
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
