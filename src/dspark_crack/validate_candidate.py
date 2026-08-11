"""Validate an edited checkpoint against its source at tensor granularity."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open

from .identity import IdentityError, verify_checkpoint


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compare_rewritten_shard(
    source_path: Path,
    candidate_path: Path,
    edited_keys: set[str],
    required_changed_keys: set[str] | None = None,
) -> dict[str, Any]:
    """Require all non-target tensors to remain bit-exact in a rewritten shard."""
    unchanged_failures: list[str] = []
    edited_equal: list[str] = []
    required_changed_keys = edited_keys if required_changed_keys is None else required_changed_keys
    checked_unchanged = 0
    checked_edited = 0
    with (
        safe_open(str(source_path), framework="pt", device="cpu") as source,
        safe_open(str(candidate_path), framework="pt", device="cpu") as candidate,
    ):
        source_keys = set(source.keys())
        candidate_keys = set(candidate.keys())
        if source_keys != candidate_keys:
            missing = sorted(source_keys - candidate_keys)
            extra = sorted(candidate_keys - source_keys)
            return {
                "ok": False,
                "missing_keys": missing,
                "extra_keys": extra,
                "unchanged_failures": [],
                "edited_equal": [],
                "required_unchanged": [],
                "checked_unchanged": 0,
                "checked_edited": 0,
            }
        if source.metadata() != candidate.metadata():
            unchanged_failures.append("__metadata__")
        for key in sorted(source_keys):
            before = source.get_tensor(key)
            after = candidate.get_tensor(key)
            if before.dtype != after.dtype or before.shape != after.shape:
                unchanged_failures.append(key)
                continue
            equal = torch.equal(before, after)
            if key in edited_keys:
                checked_edited += 1
                if equal:
                    edited_equal.append(key)
            else:
                checked_unchanged += 1
                if not equal:
                    unchanged_failures.append(key)
    required_unchanged = sorted(required_changed_keys & set(edited_equal))
    return {
        "ok": not unchanged_failures and not required_unchanged,
        "missing_keys": [],
        "extra_keys": [],
        "unchanged_failures": unchanged_failures,
        "edited_equal": edited_equal,
        "required_unchanged": required_unchanged,
        "checked_unchanged": checked_unchanged,
        "checked_edited": checked_edited,
    }


def manifest_consistency_failures(
    lock: dict[str, Any],
    edit_report: dict[str, Any],
    edit_manifest: dict[str, Any],
) -> list[str]:
    """Check that the compact release manifest describes the measured edit report."""
    expected_fields = {
        "format": "dspark-crack-checkpoint-v1",
        "source_model": lock["model_id"],
        "source_revision": lock["revision"],
        "direction_sha256": edit_report.get("direction_sha256"),
        "direction_mode": edit_report.get("direction_mode"),
        "layers": edit_report.get("layers"),
        "strength": edit_report.get("strength"),
        "preserve_row_norm": edit_report.get("preserve_row_norm"),
        "mtp": edit_report.get("mtp"),
        "scale_policy": edit_report.get("scale_policy"),
        "report": "CRACK_EDIT_REPORT.json",
    }
    failures = [
        f"edit manifest {field} does not match report/source lock"
        for field, expected in expected_fields.items()
        if edit_manifest.get(field) != expected
    ]
    layers = edit_report.get("layers")
    if not isinstance(layers, list) or not all(isinstance(layer, int) for layer in layers):
        failures.append("edit report layers must be an integer list")
        return failures
    expected_bases = [f"layers.{layer}.attn.wo_b" for layer in layers]
    if edit_report.get("mtp") == "matched":
        expected_bases.extend(lock["mtp_targets"])
    elif edit_report.get("mtp") != "stock":
        failures.append(f"unsupported edit report mtp mode: {edit_report.get('mtp')!r}")
    actual_bases = [item.get("base") for item in edit_report.get("targets", [])]
    if len(actual_bases) != len(set(actual_bases)):
        failures.append("edit report contains duplicate targets")
    if len(actual_bases) != len(expected_bases) or set(actual_bases) != set(expected_bases):
        failures.append("edit report targets do not match manifest layers/MTP mode")
    return failures


def validate_candidate(
    source_dir: Path,
    candidate_dir: Path,
    lock_path: Path,
) -> dict[str, Any]:
    source_dir = source_dir.resolve()
    candidate_dir = candidate_dir.resolve()
    lock = _read_json(lock_path)
    source_identity = verify_checkpoint(source_dir, lock)
    candidate_identity = verify_checkpoint(candidate_dir, lock)
    edit_report = _read_json(candidate_dir / "CRACK_EDIT_REPORT.json")
    edit_manifest = _read_json(candidate_dir / "CRACK_EDIT_MANIFEST.json")
    failures: list[str] = []
    if edit_report.get("status") != "complete":
        failures.append(f"edit report status is {edit_report.get('status')!r}")
    failures.extend(manifest_consistency_failures(lock, edit_report, edit_manifest))

    source_index = _read_json(source_dir / "model.safetensors.index.json")
    candidate_index = _read_json(candidate_dir / "model.safetensors.index.json")
    if source_index != candidate_index:
        failures.append("model.safetensors.index.json changed")
    weight_map: dict[str, str] = source_index["weight_map"]
    shards = sorted(set(weight_map.values()))

    edited_bases = [item["base"] for item in edit_report.get("targets", [])]
    edited_keys_by_shard: dict[str, set[str]] = {}
    required_changed_by_shard: dict[str, set[str]] = {}
    for base in edited_bases:
        weight_key = f"{base}.weight"
        scale_key = f"{base}.scale"
        shard = weight_map.get(weight_key)
        if shard is None or weight_map.get(scale_key) != shard:
            failures.append(f"invalid target mapping for {base}")
            continue
        edited_keys_by_shard.setdefault(shard, set()).add(weight_key)
        required_changed_by_shard.setdefault(shard, set()).add(weight_key)
        if edit_report.get("scale_policy") != "fixed":
            edited_keys_by_shard[shard].add(scale_key)

    report_shards = {item["file"]: item for item in edit_report.get("edited_shards", [])}
    if set(report_shards) != set(edited_keys_by_shard):
        failures.append("edited shard set does not match target mapping")

    shard_results: list[dict[str, Any]] = []
    hardlinked_unchanged = 0
    hashed_unchanged = 0
    checked_unchanged_tensors = 0
    checked_edited_tensors = 0
    for index, shard in enumerate(shards, 1):
        source_path = source_dir / shard
        candidate_path = candidate_dir / shard
        if not candidate_path.exists():
            failures.append(f"candidate is missing {shard}")
            continue
        if shard in edited_keys_by_shard:
            print(
                f"validating rewritten shard {index}/{len(shards)}: {shard}",
                file=sys.stderr,
                flush=True,
            )
            comparison = compare_rewritten_shard(
                source_path,
                candidate_path,
                edited_keys_by_shard[shard],
                required_changed_by_shard[shard],
            )
            actual_hash = _sha256(candidate_path)
            expected_hash = report_shards.get(shard, {}).get("sha256")
            if actual_hash != expected_hash:
                comparison["hash_failure"] = {
                    "expected": expected_hash,
                    "actual": actual_hash,
                }
                comparison["ok"] = False
            checked_unchanged_tensors += comparison["checked_unchanged"]
            checked_edited_tensors += comparison["checked_edited"]
            if not comparison["ok"]:
                failures.append(f"rewritten shard validation failed: {shard}")
            shard_results.append({"file": shard, "edited": True, **comparison})
            continue

        source_stat = source_path.stat()
        candidate_stat = candidate_path.stat()
        same_inode = (
            source_stat.st_dev == candidate_stat.st_dev
            and source_stat.st_ino == candidate_stat.st_ino
        )
        if same_inode:
            hardlinked_unchanged += 1
            shard_results.append({"file": shard, "edited": False, "hardlinked": True})
        else:
            source_hash = _sha256(source_path)
            candidate_hash = _sha256(candidate_path)
            hashed_unchanged += 1
            ok = source_hash == candidate_hash
            if not ok:
                failures.append(f"unchanged shard differs: {shard}")
            shard_results.append(
                {
                    "file": shard,
                    "edited": False,
                    "hardlinked": False,
                    "ok": ok,
                    "source_sha256": source_hash,
                    "candidate_sha256": candidate_hash,
                }
            )

    result = {
        "format": "dspark-crack-candidate-validation-v1",
        "ok": not failures,
        "source": str(source_dir),
        "candidate": str(candidate_dir),
        "source_identity": source_identity,
        "candidate_identity": candidate_identity,
        "edit_manifest": edit_manifest,
        "edited_target_count": len(edited_bases),
        "edited_shard_count": len(edited_keys_by_shard),
        "hardlinked_unchanged_shard_count": hardlinked_unchanged,
        "hashed_unchanged_shard_count": hashed_unchanged,
        "checked_unchanged_tensor_count": checked_unchanged_tensors,
        "checked_edited_tensor_count": checked_edited_tensors,
        "shards": shard_results,
        "failures": failures,
    }
    if failures:
        raise IdentityError(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def _default_lock_path() -> Path:
    return Path(__file__).resolve().parents[2] / "model-lock.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--lock", type=Path, default=_default_lock_path())
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = validate_candidate(args.source, args.candidate, args.lock)
    except (IdentityError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(str(exc))
        return 1
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
