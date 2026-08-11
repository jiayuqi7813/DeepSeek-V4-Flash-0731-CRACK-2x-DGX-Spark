"""Fast, non-destructive identity checks for the pinned 0731 checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path
from typing import Any


class IdentityError(RuntimeError):
    """Raised when a checkpoint does not match the frozen model identity."""


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise IdentityError(f"cannot read JSON {path}: {exc}") from exc


def read_safetensors_header(path: Path) -> dict[str, Any]:
    """Read only a safetensors header; tensor payloads are never materialized."""
    try:
        with path.open("rb") as handle:
            raw_len = handle.read(8)
            if len(raw_len) != 8:
                raise IdentityError(f"truncated safetensors length: {path}")
            header_len = struct.unpack("<Q", raw_len)[0]
            if header_len <= 0 or header_len > 512 * 1024 * 1024:
                raise IdentityError(f"implausible safetensors header length {header_len}: {path}")
            raw_header = handle.read(header_len)
            if len(raw_header) != header_len:
                raise IdentityError(f"truncated safetensors header: {path}")
    except OSError as exc:
        raise IdentityError(f"cannot read {path}: {exc}") from exc
    try:
        return json.loads(raw_header)
    except json.JSONDecodeError as exc:
        raise IdentityError(f"invalid safetensors header JSON in {path}: {exc}") from exc


def _target_bases(lock: dict[str, Any]) -> list[str]:
    pattern = lock["main_target_pattern"]
    return [pattern.format(layer=i) for i in range(lock["num_hidden_layers"])] + list(
        lock["mtp_targets"]
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_checkpoint(
    model_dir: Path,
    lock: dict[str, Any],
    *,
    full_hash: bool = False,
) -> dict[str, Any]:
    """Verify source identity, tensor inventory, target headers, and optional LFS hashes."""
    model_dir = model_dir.resolve()
    config = _read_json(model_dir / "config.json")
    index = _read_json(model_dir / "model.safetensors.index.json")
    manifest_path = model_dir / ".hf-manifest.json"
    manifest = _read_json(manifest_path) if manifest_path.exists() else None

    failures: list[str] = []

    def expect(actual: Any, expected: Any, label: str) -> None:
        if actual != expected:
            failures.append(f"{label}: expected {expected!r}, got {actual!r}")

    expect(lock.get("format"), "dspark-crack-model-lock-v1", "model lock format")
    expect(config.get("architectures"), [lock["architecture"]], "config.architectures")
    expect(config.get("model_type"), lock["model_type"], "config.model_type")
    expect(config.get("hidden_size"), lock["hidden_size"], "config.hidden_size")
    expect(
        config.get("num_hidden_layers"),
        lock["num_hidden_layers"],
        "config.num_hidden_layers",
    )
    quant = config.get("quantization_config", {})
    expect(quant.get("weight_block_size"), lock["block_size"], "weight_block_size")
    expect(quant.get("fmt"), "e4m3", "quantization_config.fmt")
    expect(quant.get("scale_fmt"), "ue8m0", "quantization_config.scale_fmt")

    weight_map = index.get("weight_map", {})
    expect(len(weight_map), lock["index_tensor_count"], "index tensor count")
    expect(index.get("metadata", {}).get("total_size"), lock["index_total_size"], "total_size")
    shards = sorted(set(weight_map.values()))
    expect(len(shards), lock["shard_count"], "shard count")

    if manifest is not None:
        expect(manifest.get("id"), lock["model_id"], "manifest model id")
        expect(manifest.get("sha"), lock["revision"], "manifest revision")

    metadata_hash_results: list[dict[str, Any]] = []
    for filename, expected_hash in lock.get("metadata_sha256", {}).items():
        if Path(filename).name != filename:
            failures.append(f"metadata hash lock contains non-basename path: {filename!r}")
            continue
        try:
            actual_hash = _sha256(model_dir / filename)
        except OSError as exc:
            failures.append(f"cannot hash metadata file {filename}: {exc}")
            continue
        ok = actual_hash == expected_hash
        metadata_hash_results.append({"file": filename, "sha256": actual_hash, "ok": ok})
        if not ok:
            failures.append(f"metadata sha256 mismatch for {filename}")

    headers: dict[str, dict[str, Any]] = {}
    checked_targets: list[str] = []
    for base in _target_bases(lock):
        weight_name = f"{base}.weight"
        scale_name = f"{base}.scale"
        weight_shard = weight_map.get(weight_name)
        scale_shard = weight_map.get(scale_name)
        if weight_shard is None or scale_shard is None:
            failures.append(f"missing target pair: {base}")
            continue
        if weight_shard != scale_shard:
            failures.append(f"weight/scale split across shards: {base}")
            continue
        shard_path = model_dir / weight_shard
        if weight_shard not in headers:
            headers[weight_shard] = read_safetensors_header(shard_path)
        header = headers[weight_shard]
        for name, dtype, shape in (
            (weight_name, lock["weight_dtype"], lock["weight_shape"]),
            (scale_name, lock["scale_dtype"], lock["scale_shape"]),
        ):
            entry = header.get(name)
            if entry is None:
                failures.append(f"{name} absent from header {weight_shard}")
                continue
            expect(entry.get("dtype"), dtype, f"{name} dtype")
            expect(entry.get("shape"), shape, f"{name} shape")
        checked_targets.append(base)

    hash_results: list[dict[str, Any]] = []
    if full_hash:
        if manifest is None:
            failures.append("--full-hash requested but .hf-manifest.json is absent")
        else:
            siblings = {item.get("rfilename"): item for item in manifest.get("siblings", [])}
            for shard in shards:
                expected = (siblings.get(shard, {}).get("lfs") or {}).get("sha256")
                if not expected:
                    failures.append(f"manifest has no LFS sha256 for {shard}")
                    continue
                actual = _sha256(model_dir / shard)
                hash_results.append({"file": shard, "sha256": actual, "ok": actual == expected})
                if actual != expected:
                    failures.append(f"sha256 mismatch for {shard}")

    result = {
        "ok": not failures,
        "model_dir": str(model_dir),
        "model_id": lock["model_id"],
        "revision": lock["revision"],
        "tensor_count": len(weight_map),
        "shard_count": len(shards),
        "target_count": len(checked_targets),
        "main_target_count": sum(x.startswith("layers.") for x in checked_targets),
        "mtp_target_count": sum(x.startswith("mtp.") for x in checked_targets),
        "metadata_hash_results": metadata_hash_results,
        "full_hash": full_hash,
        "hash_results": hash_results,
        "failures": failures,
    }
    if failures:
        raise IdentityError(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def _default_lock_path() -> Path:
    return Path(__file__).resolve().parents[2] / "model-lock.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("--lock", type=Path, default=_default_lock_path())
    parser.add_argument(
        "--full-hash",
        action="store_true",
        help="hash every shard against the Hugging Face LFS manifest (slow)",
    )
    parser.add_argument("--output", type=Path, help="also write the JSON result")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    lock = _read_json(args.lock)
    try:
        result = verify_checkpoint(args.model_dir, lock, full_hash=args.full_hash)
    except IdentityError as exc:
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
