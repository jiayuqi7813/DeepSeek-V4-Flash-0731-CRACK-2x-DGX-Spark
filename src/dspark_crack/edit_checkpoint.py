"""Create a copy-on-write 0731 FP8 checkpoint with output-subspace edits."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from .identity import IdentityError, verify_checkpoint
from .quantization import (
    dequantize_block_fp8,
    e8m0_to_float,
    expand_block_scales,
    project_output_subspace,
    quantize_block_fp8,
)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_layers(spec: str, num_layers: int) -> list[int]:
    layers: set[int] = set()
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            start_text, end_text = item.split("-", 1)
            start, end = int(start_text), int(end_text)
            if end < start:
                raise ValueError(f"descending layer range: {item}")
            layers.update(range(start, end + 1))
        else:
            layers.add(int(item))
    invalid = sorted(layer for layer in layers if layer < 0 or layer >= num_layers)
    if invalid:
        raise ValueError(f"layers outside 0..{num_layers - 1}: {invalid}")
    if not layers:
        raise ValueError("no layers selected")
    return sorted(layers)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _prepare_output(source: Path, output: Path, shards: list[str]) -> None:
    if output.exists():
        if any(output.iterdir()) if output.is_dir() else True:
            raise FileExistsError(f"output must not exist or must be empty: {output}")
    else:
        output.mkdir(parents=True)
    shard_set = set(shards)
    for source_path in source.iterdir():
        if source_path.name in {"CRACK_EDIT_MANIFEST.json", "CRACK_EDIT_REPORT.json"}:
            continue
        target = output / source_path.name
        if source_path.is_dir():
            shutil.copytree(source_path, target, symlinks=True)
        elif source_path.name in shard_set:
            try:
                os.link(source_path, target)
            except OSError:
                shutil.copy2(source_path, target)
        else:
            shutil.copy2(source_path, target)


def _direction_for(
    directions: dict[str, torch.Tensor],
    mode: str,
    layer: int,
) -> torch.Tensor:
    if mode.startswith("global."):
        key = mode
    elif mode.startswith("layer."):
        suffix = mode.split(".", 1)[1]
        key = f"layer.{layer:02d}.{suffix}"
    else:
        raise ValueError(f"unsupported direction mode {mode!r}")
    if key not in directions:
        raise KeyError(f"direction artifact has no key {key!r}")
    tensor = directions[key].float()
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 2 or tensor.shape[1] != 4096:
        raise ValueError(f"{key} must be [rank,4096], got {tuple(tensor.shape)}")
    return tensor


def _metadata(path: Path) -> dict[str, str] | None:
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        return handle.metadata()


def edit_checkpoint(
    source: Path,
    output: Path,
    lock_path: Path,
    direction_path: Path,
    *,
    layers: list[int],
    direction_mode: str,
    strength: float,
    preserve_row_norm: bool,
    mtp: str,
    scale_policy: str,
) -> dict[str, Any]:
    source = source.resolve()
    output = output.resolve()
    lock = _read_json(lock_path)
    identity = verify_checkpoint(source, lock)
    index = _read_json(source / "model.safetensors.index.json")
    weight_map: dict[str, str] = index["weight_map"]
    shards = sorted(set(weight_map.values()))
    directions = load_file(str(direction_path), device="cpu")
    direction_hash = _sha256(direction_path)

    targets: list[tuple[str, torch.Tensor]] = []
    for layer in layers:
        targets.append(
            (
                f"layers.{layer}.attn.wo_b",
                _direction_for(directions, direction_mode, layer),
            )
        )
    if mtp == "matched":
        source_layer = layers[-1]
        mtp_direction = _direction_for(directions, direction_mode, source_layer)
        targets.extend((base, mtp_direction) for base in lock["mtp_targets"])
    elif mtp != "stock":
        raise ValueError("mtp must be 'stock' or 'matched'")
    if scale_policy not in {"fixed", "expand"}:
        raise ValueError("scale_policy must be 'fixed' or 'expand'")

    by_shard: dict[str, list[tuple[str, torch.Tensor]]] = defaultdict(list)
    for base, direction in targets:
        shard = weight_map.get(f"{base}.weight")
        if shard is None or weight_map.get(f"{base}.scale") != shard:
            raise IdentityError(f"invalid weight map for {base}")
        by_shard[shard].append((base, direction))

    _prepare_output(source, output, shards)
    report: dict[str, Any] = {
        "format": "dspark-crack-edit-report-v1",
        "status": "running",
        "source": str(source),
        "source_identity": identity,
        "output": str(output),
        "direction_file": str(direction_path.resolve()),
        "direction_sha256": direction_hash,
        "direction_mode": direction_mode,
        "layers": layers,
        "strength": strength,
        "preserve_row_norm": preserve_row_norm,
        "mtp": mtp,
        "scale_policy": scale_policy,
        "started_at": time.time(),
        "targets": [],
        "edited_shards": [],
    }
    report_path = output / "CRACK_EDIT_REPORT.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    try:
        for shard_index, (shard, shard_targets) in enumerate(sorted(by_shard.items()), 1):
            source_shard = source / shard
            output_shard = output / shard
            tensors = load_file(str(source_shard), device="cpu")
            shard_metadata = _metadata(source_shard)
            for base, direction in shard_targets:
                weight_name = f"{base}.weight"
                scale_name = f"{base}.scale"
                quant_weight = tensors[weight_name]
                quant_scale = tensors[scale_name]
                dequantized = dequantize_block_fp8(quant_weight, quant_scale, lock["block_size"])
                edited, metrics = project_output_subspace(
                    dequantized,
                    direction,
                    strength=strength,
                    preserve_row_norm=preserve_row_norm,
                )
                original_scale_float = expand_block_scales(
                    e8m0_to_float(quant_scale), edited.shape, lock["block_size"]
                )
                original_scaled_abs = edited.abs() / original_scale_float
                metrics["original_scale_overflow_values"] = int(
                    torch.count_nonzero(original_scaled_abs > 448.0).item()
                )
                metrics["max_original_scaled_abs"] = float(original_scaled_abs.max().item())
                requant_weight, requant_scale = quantize_block_fp8(
                    edited,
                    lock["block_size"],
                    fixed_scale=quant_scale if scale_policy == "fixed" else None,
                    minimum_scale=quant_scale if scale_policy == "expand" else None,
                )
                roundtrip = dequantize_block_fp8(requant_weight, requant_scale, lock["block_size"])
                metrics["quantization_relative_error"] = float(
                    (
                        torch.linalg.vector_norm(roundtrip - edited)
                        / torch.linalg.vector_norm(edited).clamp_min(1e-12)
                    ).item()
                )
                metrics.update(
                    {
                        "base": base,
                        "shard": shard,
                        "direction_rank": int(direction.shape[0]),
                        "weight_shape": list(quant_weight.shape),
                        "scale_shape": list(quant_scale.shape),
                        "changed_weight_values": int(
                            torch.count_nonzero(
                                quant_weight.float() != requant_weight.float()
                            ).item()
                        ),
                        "changed_scale_blocks": int(
                            torch.count_nonzero(
                                quant_scale.view(torch.uint8) != requant_scale.view(torch.uint8)
                            ).item()
                        ),
                    }
                )
                tensors[weight_name] = requant_weight
                tensors[scale_name] = requant_scale
                report["targets"].append(metrics)
                del dequantized, edited, roundtrip

            temporary = output_shard.with_name(f".{shard}.editing.tmp")
            save_file(tensors, str(temporary), metadata=shard_metadata)
            os.replace(temporary, output_shard)
            shard_record = {
                "file": shard,
                "index": shard_index,
                "count": len(by_shard),
                "size": output_shard.stat().st_size,
                "sha256": _sha256(output_shard),
            }
            report["edited_shards"].append(shard_record)
            report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
            del tensors

        report["status"] = "complete"
        report["completed_at"] = time.time()
        report["elapsed_seconds"] = report["completed_at"] - report["started_at"]
        report["target_count"] = len(report["targets"])
        report["edited_shard_count"] = len(report["edited_shards"])
        report["mean_relative_edit_norm"] = sum(
            item["relative_edit_norm"] for item in report["targets"]
        ) / len(report["targets"])
        report["max_quantization_relative_error"] = max(
            item["quantization_relative_error"] for item in report["targets"]
        )
        manifest = {
            "format": "dspark-crack-checkpoint-v1",
            "source_model": lock["model_id"],
            "source_revision": lock["revision"],
            "direction_sha256": direction_hash,
            "direction_mode": direction_mode,
            "layers": layers,
            "strength": strength,
            "preserve_row_norm": preserve_row_norm,
            "mtp": mtp,
            "scale_policy": scale_policy,
            "report": "CRACK_EDIT_REPORT.json",
        }
        (output / "CRACK_EDIT_MANIFEST.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        return report
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = repr(exc)
        report["failed_at"] = time.time()
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        raise


def _default_lock_path() -> Path:
    return Path(__file__).resolve().parents[2] / "model-lock.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--directions", type=Path, required=True)
    parser.add_argument("--lock", type=Path, default=_default_lock_path())
    parser.add_argument("--layers", default="10-42")
    parser.add_argument(
        "--direction-mode",
        choices=("global.raw", "global.sra", "layer.raw", "layer.sra"),
        default="global.sra",
    )
    parser.add_argument("--strength", type=float, default=2.0)
    parser.add_argument(
        "--preserve-row-norm",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--mtp", choices=("stock", "matched"), default="stock")
    parser.add_argument("--scale-policy", choices=("fixed", "expand"), default="fixed")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    lock = _read_json(args.lock)
    layers = parse_layers(args.layers, lock["num_hidden_layers"])
    report = edit_checkpoint(
        args.source,
        args.output,
        args.lock,
        args.directions,
        layers=layers,
        direction_mode=args.direction_mode,
        strength=args.strength,
        preserve_row_norm=args.preserve_row_norm,
        mtp=args.mtp,
        scale_policy=args.scale_policy,
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "output": report["output"],
                "target_count": report["target_count"],
                "edited_shard_count": report["edited_shard_count"],
                "elapsed_seconds": report["elapsed_seconds"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
