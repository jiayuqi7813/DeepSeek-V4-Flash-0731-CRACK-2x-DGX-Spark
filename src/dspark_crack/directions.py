"""Estimate and score raw/SRA-cleaned refusal directions from capture files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file


def _portable_path(path: Path) -> str:
    """Prefer a repository-relative path when the target is below the current directory."""
    resolved = path.resolve()
    try:
        return resolved.relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def _unit(vector: torch.Tensor) -> torch.Tensor:
    norm = torch.linalg.vector_norm(vector)
    if not torch.isfinite(norm) or norm < 1e-12:
        raise ValueError("cannot normalize a zero/non-finite direction")
    return vector / norm


def binary_auc(positive: torch.Tensor, negative: torch.Tensor) -> float:
    """Exact Mann–Whitney AUC including a half credit for ties."""
    if positive.numel() == 0 or negative.numel() == 0:
        return float("nan")
    comparisons = positive[:, None] - negative[None, :]
    wins = (comparisons > 0).float().mean()
    ties = (comparisons == 0).float().mean()
    return float((wins + 0.5 * ties).item())


def _effect_size(positive: torch.Tensor, negative: torch.Tensor) -> float:
    pooled_variance = (positive.var(unbiased=True) + negative.var(unbiased=True)) / 2
    pooled = torch.sqrt(pooled_variance).clamp_min(1e-12)
    return float(((positive.mean() - negative.mean()) / pooled).item())


def _sra_clean(
    raw_direction: torch.Tensor,
    benign_activations: torch.Tensor,
    rank: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, float, bool]:
    centered = benign_activations.float() - benign_activations.float().mean(dim=0, keepdim=True)
    q = min(rank, centered.shape[0] - 1, centered.shape[1])
    if q <= 0:
        empty = torch.empty((0, centered.shape[1]), dtype=torch.float32)
        return _unit(raw_direction.float()), empty, 0.0, True
    # pca_lowrank uses a randomized range finder. Fork and seed the CPU RNG so
    # identical captures produce identical SRA artifacts across invocations.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        _, _, v = torch.pca_lowrank(centered, q=q, center=False, niter=3)
    capability_atoms = v.T.contiguous()
    removed = capability_atoms.T @ (capability_atoms @ raw_direction.float())
    cleaned = raw_direction.float() - removed
    removed_fraction = float(
        (
            torch.linalg.vector_norm(removed)
            / torch.linalg.vector_norm(raw_direction).clamp_min(1e-12)
        ).item()
    )
    cleaned_norm = torch.linalg.vector_norm(cleaned)
    if not torch.isfinite(cleaned_norm) or cleaned_norm < 1e-12:
        return torch.zeros_like(cleaned), capability_atoms, removed_fraction, False
    return cleaned / cleaned_norm, capability_atoms, removed_fraction, True


def _load_manifest(path: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            item = json.loads(line)
            sample_id = item["id"]
            if sample_id in result:
                raise ValueError(f"duplicate sample id {sample_id!r} at line {line_no}")
            result[sample_id] = item
    return result


def _load_captures(
    capture_dir: Path,
    manifest: dict[str, dict[str, Any]],
    surface: str,
) -> tuple[torch.Tensor, list[dict[str, Any]], list[str]]:
    tensors: list[torch.Tensor] = []
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for sample_id, row in manifest.items():
        path = capture_dir / f"{sample_id}.safetensors"
        if not path.exists():
            missing.append(sample_id)
            continue
        capture = load_file(str(path), device="cpu")
        if surface not in capture:
            raise ValueError(f"{path} has no {surface!r} tensor")
        tensor = capture[surface]
        if tensor.ndim != 2:
            raise ValueError(
                f"{path}:{surface} must be [layers, hidden], got {tuple(tensor.shape)}"
            )
        tensors.append(tensor)
        rows.append(row)
    if not tensors:
        raise ValueError(f"no captures loaded from {capture_dir}")
    return torch.stack(tensors), rows, missing


def compute_directions(
    capture_dir: Path,
    dataset_path: Path,
    output_path: Path,
    *,
    surface: str,
    sra_rank: int,
    min_auc: float,
    sra_seed: int = 0,
) -> dict[str, Any]:
    manifest = _load_manifest(dataset_path)
    captures, rows, missing = _load_captures(capture_dir, manifest, surface)
    if captures.shape[2] != 4096:
        raise ValueError(f"expected hidden size 4096, got {captures.shape[2]}")
    labels = torch.tensor([int(row["label"]) for row in rows], dtype=torch.bool)
    train = torch.tensor([row["split"] == "train" for row in rows], dtype=torch.bool)
    holdout = torch.tensor([row["split"] == "holdout" for row in rows], dtype=torch.bool)
    for split_name, mask in (("train", train), ("holdout", holdout)):
        if not (labels & mask).any() or not ((~labels) & mask).any():
            raise ValueError(f"{split_name} must contain both labels")

    output_tensors: dict[str, torch.Tensor] = {}
    layer_report: list[dict[str, Any]] = []
    raw_directions: list[torch.Tensor] = []
    cleaned_directions: list[torch.Tensor] = []

    for layer in range(captures.shape[1]):
        activation = captures[:, layer, :].float()
        positive_train = activation[train & labels]
        negative_train = activation[train & ~labels]
        difference = positive_train.mean(dim=0) - negative_train.mean(dim=0)
        difference_norm = torch.linalg.vector_norm(difference)
        raw_valid = bool(torch.isfinite(difference_norm) and difference_norm >= 1e-12)
        if raw_valid:
            raw = difference / difference_norm
            cleaned, capability_atoms, removed_fraction, sra_valid = _sra_clean(
                raw, negative_train, sra_rank, sra_seed + layer
            )
        else:
            raw = torch.zeros_like(difference)
            cleaned = torch.zeros_like(difference)
            capability_atoms = torch.empty((0, difference.shape[0]), dtype=torch.float32)
            removed_fraction = 0.0
            sra_valid = False
        raw_directions.append(raw)
        cleaned_directions.append(cleaned)
        output_tensors[f"layer.{layer:02d}.raw"] = raw.unsqueeze(0)
        output_tensors[f"layer.{layer:02d}.sra"] = cleaned.unsqueeze(0)
        output_tensors[f"layer.{layer:02d}.capability_atoms"] = capability_atoms

        layer_metrics: dict[str, Any] = {
            "layer": layer,
            "valid": raw_valid and sra_valid,
            "sra_removed_fraction": removed_fraction,
        }
        for name, direction, direction_valid in (
            ("raw", raw, raw_valid),
            ("sra", cleaned, sra_valid),
        ):
            if not direction_valid:
                layer_metrics[name] = {
                    "valid": False,
                    "train_auc": 0.5,
                    "holdout_auc": 0.5,
                    "holdout_effect_size": 0.0,
                }
                continue
            train_scores = activation[train] @ direction
            holdout_scores = activation[holdout] @ direction
            layer_metrics[name] = {
                "valid": True,
                "train_auc": binary_auc(train_scores[labels[train]], train_scores[~labels[train]]),
                "holdout_auc": binary_auc(
                    holdout_scores[labels[holdout]], holdout_scores[~labels[holdout]]
                ),
                "holdout_effect_size": _effect_size(
                    holdout_scores[labels[holdout]], holdout_scores[~labels[holdout]]
                ),
            }
        layer_report.append(layer_metrics)

    def make_global(
        directions: list[torch.Tensor], metric_name: str
    ) -> tuple[torch.Tensor, list[int]]:
        # Global artifacts are estimators, not reports: select and weight them
        # from train metrics only. Holdout AUC remains strictly diagnostic.
        aucs = torch.tensor([entry[metric_name]["train_auc"] for entry in layer_report])
        valid = [layer for layer, entry in enumerate(layer_report) if entry[metric_name]["valid"]]
        if not valid:
            raise ValueError(f"{surface} has no valid {metric_name} direction at any layer")
        eligible = [layer for layer in valid if aucs[layer] >= min_auc]
        if not eligible:
            eligible = [max(valid, key=lambda layer: float(aucs[layer].item()))]
        reference_layer = max(
            eligible,
            key=lambda layer: layer_report[layer][metric_name]["train_auc"],
        )
        reference = directions[reference_layer]
        aligned: list[torch.Tensor] = []
        weights: list[float] = []
        for layer in eligible:
            direction = directions[layer]
            if torch.dot(direction, reference) < 0:
                direction = -direction
            aligned.append(direction)
            weights.append(max(layer_report[layer][metric_name]["train_auc"] - 0.5, 1e-6))
        stacked = torch.stack(aligned)
        weight_tensor = torch.tensor(weights, dtype=torch.float32)
        global_direction = _unit((stacked * weight_tensor[:, None]).sum(dim=0))
        return global_direction, eligible

    global_raw, raw_layers = make_global(raw_directions, "raw")
    global_sra, sra_layers = make_global(cleaned_directions, "sra")
    output_tensors["global.raw"] = global_raw.unsqueeze(0)
    output_tensors["global.sra"] = global_sra.unsqueeze(0)

    cleaned_stack = torch.stack(cleaned_directions)
    cosine = cleaned_stack @ cleaned_stack.T
    output_tensors["diagnostics.layer_cosine_sra"] = cosine
    valid_raw_layers = [entry["layer"] for entry in layer_report if entry["raw"]["valid"]]
    valid_sra_layers = [entry["layer"] for entry in layer_report if entry["sra"]["valid"]]
    valid_cosine = cosine[valid_sra_layers][:, valid_sra_layers]
    if len(valid_sra_layers) > 1:
        mean_off_diagonal: float | None = float(
            (
                (valid_cosine.sum() - valid_cosine.diag().sum())
                / (valid_cosine.numel() - valid_cosine.shape[0])
            ).item()
        )
    else:
        mean_off_diagonal = None

    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "format": "dspark-crack-directions-v2",
        "surface": surface,
        "dataset": _portable_path(dataset_path),
        "capture_dir": _portable_path(capture_dir),
        "sample_count": str(len(rows)),
        "sra_rank": str(sra_rank),
        "sra_seed": str(sra_seed),
        "global_selection_metric": "train_auc",
    }
    save_file(output_tensors, str(output_path), metadata=metadata)
    report = {
        "format": "dspark-crack-direction-report-v2",
        "surface": surface,
        "sample_count": len(rows),
        "missing_count": len(missing),
        "missing_ids": missing,
        "train_count": int(train.sum().item()),
        "holdout_count": int(holdout.sum().item()),
        "sra_rank": sra_rank,
        "sra_seed": sra_seed,
        "min_auc": min_auc,
        "global_selection_metric": "train_auc",
        "global_raw_layers": raw_layers,
        "global_sra_layers": sra_layers,
        "valid_raw_layers": valid_raw_layers,
        "valid_sra_layers": valid_sra_layers,
        "inactive_raw_layers": sorted(set(range(captures.shape[1])) - set(valid_raw_layers)),
        "inactive_sra_layers": sorted(set(range(captures.shape[1])) - set(valid_sra_layers)),
        "cross_layer_cosine": {
            "mean_off_diagonal": mean_off_diagonal,
            "min": float(valid_cosine.min().item()),
            "max": float(valid_cosine.max().item()),
        },
        "layers": layer_report,
        "output": _portable_path(output_path),
    }
    report_path = output_path.with_suffix(".report.json")
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--surface", choices=("attn_in", "attn_out", "ffn_out"), default="attn_out")
    parser.add_argument("--sra-rank", type=int, default=8)
    parser.add_argument("--sra-seed", type=int, default=0)
    parser.add_argument("--min-auc", type=float, default=0.65)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.sra_rank < 0:
        raise SystemExit("--sra-rank must be non-negative")
    if not 0.5 <= args.min_auc <= 1.0:
        raise SystemExit("--min-auc must be between 0.5 and 1.0")
    report = compute_directions(
        args.capture_dir,
        args.dataset,
        args.output,
        surface=args.surface,
        sra_rank=args.sra_rank,
        min_auc=args.min_auc,
        sra_seed=args.sra_seed,
    )
    summary = {
        "output": report["output"],
        "sample_count": report["sample_count"],
        "missing_count": report["missing_count"],
        "global_sra_layers": report["global_sra_layers"],
        "mean_cross_layer_cosine": report["cross_layer_cosine"]["mean_off_diagonal"],
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
