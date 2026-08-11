"""Native FP8 E4M3 + UE8M0 128x128 block quantization helpers."""

from __future__ import annotations

from collections.abc import Sequence

import torch

FP8_MAX = 448.0
AMAX_MIN = 1e-4


def e8m0_to_float(scale: torch.Tensor) -> torch.Tensor:
    """Decode exponent-only E8M0 values using the representation used by vLLM."""
    if scale.dtype not in (torch.float8_e8m0fnu, torch.uint8):
        raise TypeError(f"expected E8M0/uint8 scale, got {scale.dtype}")
    exponent_bytes = scale.contiguous().view(torch.uint8).to(torch.int32)
    return (exponent_bytes << 23).view(torch.float32)


def expand_block_scales(
    scale: torch.Tensor,
    shape: Sequence[int],
    block_size: Sequence[int] = (128, 128),
) -> torch.Tensor:
    if len(shape) != 2 or len(block_size) != 2:
        raise ValueError("only 2D matrices and 2D block sizes are supported")
    block_m, block_k = (int(x) for x in block_size)
    expanded = scale.repeat_interleave(block_m, dim=0).repeat_interleave(block_k, dim=1)
    return expanded[: int(shape[0]), : int(shape[1])]


def dequantize_block_fp8(
    weight: torch.Tensor,
    scale: torch.Tensor,
    block_size: Sequence[int] = (128, 128),
) -> torch.Tensor:
    if weight.dtype != torch.float8_e4m3fn:
        raise TypeError(f"expected float8_e4m3fn weight, got {weight.dtype}")
    scale_float = e8m0_to_float(scale)
    expected = (
        (weight.shape[0] + int(block_size[0]) - 1) // int(block_size[0]),
        (weight.shape[1] + int(block_size[1]) - 1) // int(block_size[1]),
    )
    if tuple(scale.shape) != expected:
        raise ValueError(f"scale shape {tuple(scale.shape)} does not match expected {expected}")
    return weight.float() * expand_block_scales(scale_float, weight.shape, block_size)


def quantize_block_fp8(
    weight: torch.Tensor,
    block_size: Sequence[int] = (128, 128),
    *,
    minimum_scale: torch.Tensor | None = None,
    fixed_scale: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize to E4M3/UE8M0, optionally preserving conservative source scales."""
    if weight.ndim != 2:
        raise ValueError(f"expected a 2D matrix, got shape {tuple(weight.shape)}")
    if not torch.isfinite(weight).all():
        raise ValueError("weight contains non-finite values")
    block_m, block_k = (int(x) for x in block_size)
    rows, cols = weight.shape
    if rows % block_m or cols % block_k:
        raise ValueError(
            f"shape {tuple(weight.shape)} must be divisible by block size {(block_m, block_k)}"
        )

    if minimum_scale is not None and fixed_scale is not None:
        raise ValueError("minimum_scale and fixed_scale are mutually exclusive")
    source = weight.float().contiguous()
    blocks = (
        source.view(rows // block_m, block_m, cols // block_k, block_k)
        .permute(0, 2, 1, 3)
        .contiguous()
    )
    # Match DeepGEMM's checkpoint quantizer exactly. The 1e-4 floor is applied
    # to block amax *before* division; omitting it silently changes low-magnitude
    # blocks even when no edit has been applied.
    expected_scale_shape = (rows // block_m, cols // block_k)
    if fixed_scale is not None:
        if tuple(fixed_scale.shape) != expected_scale_shape:
            raise ValueError(
                f"fixed scale shape {tuple(fixed_scale.shape)} does not match "
                f"expected {expected_scale_shape}"
            )
        scale_float = e8m0_to_float(fixed_scale).to(source.device)
    else:
        amax = blocks.abs().amax(dim=(-1, -2)).clamp_min(AMAX_MIN)
        raw_scale = amax / FP8_MAX
        exponent = torch.ceil(torch.log2(raw_scale)).clamp(-127, 127)
        scale_float = torch.pow(2.0, exponent)
        if minimum_scale is not None:
            if tuple(minimum_scale.shape) != tuple(scale_float.shape):
                raise ValueError(
                    f"minimum scale shape {tuple(minimum_scale.shape)} does not match "
                    f"expected {tuple(scale_float.shape)}"
                )
            minimum_float = e8m0_to_float(minimum_scale).to(source.device)
            scale_float = torch.maximum(scale_float, minimum_float)
    quant_blocks = (blocks / scale_float[..., None, None]).clamp(-FP8_MAX, FP8_MAX)
    quant = quant_blocks.to(torch.float8_e4m3fn).permute(0, 2, 1, 3).contiguous().view(rows, cols)
    scale = fixed_scale.clone() if fixed_scale is not None else scale_float.to(torch.float8_e8m0fnu)
    return quant, scale


def project_output_subspace(
    weight: torch.Tensor,
    directions: torch.Tensor,
    *,
    strength: float,
    preserve_row_norm: bool,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Apply W <- W - strength V^T(VW) to output rows of a linear matrix."""
    if weight.ndim != 2 or directions.ndim != 2:
        raise ValueError("weight and directions must both be matrices")
    if directions.shape[1] != weight.shape[0]:
        raise ValueError(f"direction width {directions.shape[1]} != output rows {weight.shape[0]}")
    if not (0.0 < strength <= 4.0):
        raise ValueError("strength must be in (0, 4]")

    w = weight.float()
    v = directions.float()
    if not torch.isfinite(v).all():
        raise ValueError("directions contain non-finite values")
    singular_values = torch.linalg.svdvals(v)
    if singular_values.numel() == 0 or singular_values[-1] <= 1e-8:
        raise ValueError("directions must have full row rank and non-zero norm")
    # QR on V.T produces an orthonormal basis even when supplied vectors drift.
    basis = torch.linalg.qr(v.T, mode="reduced").Q.T
    before_component = basis @ w
    edited = w - strength * (basis.T @ before_component)
    if preserve_row_norm:
        before_norm = torch.linalg.vector_norm(w, dim=1)
        after_norm = torch.linalg.vector_norm(edited, dim=1).clamp_min(1e-12)
        edited = edited * (before_norm / after_norm).unsqueeze(1)
    after_component = basis @ edited
    delta_norm = torch.linalg.vector_norm(edited - w)
    weight_norm = torch.linalg.vector_norm(w).clamp_min(1e-12)
    before_proj_norm = torch.linalg.vector_norm(before_component).clamp_min(1e-12)
    projection_energy = torch.sum(before_component * before_component).clamp_min(1e-12)
    return edited, {
        "relative_edit_norm": float((delta_norm / weight_norm).item()),
        "projection_residual": float(
            (torch.linalg.vector_norm(after_component) / before_proj_norm).item()
        ),
        "projection_signed_gain": float(
            (torch.sum(after_component * before_component) / projection_energy).item()
        ),
        "row_norm_preserved": float(preserve_row_norm),
    }
