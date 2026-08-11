from __future__ import annotations

import pytest
import torch

from dspark_crack.quantization import (
    dequantize_block_fp8,
    e8m0_to_float,
    project_output_subspace,
    quantize_block_fp8,
)


def test_e8m0_known_powers() -> None:
    values = torch.tensor([0.5, 1.0, 2.0, 4.0], dtype=torch.float32)
    encoded = values.to(torch.float8_e8m0fnu)
    assert encoded.view(torch.uint8).tolist() == [126, 127, 128, 129]
    torch.testing.assert_close(e8m0_to_float(encoded), values)


def test_block_quantization_roundtrip_and_scale_shape() -> None:
    generator = torch.Generator().manual_seed(7)
    source = torch.randn((256, 384), generator=generator) * 0.07
    quantized, scale = quantize_block_fp8(source)
    restored = dequantize_block_fp8(quantized, scale)
    assert quantized.dtype == torch.float8_e4m3fn
    assert scale.dtype == torch.float8_e8m0fnu
    assert scale.shape == (2, 3)
    relative_error = torch.linalg.vector_norm(restored - source) / torch.linalg.vector_norm(source)
    assert relative_error < 0.04


def test_block_requantization_is_bit_exact() -> None:
    generator = torch.Generator().manual_seed(19)
    source = torch.randn((256, 256), generator=generator) * 0.01
    source[:128, :128] = 0.0
    quantized, scale = quantize_block_fp8(source)
    restored = dequantize_block_fp8(quantized, scale)
    requantized, rescale = quantize_block_fp8(restored)
    assert torch.equal(requantized, quantized)
    assert torch.equal(rescale, scale)


def test_minimum_scale_preserves_conservative_source_scale() -> None:
    source_scale = torch.full((1, 1), 2.0**-10, dtype=torch.float32).to(torch.float8_e8m0fnu)
    quantized = torch.full((128, 128), 224.0, dtype=torch.float32).to(torch.float8_e4m3fn)
    restored = dequantize_block_fp8(quantized, source_scale)
    requantized, rescale = quantize_block_fp8(restored, minimum_scale=source_scale)
    assert torch.equal(requantized, quantized)
    assert torch.equal(rescale, source_scale)


def test_fixed_scale_clamps_without_changing_scale() -> None:
    source_scale = torch.ones((1, 1), dtype=torch.float32).to(torch.float8_e8m0fnu)
    source = torch.zeros((128, 128), dtype=torch.float32)
    source[0, 0] = 500.0
    quantized, rescale = quantize_block_fp8(source, fixed_scale=source_scale)
    assert quantized[0, 0].float() == 448.0
    assert torch.equal(rescale, source_scale)


def test_output_projection_eliminates_rank_two_subspace_at_strength_one() -> None:
    generator = torch.Generator().manual_seed(11)
    weight = torch.randn((32, 48), generator=generator)
    directions = torch.randn((2, 32), generator=generator)
    edited, metrics = project_output_subspace(
        weight,
        directions,
        strength=1.0,
        preserve_row_norm=False,
    )
    basis = torch.linalg.qr(directions.T, mode="reduced").Q.T
    assert torch.linalg.vector_norm(basis @ edited) < 1e-5 * torch.linalg.vector_norm(weight)
    assert metrics["projection_residual"] < 1e-5
    assert abs(metrics["projection_signed_gain"]) < 1e-5


def test_strength_two_is_a_householder_reflection() -> None:
    generator = torch.Generator().manual_seed(29)
    weight = torch.randn((32, 48), generator=generator)
    directions = torch.randn((2, 32), generator=generator)
    edited, metrics = project_output_subspace(
        weight,
        directions,
        strength=2.0,
        preserve_row_norm=False,
    )
    basis = torch.linalg.qr(directions.T, mode="reduced").Q.T
    torch.testing.assert_close(basis @ edited, -(basis @ weight), rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(edited.T @ edited, weight.T @ weight, rtol=1e-5, atol=2e-5)
    assert abs(metrics["projection_residual"] - 1.0) < 1e-5
    assert abs(metrics["projection_signed_gain"] + 1.0) < 1e-5


@pytest.mark.parametrize(
    "directions",
    [torch.zeros((1, 4)), torch.tensor([[float("nan"), 0.0, 0.0, 0.0]])],
)
def test_output_projection_rejects_invalid_directions(directions: torch.Tensor) -> None:
    with pytest.raises(ValueError):
        project_output_subspace(
            torch.eye(4),
            directions,
            strength=2.0,
            preserve_row_norm=False,
        )
