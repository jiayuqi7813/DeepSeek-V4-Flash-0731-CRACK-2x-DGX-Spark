from __future__ import annotations

import torch
from safetensors.torch import save_file

from dspark_crack.validate_candidate import (
    compare_rewritten_shard,
    manifest_consistency_failures,
)


def test_compare_rewritten_shard_accepts_only_declared_changes(tmp_path) -> None:
    source = tmp_path / "source.safetensors"
    candidate = tmp_path / "candidate.safetensors"
    common = torch.arange(8, dtype=torch.float32)
    save_file(
        {"common": common, "target": torch.zeros(8)},
        str(source),
        metadata={"format": "test"},
    )
    save_file(
        {"common": common, "target": torch.ones(8)},
        str(candidate),
        metadata={"format": "test"},
    )
    result = compare_rewritten_shard(source, candidate, {"target"})
    assert result["ok"]
    assert result["checked_unchanged"] == 1
    assert result["checked_edited"] == 1


def test_compare_rewritten_shard_rejects_undeclared_change(tmp_path) -> None:
    source = tmp_path / "source.safetensors"
    candidate = tmp_path / "candidate.safetensors"
    save_file({"common": torch.zeros(8), "target": torch.zeros(8)}, str(source))
    save_file({"common": torch.ones(8), "target": torch.ones(8)}, str(candidate))
    result = compare_rewritten_shard(source, candidate, {"target"})
    assert not result["ok"]
    assert result["unchanged_failures"] == ["common"]


def test_compare_rewritten_shard_requires_fixed_scale_to_stay_equal(tmp_path) -> None:
    source = tmp_path / "source.safetensors"
    candidate = tmp_path / "candidate.safetensors"
    save_file({"scale": torch.zeros(2), "weight": torch.zeros(8)}, str(source))
    save_file({"scale": torch.ones(2), "weight": torch.ones(8)}, str(candidate))

    result = compare_rewritten_shard(source, candidate, {"weight"}, {"weight"})

    assert not result["ok"]
    assert result["unchanged_failures"] == ["scale"]


def test_manifest_consistency_rejects_stale_direction_hash() -> None:
    lock = {
        "model_id": "example/model",
        "revision": "abc123",
        "mtp_targets": ["mtp.0.attn.wo_b"],
    }
    report = {
        "direction_sha256": "new",
        "direction_mode": "layer.sra",
        "layers": [10],
        "strength": 2.0,
        "preserve_row_norm": False,
        "mtp": "stock",
        "scale_policy": "fixed",
        "targets": [{"base": "layers.10.attn.wo_b"}],
    }
    manifest = {
        "format": "dspark-crack-checkpoint-v1",
        "source_model": "example/model",
        "source_revision": "abc123",
        "direction_sha256": "old",
        "direction_mode": "layer.sra",
        "layers": [10],
        "strength": 2.0,
        "preserve_row_norm": False,
        "mtp": "stock",
        "scale_policy": "fixed",
        "report": "CRACK_EDIT_REPORT.json",
    }

    failures = manifest_consistency_failures(lock, report, manifest)

    assert failures == ["edit manifest direction_sha256 does not match report/source lock"]
