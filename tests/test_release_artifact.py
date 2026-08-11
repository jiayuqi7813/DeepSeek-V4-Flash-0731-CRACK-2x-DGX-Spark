from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[1]
DIRECTION = ROOT / "artifacts/directions/attn-out-sra-r8.safetensors"
REPORT = ROOT / "artifacts/directions/attn-out-sra-r8.report.json"
EDITOR_WRAPPER = ROOT / "scripts/edit_checkpoint_in_runtime.sh"
START_WRAPPER = ROOT / "scripts/start_production_cluster.sh"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_tracked_release_direction_is_complete_and_locked() -> None:
    assert _sha256(DIRECTION) == (
        "fe8c263a8d32deb71e3f6e866b90f8246f452f6e2103b0e0400a77480fd2602a"
    )
    assert _sha256(REPORT) == (
        "c5391fc811c33556143bd6f4807e407d10ddb190618230ce0d05f35fdbff9a24"
    )
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    assert report["format"] == "dspark-crack-direction-report-v2"
    assert report["surface"] == "attn_out"
    assert report["sample_count"] == 832
    assert report["train_count"] == 662
    assert report["holdout_count"] == 170
    assert report["global_selection_metric"] == "train_auc"

    with safe_open(str(DIRECTION), framework="pt", device="cpu") as handle:
        metadata = handle.metadata()
        assert metadata is not None
        assert metadata["format"] == "dspark-crack-directions-v2"
        assert metadata["surface"] == "attn_out"
        assert metadata["sample_count"] == "832"
        assert metadata["global_selection_metric"] == "train_auc"
        for layer in range(10, 43):
            direction = handle.get_tensor(f"layer.{layer:02d}.sra")
            assert direction.shape == (1, 4096)
            assert direction.dtype == torch.float32
            assert torch.isfinite(direction).all()
            assert torch.isclose(torch.linalg.vector_norm(direction), torch.tensor(1.0), atol=1e-5)


def test_runtime_editor_does_not_create_root_owned_release_files() -> None:
    wrapper = EDITOR_WRAPPER.read_text(encoding="utf-8")
    assert wrapper.count('--user "$(id -u):$(id -g)"') == 2


def test_production_start_cleans_up_both_nodes_on_failure_or_interrupt() -> None:
    wrapper = START_WRAPPER.read_text(encoding="utf-8")
    assert "cleanup_failed_start" in wrapper
    assert "stop_production_cluster.sh" in wrapper
    assert "trap 'cleanup_failed_start $?' ERR" in wrapper
    assert "trap 'cleanup_failed_start 130' INT" in wrapper
    assert "trap 'cleanup_failed_start 143' TERM" in wrapper
