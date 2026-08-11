from __future__ import annotations

import json

import torch
from safetensors.torch import load_file, save_file

from dspark_crack.directions import _sra_clean, binary_auc, compute_directions


def test_binary_auc_perfect_and_reversed() -> None:
    positive = torch.tensor([3.0, 4.0, 5.0])
    negative = torch.tensor([0.0, 1.0, 2.0])
    assert binary_auc(positive, negative) == 1.0
    assert binary_auc(negative, positive) == 0.0


def test_binary_auc_ties() -> None:
    assert binary_auc(torch.tensor([1.0]), torch.tensor([1.0])) == 0.5


def test_sra_seed_is_reproducible_independent_of_global_rng() -> None:
    generator = torch.Generator().manual_seed(123)
    raw = torch.randn(32, generator=generator)
    benign = torch.randn((16, 32), generator=generator)
    torch.manual_seed(1)
    first, first_atoms, _, _ = _sra_clean(raw, benign, rank=4, seed=17)
    torch.manual_seed(999)
    second, second_atoms, _, _ = _sra_clean(raw, benign, rank=4, seed=17)
    assert torch.equal(first, second)
    assert torch.equal(first_atoms, second_atoms)


def test_zero_difference_layer_is_inactive(tmp_path) -> None:
    capture_dir = tmp_path / "captures"
    capture_dir.mkdir()
    dataset = tmp_path / "dataset.jsonl"
    rows = []
    for split in ("train", "holdout"):
        for label in (0, 1):
            for repeat in range(2):
                sample_id = f"{split}-{label}-{repeat}"
                rows.append({"id": sample_id, "label": label, "split": split})
                tensor = torch.zeros((2, 4096), dtype=torch.float16)
                tensor[1, 0] = 1.0 if label else -1.0
                save_file({"attn_in": tensor}, str(capture_dir / f"{sample_id}.safetensors"))
    dataset.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    output = tmp_path / "directions.safetensors"

    report = compute_directions(
        capture_dir,
        dataset,
        output,
        surface="attn_in",
        sra_rank=0,
        min_auc=0.65,
    )

    directions = load_file(str(output))
    assert report["inactive_raw_layers"] == [0]
    assert report["inactive_sra_layers"] == [0]
    assert report["global_sra_layers"] == [1]
    assert report["global_selection_metric"] == "train_auc"
    assert report["layers"][0]["sra"]["holdout_auc"] == 0.5
    assert torch.count_nonzero(directions["layer.00.sra"]) == 0
