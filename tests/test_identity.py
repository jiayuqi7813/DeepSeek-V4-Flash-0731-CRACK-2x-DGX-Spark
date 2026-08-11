from __future__ import annotations

import json
import struct
from pathlib import Path

from dspark_crack.identity import read_safetensors_header


def test_header_reader_does_not_require_tensor_payload(tmp_path: Path) -> None:
    header = {
        "x": {"dtype": "F8_E4M3", "shape": [2, 2], "data_offsets": [0, 4]},
        "__metadata__": {"format": "pt"},
    }
    encoded = json.dumps(header).encode()
    path = tmp_path / "tiny.safetensors"
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + b"\x00" * 4)
    assert read_safetensors_header(path) == header
