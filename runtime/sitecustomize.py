"""Opt-in activation capture patch loaded automatically by Python.

This module is inert unless DSPARK_CRACK_CAPTURE=1. The capture deployment
places this directory on PYTHONPATH in every vLLM process.
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
import traceback
from pathlib import Path
from typing import Any

if os.environ.get("DSPARK_CRACK_CAPTURE") == "1":
    import torch
    import torch.distributed as dist
    from safetensors.torch import save_file
    from vllm.models.deepseek_v4.attention import DeepseekV4Attention
    from vllm.models.deepseek_v4.nvidia.model import DeepseekV4DecoderLayer

    _CONTROL = Path(os.environ.get("DSPARK_CRACK_CONTROL", "/capture/control.json"))
    _OUTPUT = Path(os.environ.get("DSPARK_CRACK_OUTPUT", "/capture/samples"))
    _NUM_LAYERS = int(os.environ.get("DSPARK_CRACK_NUM_LAYERS", "43"))
    _STRICT = os.environ.get("DSPARK_CRACK_STRICT", "1") == "1"
    _LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)")

    def _rank_zero() -> bool:
        return dist.is_available() and dist.is_initialized() and dist.get_rank() == 0

    def _layer_from_prefix(prefix: str) -> int | None:
        match = _LAYER_RE.search(prefix)
        return int(match.group(1)) if match else None

    def _last_vector(tensor: torch.Tensor) -> torch.Tensor:
        value = tensor.detach()
        if value.ndim != 2 or value.shape[-1] != 4096:
            raise RuntimeError(f"unexpected capture shape {tuple(value.shape)}")
        return value[-1].to(device="cpu", dtype=torch.float16).contiguous()

    def _is_prefill(positions: torch.Tensor) -> bool:
        return positions.ndim == 1 and positions.numel() > 1

    class _State:
        def __init__(self) -> None:
            self.lock = threading.RLock()
            self.sample_id: str | None = None
            self.nonce: str | None = None
            self.position_count = 0
            self.started_at = 0.0
            self.attn_in: dict[int, torch.Tensor] = {}
            self.attn_out: dict[int, torch.Tensor] = {}
            self.ffn_out: dict[int, torch.Tensor] = {}

        def _reset(self) -> None:
            self.sample_id = None
            self.nonce = None
            self.position_count = 0
            self.started_at = 0.0
            self.attn_in.clear()
            self.attn_out.clear()
            self.ffn_out.clear()

        def begin(self, positions: torch.Tensor) -> bool:
            with self.lock:
                try:
                    control = json.loads(_CONTROL.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    return False
                if control.get("armed") is not True:
                    return False
                sample_id = str(control.get("sample_id", ""))
                nonce = str(control.get("nonce", ""))
                if not sample_id or not nonce or "/" in sample_id or ".." in sample_id:
                    raise RuntimeError(f"invalid capture control: {control!r}")
                target = _OUTPUT / f"{sample_id}.safetensors"
                if target.exists():
                    return False
                self._reset()
                self.sample_id = sample_id
                self.nonce = nonce
                self.position_count = int(positions.numel())
                self.started_at = time.time()
                return True

        def record_attention(
            self,
            layer: int,
            input_vector: torch.Tensor,
            output: torch.Tensor,
            positions: torch.Tensor,
        ) -> None:
            with self.lock:
                if layer == 0:
                    self.begin(positions)
                if self.sample_id is None:
                    return
                if input_vector.ndim != 1 or input_vector.shape[0] != 4096:
                    raise RuntimeError(
                        f"unexpected pre-attention snapshot shape {tuple(input_vector.shape)}"
                    )
                self.attn_in[layer] = input_vector
                self.attn_out[layer] = _last_vector(output)

        def record_ffn(self, layer: int, output: torch.Tensor) -> None:
            with self.lock:
                if self.sample_id is None:
                    return
                self.ffn_out[layer] = _last_vector(output)
                if layer == _NUM_LAYERS - 1:
                    self.finalize()

        def finalize(self) -> None:
            with self.lock:
                sample_id = self.sample_id
                nonce = self.nonce
                if sample_id is None or nonce is None:
                    return
                expected = set(range(_NUM_LAYERS))
                inventories = {
                    "attn_in": set(self.attn_in),
                    "attn_out": set(self.attn_out),
                    "ffn_out": set(self.ffn_out),
                }
                incomplete = {
                    name: sorted(expected - layers)
                    for name, layers in inventories.items()
                    if layers != expected
                }
                if incomplete:
                    raise RuntimeError(f"incomplete capture for {sample_id}: {incomplete}")
                tensors = {
                    "attn_in": torch.stack([self.attn_in[i] for i in range(_NUM_LAYERS)]),
                    "attn_out": torch.stack([self.attn_out[i] for i in range(_NUM_LAYERS)]),
                    "ffn_out": torch.stack([self.ffn_out[i] for i in range(_NUM_LAYERS)]),
                }
                _OUTPUT.mkdir(parents=True, exist_ok=True)
                target = _OUTPUT / f"{sample_id}.safetensors"
                temporary = _OUTPUT / f".{sample_id}.{nonce}.tmp"
                metadata = {
                    "format": "dspark-crack-capture-v1",
                    "sample_id": sample_id,
                    "nonce": nonce,
                    "num_layers": str(_NUM_LAYERS),
                    "hidden_size": "4096",
                    "position_count": str(self.position_count),
                    "tp_rank": "0",
                    "elapsed_seconds": f"{time.time() - self.started_at:.6f}",
                }
                save_file(tensors, str(temporary), metadata=metadata)
                os.replace(temporary, target)
                # The vLLM container runs as root while the host-side capture
                # driver runs as UID 1000. Keep payloads read-only to ordinary
                # users but readable for validation and rsync.
                os.chmod(target, 0o644)
                ack = {
                    "sample_id": sample_id,
                    "nonce": nonce,
                    "path": str(target),
                    "shapes": {name: list(tensor.shape) for name, tensor in tensors.items()},
                    "completed_at": time.time(),
                }
                ack_tmp = _OUTPUT / f".{sample_id}.{nonce}.ack.tmp"
                ack_tmp.write_text(json.dumps(ack), encoding="utf-8")
                os.replace(ack_tmp, _OUTPUT / f"{sample_id}.ack.json")
                os.chmod(_OUTPUT / f"{sample_id}.ack.json", 0o644)
                self._reset()

    _STATE = _State()
    _ORIGINAL_ATTN_FORWARD = DeepseekV4Attention.forward
    _ORIGINAL_DECODER_FORWARD = DeepseekV4DecoderLayer.forward

    def _capture_failure(where: str, exc: BaseException) -> None:
        print(
            f"[dspark-crack] capture failure in {where}: {exc}\n{traceback.format_exc()}",
            flush=True,
        )
        if _STRICT:
            raise exc

    def _attention_forward(
        self: Any,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        llama_4_scaling: torch.Tensor | None = None,
    ) -> torch.Tensor:
        capture_layer: int | None = None
        input_snapshot: torch.Tensor | None = None
        if _rank_zero() and _is_prefill(positions):
            capture_layer = _layer_from_prefix(self.prefix)
            if capture_layer is not None and capture_layer < _NUM_LAYERS:
                # Some fused attention/mHC kernels reuse the input buffer.
                # Snapshot before the original forward, never after it.
                input_snapshot = _last_vector(hidden_states)
        output = _ORIGINAL_ATTN_FORWARD(self, positions, hidden_states, llama_4_scaling)
        if capture_layer is not None and input_snapshot is not None:
            try:
                _STATE.record_attention(capture_layer, input_snapshot, output, positions)
            except Exception as exc:  # noqa: BLE001 - failure policy is configurable
                _capture_failure("attention", exc)
        return output

    def _decoder_forward(self: Any, *args: Any, **kwargs: Any) -> Any:
        result = _ORIGINAL_DECODER_FORWARD(self, *args, **kwargs)
        if _rank_zero():
            positions = kwargs.get("positions")
            if positions is None and len(args) >= 2:
                positions = args[1]
            if isinstance(positions, torch.Tensor) and _is_prefill(positions):
                layer = _layer_from_prefix(self.attn.prefix)
                if layer is not None and layer < _NUM_LAYERS:
                    try:
                        _STATE.record_ffn(layer, result[0])
                    except Exception as exc:  # noqa: BLE001 - failure policy is configurable
                        _capture_failure("decoder", exc)
        return result

    if not getattr(DeepseekV4Attention.forward, "_dspark_crack_capture", False):
        _attention_forward._dspark_crack_capture = True  # type: ignore[attr-defined]
        _decoder_forward._dspark_crack_capture = True  # type: ignore[attr-defined]
        DeepseekV4Attention.forward = _attention_forward
        DeepseekV4DecoderLayer.forward = _decoder_forward
        print(
            f"[dspark-crack] capture patch installed: output={_OUTPUT} layers={_NUM_LAYERS}",
            flush=True,
        )
