#!/usr/bin/env bash
# Backport of vLLM PR #49897 for DeepSeek V4 on SM12x/GB10.
# Long-prefill MQA logits can contain NaNs; the CUDA histogram top-k path may
# then emit invalid indices and trigger Xid 31. Route SM12x prefill selection
# through torch.topk, discard non-finite scores, and emit -1 padding.
set -euo pipefail

VLLM_ROOT="${VLLM_ROOT:-/usr/local/lib/python3.12/dist-packages/vllm}"
TARGET="$VLLM_ROOT/model_executor/layers/sparse_attn_indexer.py"

if [ ! -f "$TARGET" ]; then
  echo "Missing vLLM sparse indexer: $TARGET" >&2
  exit 1
fi

python3 - "$TARGET" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
text = path.read_text()
marker = "[PORT #49897]"

if marker in text:
    print(f"[OK] PR #49897 already applied: {path}")
    raise SystemExit(0)

anchor = "@triton.jit\ndef _fused_indexer_q_rope_quant_kernel("
helper = '''def _top_k_per_row_prefill_torch(
    logits: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
    topk_indices: torch.Tensor,
    topk_tokens: int,
) -> None:
    """SM12x fallback for vLLM PR #49897."""
    num_cols = logits.shape[1]
    ks = cu_seqlen_ks.to(torch.long)[:, None]
    cols = torch.arange(num_cols, device=logits.device)[None, :]
    valid = (cols >= ks) & (cols < cu_seqlen_ke.to(torch.long)[:, None])
    logits.masked_fill_(~valid, float("-inf"))
    k = min(topk_tokens, num_cols)
    top_values, top_cols = logits.topk(k, dim=-1)
    relative = (top_cols - ks).to(torch.int32)
    pad_sentinel = torch.iinfo(torch.int32).max
    relative = torch.where(
        top_values.isfinite(), relative, relative.new_full((), pad_sentinel)
    )
    relative, _ = relative.sort(dim=-1)
    relative = torch.where(
        relative == pad_sentinel, relative.new_full((), -1), relative
    )
    topk_indices[:, :k] = relative
    if k < topk_tokens:
        topk_indices[:, k:] = -1


'''

if anchor not in text:
    raise SystemExit(f"Helper insertion anchor not found in {path}")
text = text.replace(anchor, helper + anchor, 1)

old = '''                ops.top_k_per_row_prefill(
                    logits,
                    cu_seqlen_ks,
                    cu_seqlen_ke,
                    topk_indices,
                    num_rows,
                    logits.stride(0),
                    logits.stride(1),
                    topk_tokens,
                )'''
new = '''                if current_platform.is_cuda() and (
                    current_platform.is_device_capability_family(120)
                ):
                    # [PORT #49897] Avoid invalid indices from the SM12x
                    # histogram top-k path when MQA logits contain NaNs.
                    _top_k_per_row_prefill_torch(
                        logits,
                        cu_seqlen_ks,
                        cu_seqlen_ke,
                        topk_indices,
                        topk_tokens,
                    )
                else:
                    ops.top_k_per_row_prefill(
                        logits,
                        cu_seqlen_ks,
                        cu_seqlen_ke,
                        topk_indices,
                        num_rows,
                        logits.stride(0),
                        logits.stride(1),
                        topk_tokens,
                    )'''

count = text.count(old)
if count != 1:
    raise SystemExit(f"Expected one prefill top-k block in {path}, found {count}")
text = text.replace(old, new, 1)
compile(text, str(path), "exec")
path.write_text(text)
print(f"[OK] Applied vLLM PR #49897 SM12x prefill top-k fallback: {path}")
PY

