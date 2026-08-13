#!/usr/bin/env python3
"""Hotfix: do not let SWA groups shrink the hybrid prefix-cache common hit.

On DSV4-Flash + DSpark the v1 HybridKVCacheCoordinator has four KV groups
(1x MLAAttentionSpec + 3x SlidingWindowMLASpec). ``find_longest_cache_hit``
takes the min hit length across groups. Sliding-window managers free blocks
outside the attention window by design, so at 32K+ their hit collapses to 0
and zeroes the common hit even when the full-attention MLA group still has
the prefix. Warm x8 32K/62K then re-prefill with 0 cache hits (issue #26).

Fix: after a SlidingWindowSpec (covers SlidingWindowMLASpec) lookup, keep
that group's hit blocks but do not assign its window-limited length to
``curr_hit_length``. Full-attention / MLA still bound the common hit.

Idempotent: re-applying is a no-op once the marker is present.

Patches /usr/local/lib/python3.12/dist-packages/vllm/v1/core/kv_cache_coordinator.py
in-place inside the container (called from the compose entrypoint before
``exec vllm serve``).
"""
from pathlib import Path

P = Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/v1/core/kv_cache_coordinator.py"
)
src = P.read_text()
MARK = "# [issue26-hotfix] SWA groups must not shrink the hybrid common hit"
if MARK in src:
    print(f"[issue26-hotfix] already applied to {P}")
    raise SystemExit(0)

ANCHOR = (
    "                _new_hit_length = len(hit_blocks[0]) * spec.block_size\n"
    "                if drop_eagle_block:\n"
    "                    eagle_verified.add(idx)\n"
    "                elif _new_hit_length < curr_hit_length:\n"
    "                    # length shrunk; invalidate previous eagle verifications\n"
    "                    eagle_verified.clear()\n"
    "                curr_hit_length = _new_hit_length\n"
)
assert ANCHOR in src, "hybrid min-hit assign anchor not found; refusing to patch"

INJECT = (
    "                _new_hit_length = len(hit_blocks[0]) * spec.block_size\n"
    "                # [issue26-hotfix] SWA groups must not shrink the hybrid common hit.\n"
    "                # Sliding-window managers retain only the last window tokens;\n"
    "                # using their hit as the next candidate (min-across-groups)\n"
    "                # zeroes warm prefix-cache hits at 32K+ x8 (issue #26).\n"
    "                if isinstance(spec, SlidingWindowSpec):\n"
    "                    if drop_eagle_block:\n"
    "                        eagle_verified.add(idx)\n"
    "                    for group_id, blocks in zip(group_ids, hit_blocks):\n"
    "                        hit_blocks_by_group[group_id] = blocks\n"
    "                    continue\n"
    "                if drop_eagle_block:\n"
    "                    eagle_verified.add(idx)\n"
    "                elif _new_hit_length < curr_hit_length:\n"
    "                    # length shrunk; invalidate previous eagle verifications\n"
    "                    eagle_verified.clear()\n"
    "                curr_hit_length = _new_hit_length\n"
)
src = src.replace(ANCHOR, INJECT, 1)
P.write_text(src)
print(f"[issue26-hotfix] patched {P}")
