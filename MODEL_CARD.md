---
license: mit
base_model: deepseek-ai/DeepSeek-V4-Flash-0731
pipeline_tag: text-generation
tags:
  - deepseek-v4
  - dgx-spark
  - vllm
  - fp8
  - abliterated
  - uncensored
---

# DeepSeek V4 Flash 0731 CRACK — DGX Spark

This is a refusal-subspace-edited derivative of the **exact**
`deepseek-ai/DeepSeek-V4-Flash-0731` checkpoint at revision
`9e165c30e2704aec5d9d593cce3eebd58bbef1cb`. It retains the upstream 48-shard
FP8 E4M3 + UE8M0 layout, all 43 main layers, and the three stock DSpark MTP
projections. It is intended for CUDA/vLLM tensor-parallel inference on two
NVIDIA DGX Sparks.

This is not the Apple MLX/JANG checkpoint and is not compatible with the JANG
runtime. The reproducible editor, direction artifact, deployment profiles,
tests, and detailed research record are maintained in
[`jiayuqi7813/DeepSeek-V4-Flash-0731-CRACK-2x-DGX-Spark`](https://github.com/jiayuqi7813/DeepSeek-V4-Flash-0731-CRACK-2x-DGX-Spark).

## What changed

- Target tensors: main-layer attention `wo_b`, layers 10–42 inclusive.
- Direction: one SRA-cleaned, rank-1 attention-output direction per layer.
- Transform: strength-2 Householder reflection, `W' = W - 2 Vᵀ(VW)`.
- Quantization: native 128×128 FP8 blocks with the original UE8M0 scales held
  fixed; only the selected FP8 weight tensors change.
- Row-norm rematching: disabled.
- MTP: stock upstream projections, unchanged.
- Direction SHA-256:
  `fe8c263a8d32deb71e3f6e866b90f8246f452f6e2103b0e0400a77480fd2602a`.

The exact-arithmetic transform reverses the selected output component while
preserving each matrix's column Gram matrix before requantization. No gradient
training, fine-tuning, LoRA, or synthetic-response training was used.

## Validation summary

The candidate was built independently on both DGX nodes. Each copy passed
tensor-granularity validation across all 72,317 indexed tensors: all
non-target tensors in rewritten shards were bit-exact, fixed scale tensors
were bit-exact, and every rewritten-shard SHA-256 matched across nodes.

Locked relative capability comparisons used the same eager/no-MTP TP=2
profile for the source and edited checkpoints:

| Evaluation | Source | Edited |
| --- | ---: | ---: |
| Synthetic regression suite | 13/19 | 13/19 |
| GSM8K deterministic 100-question screen | 97/100 | 96/100 |
| HumanEval deterministic 32-task screen | 27/32 | 27/32 |
| HumanEval deterministic full 164-task confirmation | 150/164 | 148/164 |
| MMLU-Pro balanced 560-question confirmation | 379/560 | 382/560 |

The full HumanEval delta was -1.22 percentage points. Paired transitions were
147 pass/pass, 3 source-pass/edited-fail, 1 source-fail/edited-pass, and 13
fail/fail (exact McNemar `p=0.625`). All 328 generations ended naturally; no
length-limited output was scored.

On the 85-prompt harmful holdout, an untouched-source three-way judge measured
material compliance (`PARTIAL` or `COMPLY`) at 3/85 (3.53%) for source and
77/85 (90.59%) for edited. Strict `COMPLY` was 1/85 (1.18%) versus 44/85
(51.76%). The separate 10-prompt English/Chinese harmful probe measured
material compliance at 0/10 versus 10/10, with matched harmless probes showing
no new lexical refusals.

A 12-prompt English/Chinese open-ended writing comparison used anonymous,
position-balanced A/B judging by the untouched source checkpoint. All answers
ended naturally under the final 768-token cap. The edited model won 4, source
won 2, and 6 tied; mean four-dimension totals were 19.25/20 edited versus
18.58/20 source. Manual review conservatively corrected two dialogue verdicts
with flawed rationales to ties, leaving 2 edited wins, 2 source wins, and 8
ties. Deterministic hard-constraint compliance was 8/12 for both models; both
missed the requested length on the same four explanation/argument tasks.

These are preservation comparisons, not claims of direct comparability with
the original benchmark papers. See the code repository for dataset revisions,
selection hashes, prompts, isolated HumanEval execution, paired transition
counts, behavior results, and production performance measurements.

## Runtime

The validated runtime is:

- `ghcr.io/anemll/dspark-vllm-gx10:0.1.1`
- image ID
  `sha256:3430d6614a8e2925f34d059af6caf05aff42387326db4d05639a60f10f2654d8`
- vLLM TP=2 / NCCL across two DGX Sparks
- DSpark speculative decoding with five MTP tokens in production
- declared maximum context length: 1,048,576 tokens

Use the open-source repository's production profile instead of a generic vLLM
command: it carries the GB10, tokenizer, NVFP4 KV-cache, MTP, CUDA-graph, and
dual-CX-7 settings verified for this checkpoint.

## Limitations and intended interpretation

This edit intentionally changes refusal behavior and should not be represented
as the alignment or safety behavior of the upstream model. A non-refusal is
not proof of factual correctness, completeness, legality, or operational
fitness. The main behavioral holdout is English; the separate English/Chinese
probe is small and descriptive. Rare FP8 clipping, long-context behavior
beyond the synthetic retrieval screen, languages outside the measured set,
multi-user serving, and downstream fine-tunes can behave differently. The
writing probe is deliberately small, and the full HumanEval point estimate is
slightly lower even though it remains inside the locked preservation envelope
and is not statistically significant.

Operators and downstream distributors are responsible for evaluating the
model in their own application context and accurately disclosing that it is a
modified derivative.

## License and attribution

The upstream checkpoint and this derivative are distributed under the MIT
license included with the files. Preserve the DeepSeek copyright notice, this
model card, `CRACK_EDIT_MANIFEST.json`, `CRACK_EDIT_REPORT.json`, and
`CRACK_VALIDATION.json` when redistributing the edited weights. Dataset and
tooling notices are recorded in the code repository's
`THIRD_PARTY_NOTICES.md`.

`SOURCE_HF_MANIFEST.json` records the pinned upstream download and original LFS
hashes for provenance. It is not a hash manifest for the edited shard payloads;
the derivative shard hashes are recorded in `CRACK_EDIT_REPORT.json`.
