# Third-party notices

## DeepSeek V4 Flash 0731

- Repository: `deepseek-ai/DeepSeek-V4-Flash-0731`
- Pinned revision: `9e165c30e2704aec5d9d593cce3eebd58bbef1cb`
- License: MIT as declared by the upstream model repository

Edited model weights remain derivative weights and must retain the upstream
license and attribution.

## vLLM

The capture adapter patches Python methods at runtime and does not redistribute
vLLM source. vLLM is licensed under Apache-2.0.

## Anemll DSpark runtime

The deployment profile expects
`ghcr.io/anemll/dspark-vllm-gx10:0.1.1`; it does not redistribute that image.

Production orchestration files and selected runtime hotfixes under
`runtime/miaai-dspark/` are derived from
`MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark`, pinned at commit `018c6bc`.
That repository is MIT licensed, copyright (c) 2026 Tony Deangelo. The
upstream license is reproduced as `runtime/miaai-dspark/LICENSE.upstream`.

## Semantic Harmful–Harmless Prompt Pairs

- Repository: `heretic-org/Semantic-Harmless`
- Pinned revision: `7e9f2b01272da85f2be7a3437f31ac46698e8735`
- License: CC BY 4.0
- Upstream sources named by that dataset: `mlabonne/harmful_behaviors` and
  `mlabonne/harmless_alpaca`

The build script downloads the paired metadata at the pinned revision and
emits a local calibration/holdout manifest. The dataset itself is not vendored
in this repository.

## Evaluation datasets

- OpenAI GSM8K, revision
  `3101c7d5072418e28b9008a6636bde82a006892c`, MIT License. The pinned test
  JSONL SHA-256 is
  `3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14`.
- OpenAI HumanEval, revision
  `6d43fb980f9fee3c892a914eda09951f772ad10d`, MIT License. The pinned dataset
  SHA-256 is
  `b796127e635a67f93fb35c04f4cb03cf06f38c8072ee7cee8833d7bee06979ef`.
- TIGER-Lab MMLU-Pro, revision
  `b189ec765aa7ed75c8acfea42df31fdae71f97be`, MIT License. The pinned test
  parquet SHA-256 is
  `0e24a191921c2f453518a537a8b2117bd137e7714d4ef1565e9ba06c1ecb9ad8`.

The repository contains a hash-verifying fetcher, not copies of these datasets.

## Related implementations reviewed

The design was compared against the public model cards and/or MIT-licensed
repositories for `dealignai/DeepSeek-V4-Flash-0731-JANG-CRACK`,
`lovesenko/DeepSeek-V4-Flash-DSpark-Abliterated`, and the MiaAI/drowzeys
DeepSeek V4 ablation tools. This code is a clean implementation for the exact
0731 FP8 checkpoint and the local two-DGX-Spark runtime.
