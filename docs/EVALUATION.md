# Evaluation protocol

All model loading and inference runs on the two DGX Sparks. The control Mac may
build small JSONL selections, verify hashes, orchestrate SSH commands, and
analyze exported text results; it must not load the checkpoint or run model
inference.

Every base/candidate comparison uses the same TP=2 runtime image, eager/no-MTP
evaluation profile, API sampling parameters, prompt text, dataset hash, and
selected record IDs. Temperature is zero and DeepSeek thinking is disabled.
The production MTP/CUDA-graph profile is tested separately so performance
features do not confound capability comparisons.

## Behavior

- Calibration source: `heretic-org/Semantic-Harmless` at revision
  `7e9f2b01272da85f2be7a3437f31ac46698e8735`.
- The pair-level holdout contains 85 harmful and 85 semantically matched
  harmless requests not used to estimate directions.
- The first metric is a transparent refusal-marker classifier over 96-token
  deterministic responses. It is paired with exact McNemar statistics and is
  explicitly only a sweep screen: all 85 C2 harmful responses reached the
  96-token limit, so marker absence is not treated as final compliance.
- The release gate regenerates the 85 harmful holdout responses with a
  512-token budget. The untouched source model then runs a frozen three-way
  `REFUSE` / `PARTIAL` / `COMPLY` rubric with a short evidence rationale over
  both source and candidate responses. Strict and material-compliance
  transitions are paired by prompt, and suspicious redirects are manually
  audited.
- A separate, repository-tracked bilingual probe contains five independently
  authored harmful/harmless semantic pairs in both English and Simplified
  Chinese (20 prompts total). It covers cyber abuse, credential phishing,
  non-consensual tracking, theft, and coercion. It is not used for direction
  estimation or candidate selection. File SHA-256:
  `ba1587a255168d48e4048caac84ab733ec35510ab7fcbdce7df2f823ba89a966`.
  This small probe checks cross-language transfer and benign contrast; it is
  reported descriptively rather than treated as a statistically powered
  benchmark.

## Synthetic regression suite

`scripts/build_capability_suite.py` creates 19 deterministic cases covering
strict English/Chinese instructions, arithmetic, JSON, DeepSeek tool-call
parsing, and retrieval near the start/middle/end of a roughly 5k-token
synthetic log. The generated suite SHA-256 is
`9bbf2b65b48bdc42a76038aa60e2c8943d9aec0f3af23fdea98640480a36b565`.

## Open-ended writing preservation

The repository-tracked writing suite contains 12 prompts, split evenly between
English and Simplified Chinese. Six matched categories cover literary fiction,
professional rewriting, faithful summarization, technical explanation,
balanced argument, and constrained dialogue. Dataset SHA-256:
`1b6749963ab9f513cf28c9a80ce51efc007de455baf3c938858ee0dade634d31`.

Source and candidate generate with the same eager/no-MTP TP=2 profile,
temperature zero, thinking disabled, and a final 768-token budget. Earlier
384- and 512-token preflights were excluded after exposing asymmetric
artificial truncation; the final run had 12/12 natural completions for each
model. The untouched source checkpoint then compares anonymous A/B pairs using
task fidelity, coherence, style/audience fit, and language correctness (1–5
each). A/B order is deterministically balanced from seed 731. The judge is
conservative because it is the source model itself; identities remain hidden,
all rationales are retained, malformed judge JSON is retried without replacing
valid judgments, and every loss/tie is manually audited. This is a paired
regression probe, not a general writing leaderboard.

## GSM8K

- OpenAI repository revision:
  `3101c7d5072418e28b9008a6636bde82a006892c`.
- Test JSONL SHA-256:
  `3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14`.
- A seed-731 selection of 100 test questions is used; selected-index SHA-256:
  `3fadf738b503bb943448bde5a8c4dc8feb9044aac2fd94d900c64ad3aa12ba22`.
- The model receives a zero-shot chat prompt, a 256-token output budget, and an
  explicit `#### <number>` final-answer format.

This is a locked relative preservation screen, not a claim of comparability to
published few-shot or chain-of-thought GSM8K numbers.

## HumanEval

- OpenAI repository revision:
  `6d43fb980f9fee3c892a914eda09951f772ad10d`.
- Dataset SHA-256:
  `b796127e635a67f93fb35c04f4cb03cf06f38c8072ee7cee8833d7bee06979ef`.
- The seed-731 selection contains 32 tasks; selection SHA-256:
  `aeb876fcea8edcd284945ffca51010393c90a7d0723801ae7efc2d96fa29427a`.
- After the 32-task screen tied exactly, the final confirmation ran all 164
  HumanEval tasks with the same deterministic generation and isolated scoring
  protocol. No candidate selection was performed on the full-set result.
- Generated code is executed only in a separate networkless, read-only,
  capability-dropped container with CPU, memory, PID, file, and wall-clock
  limits. The container receives neither model weights nor the API key.

This is a deterministic chat-generation pass@1 preservation screen rather
than the exact sampling protocol used by the original HumanEval paper.

## MMLU-Pro

- TIGER-Lab dataset revision:
  `b189ec765aa7ed75c8acfea42df31fdae71f97be` (including the 2026 option-spacing
  correction).
- Test parquet SHA-256:
  `0e24a191921c2f453518a537a8b2117bd137e7714d4ef1565e9ba06c1ecb9ad8`.
- Selections are category-balanced across all 14 categories. The 140-question
  screen uses 10/category and SHA-256
  `6b9742ac39794f3fe008f2032908bb463482461279e443d9cfa73c7af9426ace`.
- If the 140-question result is within one item of a gate, the confirmatory set
  uses 40/category (560 total), seed 731, SHA-256
  `c170258f9b8b381510bafd17e2c2a838ea20a9d482483a8f9318aeda433429bf`.
- The direct-answer prompt returns only `Answer: <letter>` to prevent output
  budget truncation from being scored as subject-matter failure.

## Interpretation

The release gate is a paired preservation decision, not benchmark
leaderboarding. An aggregate change outside three absolute points triggers a
larger confirmation set or a new candidate. Small category slices are reported
for diagnostics but are not individually thresholded as if ten questions were
a precise population estimate.

Detailed code and writing disagreement audits are in
`docs/INTELLIGENCE_PRESERVATION.md`.
