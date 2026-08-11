# Research freeze — 2026-08-11

## Findings that determine the implementation

- DeepSeek V4 Flash is a 43-layer 285B-class MoE with mHC and 1M-context
  attention. Its architecture is not equivalent to the residual-stream layout
  assumed by early 2024 refusal-direction examples.
- Current public DeepSeek V4 edits converge on attention output projection as
  the least destructive target. One documented implementation captures every
  4096-dimensional `wo_b` output on all 43 layers under TP=2 and reports that
  broad MLP edits harmed capabilities.
- 2026 mechanistic work finds more than one refusal-related direction and
  task-dependent over-refusal subspaces. Therefore this project measures
  layer/category geometry instead of assuming one universal vector.
- The exact 0731 checkpoint stores every main `attn.wo_b.weight` as
  `[4096,8192]` E4M3 with `[32,64]` UE8M0 scales. Three MTP projections use the
  same layout. A correct release must preserve these tensors and the DSpark
  speculative path.
- The public JANG bundle is 102-shard MLX affine mixed quantization and records
  `bundle_has_mtp=false`; it cannot be served by the local CUDA/vLLM runtime.

## Primary references

References were rechecked on 2026-08-11; the implementation is not based on
the older 2024 residual-stream recipes alone.

- vLLM team, ["DeepSeek V4 in vLLM: Efficient Long-context Attention"][vllm-blog],
  2026-04-24; current [DeepSeek V4 support roadmap][vllm-roadmap] and
  [vLLM releases][vllm-releases].
- DeepSeek [`DeepSeek-V4-Flash-0731`][upstream-0731] and the official
  [`DeepSeek-V4-Flash-DSpark`][upstream-dspark] repositories.
- [`lovesenko/DeepSeek-V4-Flash-DSpark-Abliterated`][lovesenko] model card.
- [`dealignai/DeepSeek-V4-Flash-0731-JANG-CRACK`][jang] model metadata.
- ["There Is More to Refusal in Large Language Models than a Single
  Direction"][multi-refusal], arXiv:2602.02132.
- ["Over-Refusal and Representation Subspaces"][over-refusal],
  arXiv:2603.27518.
- ["Efficient Refusal Ablation in LLM through Optimal Transport"][ot-refusal],
  arXiv:2603.04355.
- ["Selective Refusal Ablation"][sra], arXiv:2601.08489.

[vllm-blog]: https://github.com/vllm-project/vllm-project.github.io/blob/main/_posts/2026-04-24-deepseek-v4.md
[vllm-roadmap]: https://github.com/vllm-project/vllm/issues/40902
[vllm-releases]: https://github.com/vllm-project/vllm/releases
[upstream-0731]: https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731
[upstream-dspark]: https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-DSpark
[lovesenko]: https://huggingface.co/lovesenko/DeepSeek-V4-Flash-DSpark-Abliterated
[jang]: https://huggingface.co/dealignai/DeepSeek-V4-Flash-0731-JANG-CRACK
[multi-refusal]: https://arxiv.org/abs/2602.02132
[over-refusal]: https://arxiv.org/abs/2603.27518
[ot-refusal]: https://arxiv.org/abs/2603.04355
[sra]: https://arxiv.org/abs/2601.08489

## Frozen initial sweep after activation geometry

The first sweep is intentionally small enough to run end-to-end before adding
complexity:

1. Layer-local SRA-cleaned rank-1, layers 10–42, lambda 1.0, fixed source
   scales, no row-norm rematch, MTP stock.
2. If candidate 1 is behaviorally too weak, test the geometrically distinct
   lambda 2.0 Householder reflection before adding layers or modifying MTP.
3. Compare an attention-output global SRA direction only if layer-local editing
   fails the Pareto gate. Do not use the FFN global direction: independently
   seeded captures showed stable layer-local FFN directions but unstable global
   aggregation because cross-layer directions nearly cancel.

Only the Pareto-leading candidate advances to the full benchmark suite.

## Sweep outcome

- Candidate 1 (`lambda=1.0`) removed the measured component but remained too
  conservative: its 40-sample harmful refusal rate was 70%, versus 95% for the
  exact source under the same deterministic profile.
- Candidate 2 (`lambda=2.0`) reflects the component. Its full 85-prompt harmful
  holdout refusal-marker rate was 4.71%, versus 98.82% for source, while the
  paired harmless rate remained 1.18% for both. The paired McNemar exact
  p-value was `1.65e-24`.
- Candidate 2 was therefore the only candidate advanced to the locked
  capability, writing, and production-MTP gates. It passed all three: full
  HumanEval changed by -1.22 points (`p=0.625`), MMLU-Pro-560 changed by +0.54
  points (`p=0.690`), the conservatively audited writing result tied 2–2 with
  eight ties, and the production MTP token acceptance rate was 79.50%.
- A diagnostic rank-expansion probe used only the exported activations (no
  checkpoint load) at layers 10, 20, 30, and 42. After removing the rank-1 SRA
  direction and benign capability atoms, the first four paired-difference PCs
  had individual signed holdout AUCs of only 0.50–0.60. At layer 10, expanding
  projection-energy rank from 1 to 3 reduced AUC from 0.987 to 0.626. A naive
  multi-rank Householder edit is therefore not justified by the captured
  geometry; rank expansion remains a falsified branch rather than an untested
  assumption.
