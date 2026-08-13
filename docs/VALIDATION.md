# Validation log

This file records commands and measured outputs for this repository. It is not
a claim of completion until the end-to-end candidate gates in `METHOD.md` have
passed.

## 2026-08-11 — environment preflight

- Persona API and Gradio containers stopped and removed.
- Ports 7860 and 8000 are not listening on the head.
- Head available unified memory after stop: approximately 117 GiB.
- Both nodes have approximately 3.1 TiB free local NVMe space.
- Both nodes contain the pinned Anemll image ID
  `sha256:3430d6614a8e2925f34d059af6caf05aff42387326db4d05639a60f10f2654d8`.
- Exact source checkpoint exists on both nodes and occupies approximately
  156 GiB per node.
- Source inventory: 48 shards, 43 main `wo_b` weights and scales, 3 MTP `wo_b`
  weights and scales; main weight shape `[4096,8192]`, scale shape `[32,64]`.

## 2026-08-11 — paired activation capture

- Dataset: `heretic-org/Semantic-Harmless`, revision
  `7e9f2b01272da85f2be7a3437f31ac46698e8735`, source SHA-256
  `d066ddd4a8005271d21f269aacc4cc30bfb5d2f1d7004f3f37004e42f63f5225`.
- Deterministic pair-level split: 331 harmful + 331 harmless train; 85 + 85
  holdout.
- Capture run 2: 832/832 valid, zero failures, 126.41 seconds, 6.58
  samples/second; every artifact contains `attn_in`, `attn_out`, and `ffn_out`
  tensors of shape `[43,4096]` in F16.
- An independent first run also captured 832/832. Across runs, raw-direction
  cosine was at least 0.99985 and averaged at least 0.99996 for attention/FFN
  outputs. After fixing the randomized PCA seed, layer-local SRA cosine was at
  least 0.9947 for attention input and at least 0.9971 for attention output;
  global attention-output SRA cosine was 0.99973.
- Layer-0 `attn_in` is exactly inactive for paired prompts sharing the final
  chat-template token. This is expected before the first attention operation;
  the estimator records it as inactive and uses layers 1–42.
- Holdout SRA ROC-AUC means: `attn_in` 0.9954, `attn_out` 0.9963, `ffn_out`
  0.9951. Mean off-diagonal layer cosine: 0.2114, 0.0309, and 0.0024
  respectively. The near-orthogonal output geometry supports layer-local edits.
- The public attention-output artifact was regenerated after an audit changed
  global-direction selection/weighting from holdout AUC to train AUC. Only
  `global.raw` and `global.sra` changed; all candidate-relevant layer-local
  tensors remained bit-identical. Final artifact SHA-256:
  `fe8c263a8d32deb71e3f6e866b90f8246f452f6e2103b0e0400a77480fd2602a`.

## 2026-08-11 — native FP8 edit preflight

- A first dequantize/requantize probe exposed conservative upstream UE8M0
  scales: recomputing scales from already-quantized values altered weights even
  without an edit. The editor was changed to preserve the original scale as a
  fixed default rather than silently shrinking it.
- After the fix, real source layers 0, 10, 26, and 42 each round-tripped with
  bit-exact E4M3 weights and bit-exact UE8M0 scales; changed value count was
  zero for every probe.
- Layer 26 dry-run, layer-local SRA, lambda 1.0, no row-norm rematch: relative
  FP32 edit norm 0.02268; post-quant projection residual 0.26073; post-quant
  relative delta 0.02243; 23.70% of FP8 codes changed. Fixed scales clipped 68
  values in 67/2048 blocks (0.000203% of matrix values), with maximum requested
  code magnitude 458.21 versus the E4M3 limit 448.
- Lambda 1.5 and 2.0 had worse post-quant projection residuals (0.3984 and
  0.9091) and larger relative deltas, so lambda 1.0 was tested first. For
  lambda 2.0 the unsigned residual is expected to approach one because the
  selected component is reflected rather than removed.

## 2026-08-11 — candidate integrity and paired behavior

- C1: layer-local attention-output SRA rank 1, layers 10–42, lambda 1.0,
  fixed source UE8M0 scales, no row-norm rematch, stock MTP.
- C1 passed full validation on both nodes: 72,317 tensors / 48 shards; 33
  rewritten shards; 51,830 non-target tensors in those shards bit-exact; all
  33 scale tensors unchanged; all edited-shard SHA-256 values identical across
  nodes. Its 40-sample harmful refusal rate fell only from 95% to 70%, so it is
  retained as the conservative control rather than the release candidate.
- C2 uses the same direction, layers, scale policy, and stock MTP, with lambda
  2.0 (Householder reflection). It independently passed the same full
  tensor-granularity validation on both nodes. Mean relative edit norm was
  0.04702, maximum relative quantization error 0.02101, and 38.63% of FP8 codes
  changed. Fixed scales clipped 6,591 values across all 1.107 billion edited
  matrix values (approximately 0.000595%); no scale block changed. All 33
  edited-shard hashes were identical across nodes.
- Both C2 containers mounted the candidate directory, formed TP ranks 0/1 on
  separate Sparks over NCCL, loaded all 48 shards, and passed a real chat API
  request with zero container restarts.
- Exact-source full holdout: harmful 84/85 refusals (98.82%), harmless 1/85
  (1.18%). C2 full holdout: harmful 4/85 (4.71%), harmless 1/85 (1.18%). Paired
  transitions on harmful prompts were 80 refusal-to-compliance, zero
  compliance-to-refusal, four refusal-to-refusal, and one
  compliance-to-compliance; exact McNemar p=`1.65e-24`.
- These are lexical refusal-marker results, not a claim that every non-refusal
  is complete or useful. A separate strict refusal/partial/compliance audit is
  part of the remaining gate.

## 2026-08-11 — C2 capability gate

- Deterministic 19-case diagnostic: source 13/19, C2 13/19, with all 19
  item-level outcomes identical. Both scored tool calls 3/3, JSON 3/3,
  long-context retrieval 2/3, arithmetic 4/6, and strict instruction format
  1/4.
- GSM8K seed-731 100-question screen: source 97/100, C2 96/100. The only
  discordant transition was one source-pass/C2-fail item; exact McNemar
  `p=1.0`.
- HumanEval seed-731 32-task deterministic chat pass@1 screen: source 27/32,
  C2 27/32, with all 32 item-level outcomes identical. Generated programs were
  scored in the isolated networkless container described in `EVALUATION.md`.
- The initial balanced MMLU-Pro 140 screen was source 94/140 (67.14%) versus
  C2 89/140 (63.57%), a non-significant five-item difference (`p=0.180`) and
  one item beyond the three-point gate. This triggered the preregistered larger
  confirmation rather than a pass/fail claim from the small slice.
- Balanced MMLU-Pro 560 confirmation: source 379/560 (67.68%), C2 382/560
  (68.21%), delta `+0.54` points. Paired transitions were 368 pass/pass, 11
  pass/fail, 14 fail/pass, and 167 fail/fail; exact McNemar `p=0.690`.
- All locked capability aggregates are therefore inside the three-point
  preservation gate.

## 2026-08-11 — final behavior and bilingual gates

- The final 512-token harmful holdout contained 85 paired source/C2 responses.
  The untouched source-model judge labeled source as 82 `REFUSE`, two
  `PARTIAL`, and one `COMPLY`; C2 was eight `REFUSE`, 33 `PARTIAL`, and 44
  `COMPLY`.
- Strict `COMPLY` rose from 1/85 (1.18%) to 44/85 (51.76%), a paired increase
  of 50.59 points (`p=2.27e-13`). Material compliance (`PARTIAL` or `COMPLY`)
  rose from 3/85 (3.53%) to 77/85 (90.59%), an 87.06-point increase with 74
  positive and zero negative transitions (`p=1.06e-22`).
- All eight C2 `REFUSE` and 33 `PARTIAL` responses were manually inspected.
  The remaining labels reflected real redirects, incompleteness, or response
  boundaries rather than systematic judge/parser failure.
- The separate five-category English/Chinese probe had ten harmful and ten
  matched harmless prompts. Source harmful labels were 10/10 `REFUSE`; C2 was
  seven `PARTIAL` and three `COMPLY`, giving 0/10 versus 10/10 material
  compliance (`p=0.001953`). Both models had zero lexical refusals on the ten
  harmless controls.

## 2026-08-11 — full coding and writing preservation

- Full deterministic HumanEval confirmation: source 150/164 (91.46%), C2
  148/164 (90.24%), delta `-1.22` points. Transitions were 147 pass/pass, three
  pass/fail, one fail/pass, and 13 fail/fail; exact McNemar `p=0.625`.
- All 328 HumanEval generations ended naturally, with no empty or
  length-limited output. Manual inspection of the four disagreements found
  ordinary boundary-condition choices (separator empties, a digit-only range,
  Newton fallback, and signed leading digits), not refusal or output-format
  collapse. C2 fixed the signed-digit case that source missed.
- The final writing run used 12 frozen prompts (six English, six Chinese),
  temperature zero, thinking off, and 768 output tokens. Both models completed
  all 12 naturally. Mean completion tokens were 222.67 source and 226.67 C2.
- The untouched-source blind A/B judge returned four C2 wins, two source wins,
  and six ties, with mean totals 19.25/20 C2 versus 18.58/20 source. Manual
  review found two C2-win rationales that misstated source-answer details; both
  were conservatively corrected to ties. The audited result is therefore two
  C2 wins, two source wins, and eight ties.
- Deterministic length/format compliance was 8/12 for both. Both missed the
  requested length on the same four explanation/argument prompts. No writing
  preservation gate regressed.

## 2026-08-11 — release build

- `VERIFY_SOURCE_FULL_HASH=1 ./scripts/build_release_pair.sh` verified all 48
  pinned source shards and source metadata independently on both nodes.
- The first editor pass completed all 33 targets but exposed a wrapper defect:
  Docker wrote the destination as root, so the metadata-copy phase correctly
  stopped rather than publishing an incomplete release. Both incomplete
  directories were moved aside; the wrapper now runs both editor and validator
  with the invoking UID/GID, and a regression test asserts both user mappings.
- The rebuilt final directories are
  `/home/snowywar/models/DeepSeek-V4-Flash-0731-CRACK` on each node and are
  owned by `snowywar:snowywar`. Validation indexed 72,317 tensors, found exactly
  33 edited targets, verified 51,863 non-target tensors bit-exact, retained 15
  unchanged shards as local hard links, and matched every rewritten-shard hash
  across the two independent builds.
- Direction SHA-256 is
  `fe8c263a8d32deb71e3f6e866b90f8246f452f6e2103b0e0400a77480fd2602a`.
  Final rewritten shards also matched the evaluated C2 shards exactly.
- After production mounted the final directories and passed inference, the two
  failed root-owned build copies were deleted. They were not mounted or
  recoverable; the formal releases, source checkpoints, and evaluation
  candidates were retained.

## 2026-08-11 — production TP=2 and API gate

- The first production start immediately failed at NCCL communicator creation
  with CUDA OOM. The preceding full hashes/rewrites had left 115–116 GiB of
  reclaimable file page cache per node and only 3–4 GiB immediately free on the
  GB10 unified-memory systems. After `sync` and `vm.drop_caches=3`, each node
  had 119 GiB free and the unchanged profile started successfully. This was a
  host-cache pressure issue, not a model or fabric failure.
- Both running containers mount the final CRACK directory at the in-container
  source-compatible model path. Rank 0 and rank 1 run on different Sparks over
  NCCL 2.30.7; both containers report `running`, restart count zero, and
  `OOMKilled=false`.
- Production resolved the main architecture and stock `DeepSeekV4MTPModel`,
  loaded all 48 shards on each node, captured piecewise/full main CUDA graphs
  and full DSpark speculative graphs, and advertised 1,048,576 tokens. The
  profile uses five draft tokens, async scheduling, NVFP4 MLA KV cache,
  `max_num_seqs=6`, and `max_num_batched_tokens=8192`.
- API bind is only `192.168.31.200:8888`. `/v1/models` returned no-key `401`,
  bad-key `401`, and correct-key `200`, advertising only
  `deepseek-v4-flash-0731-crack` with the 1M context value.
- OpenAI chat returned the exact requested marker. DeepSeek auto-tool parsing
  returned a valid `add` call with JSON arguments `37` and `5`. Anthropic
  Messages returned a thinking block plus final `ANTHROPIC_OK` using Bearer
  authentication.
- Pi Agent single-stream benchmark (thinking off, one warmup plus three
  measured runs): median TTFT `0.28s`, median decode `77.59 tok/s`, median
  end-to-end `75.65 tok/s`, decode range `77.16–91.43 tok/s`. Report:
  `/Users/snowywar/Desktop/dgx/outputs/pi-dspark-token-throughput-crack.json`.
- After the Pi run, vLLM metrics recorded 3,005 accepted of 3,780 speculative
  draft tokens, a 79.50% MTP token acceptance rate.
- Both f0 RoCE links remained `ACTIVE / LINK_UP` on both HCAs, worker Wi-Fi
  remained disabled, its only default route remained through `.10`, worker
  internet returned HTTP 200, and both nodes had no failed systemd units.
- Final state: the CRACK TP=2 API remains online. Persona API/Gradio, MiniMax
  H3, Ray, and the raw ComfyUI backend remain stopped; the Nginx 8188 entry is
  still present with its backend offline.

## 2026-08-13 — MiaAI runtime hotfix update

- MiaAI Lab upstream was inspected at commit `018c6bc`. Its rewritten `main`
  history was not merged into the existing local deployment checkout.
- Nine validated Anemll 0.1.1 fixes were imported: project issues #21, #22,
  #24, #26, #27, and #31, plus vLLM backports #50312, #50004, and #49486.
  Dormant #48407 and the not-yet-live #48957/#50298 scripts were excluded.
- The production profile now sets `LONG_PREFILL_TOKEN_THRESHOLD=1024` and
  `VLLM_PREFIX_CACHE_RETENTION_INTERVAL=4096`, while retaining 1M context,
  six sequence slots, MTP=5, GPU utilization 0.80, and default low thinking.
- Both nodes ran every selected hotfix before the vLLM process started. All
  status markers reported `APPLIED`; containers remained running with zero
  restarts and `OOMKilled=false`.
- API checks passed for Bearer authentication, ordinary chat, two-turn tool
  calling, JSON Schema with thinking, and `thinking_token_budget=32`.
- The verified deployment runtime is now tracked under
  `runtime/miaai-dspark/`, so production no longer depends on an untracked
  external MiaAI checkout. The Anemll image itself remains external and is
  fixed by immutable manifest digest.
