# DeepSeek V4 Flash 0731 CRACK for 2× DGX Spark

This project builds a reproducible, reversible refusal-subspace edit of the
**exact** `deepseek-ai/DeepSeek-V4-Flash-0731` checkpoint for tensor-parallel
inference across two NVIDIA DGX Sparks.

The project is intentionally not a conversion of the MLX/JANG checkpoint. The
JANG release targets Apple Silicon and drops the DSpark/MTP path. This project
keeps the original FP8 E4M3 + UE8M0 checkpoint layout, all 43 main layers, and
the three DSpark MTP projections used by the verified Anemll/vLLM runtime.

## Status

The validated 48-shard checkpoint is published at
[`Sn1waR/DeepSeek-V4-Flash-0731-CRACK-DSpark`](https://huggingface.co/Sn1waR/DeepSeek-V4-Flash-0731-CRACK-DSpark).
The initial weight release is pinned at Hub commit
`86d85ce97bdcb9897fb0d1dd9caf7ef57e124e1a`.

The released candidate is a layer-local SRA rank-1 Householder reflection on
attention `wo_b`, layers 10–42, with native fixed UE8M0 scales and stock MTP
weights. It passed full tensor integrity on both nodes. On the 85-prompt
held-out behavior audit, material compliance rose from 3.53% to 90.59%; the
separate bilingual harmful probe rose from 0/10 to 10/10, with no added lexical
refusals on matched harmless prompts.

The capability-preservation gate also passed. Full HumanEval was 150/164 for
source and 148/164 for C2 (-1.22 points, exact McNemar `p=0.625`); GSM8K-100
was 97/100 versus 96/100; balanced MMLU-Pro-560 was 379/560 versus 382/560.
The conservative manually audited bilingual writing result was two C2 wins,
two source wins, and eight ties. See
[`docs/INTELLIGENCE_PRESERVATION.md`](docs/INTELLIGENCE_PRESERVATION.md) for
the code-disagreement audit, writing methodology, and limits of the claim.
The same release measurements are available in machine-readable form at
[`docs/RELEASE_RESULTS.json`](docs/RELEASE_RESULTS.json).

The final checkpoint was independently built and hash-matched on both Sparks,
then validated in the production TP=2 profile with stock five-token MTP, CUDA
graphs, a 1,048,576-token advertised context window, OpenAI and Anthropic
routes, tool calls, and API-key enforcement. The measured MTP token acceptance
rate after the Pi throughput run was 79.50%.

The repository also vendors the production launcher and the Anemll 0.1.1
hotfix set validated on 2026-08-13. The update covers tool-argument encoding,
NVFP4 long-context decode, structured-output reasoning boundaries, hybrid
prefix caching, partial-prefill scheduling, V2 thinking budgets, and three
verified vLLM performance backports. These runtime files are pinned to MiaAI
Lab upstream commit `018c6bc`; model weights and the CRACK edit are unchanged.

## Execution boundary

Model weights are never loaded on the control Mac. Checkpoint editing,
validation, TP=2 inference, and benchmarks execute on the DGX pair. The Mac is
used only for source control, small JSONL/parquet preprocessing, orchestration,
and analysis of exported activation or text artifacts.

## Method in one page

1. Lock the source checkpoint revision and tensor inventory.
2. Capture last-prefill-token activations at three 4096-dimensional surfaces
   on all 43 layers: attention input, `wo_b` output, and aggregate FFN output.
3. Use semantically paired harmful/harmless prompts with a disjoint holdout.
4. Compute per-layer refusal directions, ROC-AUC, cross-layer cosine structure,
   and capability-PC-cleaned (SRA) alternatives.
5. Edit only attention `wo_b` output matrices, preserving the native
   128×128-block FP8/UE8M0 encoding. MLP edits are diagnostics, not the default.
6. Sweep global versus layer-local rank-1 directions, layer ranges, edit
   strength, row-norm preservation, and MTP handling.
7. Reject candidates that violate checkpoint integrity, lose more than three
   absolute points on the capability suite, regress tool calling, or exceed the
   agreed performance envelope.

This is direct model editing, not fine-tuning. Training is not required for the
primary path. See `docs/METHOD.md` for the decision gate that would justify a
training experiment.

At strength 1 the transform removes the selected component. At strength 2 it
is the Householder reflection `I - 2 VᵀV`, which flips that component while
preserving the column Gram matrix in exact arithmetic. The latter is the
current candidate.

## Source lock

- Model: `deepseek-ai/DeepSeek-V4-Flash-0731`
- Revision: `9e165c30e2704aec5d9d593cce3eebd58bbef1cb`
- Runtime image: `ghcr.io/anemll/dspark-vllm-gx10:0.1.1@sha256:a83948492cf13df455170fb42885f5ef4db54fefe0feff0f841ecbff464ac9d8`
- Runtime image ID: `sha256:3430d6614a8e2925f34d059af6caf05aff42387326db4d05639a60f10f2654d8`
- Topology: vLLM TP=2 / NCCL over the verified dual-CX-7 RoCE fabric

## Repository layout

- `src/dspark_crack/`: identity, analysis, and FP8 editing implementation
- `runtime/`: opt-in capture hook plus the pinned, production-ready MiaAI DSpark runtime
- `scripts/`: dataset, capture, deployment, and validation entry points
- `deploy/`: capture-only two-node Compose profile
- `docs/`: method, research record, capability analysis, and measured results
- `tests/`: deterministic unit and format tests

Raw prompts, captures, candidates, and benchmark responses live under
`artifacts/` or `data/generated/` and are excluded from Git. The validated
leading direction and its aggregate report are intentionally tracked at
`artifacts/directions/attn-out-sra-r8.safetensors` (SHA-256
`fe8c263a8d32deb71e3f6e866b90f8246f452f6e2103b0e0400a77480fd2602a`) and
`artifacts/directions/attn-out-sra-r8.report.json`; they contain tensors and
metrics, not prompt text or model weights.

## Control-environment setup

Python 3.11 or newer is required for orchestration, activation analysis, and
tests. This environment does not load the model checkpoint:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,eval]'
ruff check .
pytest -q
```

Checkpoint editing and validation use the pinned DGX runtime image through the
provided shell wrappers, so their FP8 behavior matches deployment.

## Rebuild the current candidate

The release builder syncs this repository, optionally verifies every source
LFS shard, refuses to overwrite an existing destination, builds independently
on both Sparks, runs tensor-granularity validation on each copy, and compares
all rewritten-shard SHA-256 values across the nodes:

```bash
VERIFY_SOURCE_FULL_HASH=1 ./scripts/build_release_pair.sh
```

The default output on both nodes is
`~/models/DeepSeek-V4-Flash-0731-CRACK`. The validation checks all non-target
tensors inside rewritten shards, not just file presence or model startup. The
lower-level `edit_checkpoint_in_runtime.sh`, `verify_source_in_runtime.sh`, and
`validate_candidate_in_runtime.sh` wrappers remain available for experiments.

## Production deployment

Copy `deploy/production.env.example` to an ignored local profile, set the two
node paths, then start the verified production runtime:

```bash
PROFILE_ENV_BASENAME=.env.production.local \
  ./scripts/start_production_cluster.sh

PROFILE_ENV_BASENAME=.env.production.local \
  ./scripts/stop_production_cluster.sh
```

The production profile preserves the DSpark MTP speculative path, 1M declared
context, CUDA graphs, TP=2, and dual-CX-7 NCCL fabric. It uses the versioned
runtime under `runtime/miaai-dspark/`; a separate checkout of the original
MiaAI repository is no longer required. The stop wrapper always stops the
worker before the head.

## Evaluation

`scripts/fetch_eval_datasets.py` downloads pinned GSM8K, HumanEval, and the
current corrected MMLU-Pro parquet and verifies their SHA-256 values. The
evaluation commands run on the DGX head against the two-node API. HumanEval
code is scored in a separate networkless, resource-limited container. Exact
protocols, hashes, selection seeds, and limitations are in
`docs/EVALUATION.md`.

## Acknowledgements

Special thanks to [MiaAI-Lab](https://github.com/MiaAI-Lab) and the
[DeepSeek V4 Flash DSpark project](https://github.com/MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark)
for the original two-DGX-Spark deployment work, Anemll/vLLM integration,
performance investigation, and runtime hotfixes that this project builds on.
The production runtime vendored here is pinned to MiaAI-Lab commit `018c6bc`,
with CRACK-specific checkpoint mounting, authentication, revision locking,
and dual-CX-7 deployment configuration layered on top.

## License and weights

The project code is MIT licensed. DeepSeek model weights remain subject to the
upstream model license. Dataset licenses and exact revisions are recorded in
`THIRD_PARTY_NOTICES.md`; generated model releases must carry the same notices
and an edit manifest. The validated checkpoint and those release artifacts are
available on [Hugging Face](https://huggingface.co/Sn1waR/DeepSeek-V4-Flash-0731-CRACK-DSpark).
