# Publishing without moving model weights through the Mac

The GitHub repository contains the editor, deployment profiles, tests, locked
source identity, and the validated direction artifact. Full model weights stay
on the DGX pair throughout build, validation, serving, and publication.

## Code release

Run the normal Git/GitHub workflow from the control machine. The repository
must not contain local `.env.*` files, API keys, raw benchmark responses,
captures, or candidate checkpoints.

## Weight release

The validated release is public at
[`Sn1waR/DeepSeek-V4-Flash-0731-CRACK-DSpark`](https://huggingface.co/Sn1waR/DeepSeek-V4-Flash-0731-CRACK-DSpark),
initial Hub commit `86d85ce97bdcb9897fb0d1dd9caf7ef57e124e1a`. It contains 81 files,
including all 48 safetensors shards, with a verified logical size of
166,898,742,883 bytes.

Run the Hub upload on the DGX head against the final validated candidate
directory. A Hugging Face write token is required on that node:

```bash
ssh dgx-spark-1
hf auth whoami
hf upload HF_NAMESPACE/DeepSeek-V4-Flash-0731-CRACK-DSpark \
  /home/USER/models/DeepSeek-V4-Flash-0731-CRACK . \
  --exclude .hf-manifest.json
```

Replace `HF_NAMESPACE` and `USER`. `build_release_pair.sh` installs the final
model card as `README.md`, preserves the upstream card as
`UPSTREAM_README.md`, and adds the release notices before validation. Re-running
the same upload command resumes an interrupted large-folder upload and skips
content already present. The current
Hugging Face CLI documentation recommends `hf upload`; the legacy
`hf upload-large-folder` command is deprecated:
[CLI guide](https://huggingface.co/docs/huggingface_hub/en/guides/cli) and
[upload guide](https://huggingface.co/docs/huggingface_hub/en/guides/upload).

The hidden `.hf-manifest.json` is the original source-download manifest and is
retained locally for source/candidate validation. It must not be presented as a
manifest of the edited Hub repository. The builder copies it to the explicitly
named `SOURCE_HF_MANIFEST.json` for provenance; the upload command excludes the
misleading hidden filename.

Before upload, require all of the following:

- `CRACK_EDIT_MANIFEST.json`, `CRACK_EDIT_REPORT.json`, and
  `CRACK_VALIDATION.json` describe the same candidate.
- The direction SHA-256 matches the artifact committed in the code release.
- Edited shard hashes match across both DGX nodes and all non-target tensors
  pass the full validator.
- The model card records the exact upstream revision, runtime image, method,
  capability/behavior results, limitations, and MIT attribution.

Hard links used by the local copy-on-write build are ordinary files to the Hub
uploader; no source checkpoint or model tensor is copied to the Mac.
