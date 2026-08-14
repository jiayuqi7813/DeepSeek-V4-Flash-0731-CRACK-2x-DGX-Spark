# Vendored MiaAI DSpark production runtime

This directory contains the production subset used to serve the CRACK
checkpoint on two DGX Sparks.

- Upstream: `MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark`
- Upstream commit inspected: `018c6bc`
- Runtime image: `ghcr.io/anemll/dspark-vllm-gx10:0.1.1`
- Runtime image manifest: `sha256:a83948492cf13df455170fb42885f5ef4db54fefe0feff0f841ecbff464ac9d8`

The Compose profile and launchers retain CRACK-specific local model mounts,
API-key loading, fixed model revision support, and the verified dual-CX-7
network configuration. The selected hotfix files are available inside each
new container, but their runtime overlays are disabled by default. Enable only
the required overlay through the corresponding `ENABLE_*` profile variable.
The validated production baseline keeps every overlay and long-prefill tuning
switch at `0`.

The vendored set deliberately excludes the experimental vision sidecar,
dormant dense-prefill patch #48407, and the #48957/#50298 scripts that had not
completed live validation in the inspected upstream revision.
