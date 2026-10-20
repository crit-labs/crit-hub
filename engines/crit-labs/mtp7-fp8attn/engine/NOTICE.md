# kestrel notices

kestrel is a bespoke engine for unsloth Qwen3.8-27B-NVFP4 on an RTX 5090, built in this crit run.

- `src/fp8_kv_attention.cu`, `src/fp8_kv_attention.h`: copied unchanged from crit Inference
  `inference/fp8-kv-attention/code/` (crit revision recorded in CORE.md), Apache-2.0, crit authors.
- crit Core (static libraries linked into `libkestrel.so`): Apache-2.0, crit authors; built from the crit checkout
  as CORE.md describes.
- NVIDIA CUDA runtime and cuBLASLt: linked dynamically from the CUDA 13.1 toolkit on the target machine (not
  redistributed), NVIDIA CUDA EULA.
- Python adapters use `crit_core.text` from the crit checkout plus `tokenizers` and `jinja2` from the serving
  environment (not vendored).

All other source in this tree was written for this run.
