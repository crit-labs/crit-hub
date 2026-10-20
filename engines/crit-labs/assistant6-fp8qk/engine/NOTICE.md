# g4 notices

g4 is a bespoke engine for nvidia/Gemma-4-31B-IT-NVFP4 with the google/gemma-4-31B-it-assistant drafter on an
RTX 5090, built in crit run run-20261018T211706Z-f7e30ecd.

- `src/gemv.cu`, `src/lt.cpp`, `src/lt.h`, `src/encode.cu`, the sampling kernels in `src/net.cu` and the Core glue in
  `src/engine.cpp`: adapted from the kestrel engine of crit run run-20261017T045435Z-b4d9b011 (written for crit runs).
  The attention tile machinery in `src/attn.cu` is adapted from crit Inference `inference/fp8-kv-attention/code/`
  (Apache-2.0, crit authors).
- crit Core (static libraries linked into `libg4.so`): Apache-2.0, crit authors; built from the crit checkout as
  CORE.md describes.
- NVIDIA CUDA runtime and cuBLASLt: linked dynamically from the CUDA 13.1 toolkit on the target machine (not
  redistributed), NVIDIA CUDA EULA.
- Model weights (not redistributed): nvidia/Gemma-4-31B-IT-NVFP4 and google/gemma-4-31B-it-assistant, Gemma terms /
  Apache-2.0 as published on Hugging Face.
- Python adapters use `crit_core.text` from the crit checkout plus `tokenizers` and `jinja2` from the serving
  environment (not vendored).

All other source in this tree was written for this run.
