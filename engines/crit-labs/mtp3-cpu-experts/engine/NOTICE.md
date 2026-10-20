# nmh notices

nmh is a crit engine for nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4 (text path), written for this run.

- `src/fp8_kv_attention.cu`, `src/fp8_kv_attention.h`: copied unchanged from crit's Inference entry
  `inference/fp8-kv-attention` (Apache License 2.0, crit).
- `src/gemm.cu`: adapted from the fnl engine's prompt GEMM (crit-hub
  `engines/crit-labs/mtp3-expert-cache/engine/src/gemm.cu`, Apache License 2.0), formats and epilogues changed for this model.
- The expert directory, plan and decode-GEMV structure follow the fnl engine's `experts.cu` (same source and license).
- `third_party/kernels/cutlass`: NVIDIA CUTLASS v4.7.1 headers (BSD-3-Clause, `CUTLASS-LICENSE.txt` in the
  package), retained with `crit kernels retain`; compiled into `libnmh.so` by `src/fp4_grouped.cu` (grouped NVFP4
  prompt GEMMs). The FP4 activation quantization in `src/fp4_prompt.cu` follows the kestrel engine's prompt kernels
  (crit run run-20261017T045435Z-b4d9b011, Apache License 2.0).
- Linked: crit Core (Apache License 2.0), CUDA runtime (NVIDIA CUDA Toolkit EULA, not redistributed).
