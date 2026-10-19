# Third-party components

| Component | Where | License | Use |
| --- | --- | --- | --- |
| crit Core 0.2.1 | built from the [crit](https://github.com/crit-labs/crit) repository (not included) | Apache-2.0 | Request runtime (Model/Session/Engine), resource ledger, CUDA backend services, safetensors reader, sampler; linked into `libnmh.so` |
| crit Inference `fp8-kv-attention` | `engine/src/fp8_kv_attention.{cu,h}`, copied unchanged | Apache-2.0 | FP8 KV cache write, prompt (BF16 and E4M3 `prefill_fp8`) and decode attention |
| crit Python packages (`crit_core.native`, `crit_core.text`, crit Eval, crit serve) | from the crit checkout (not included) | Apache-2.0 | Native engine adapters, tokenizer and chat template; the Eval and serve hosts |
| fnl engine (crit-hub `qwen3.8-flash-next@FP4-Local/mtp3-expert-cache`) | adapted in `engine/src/gemm.cu`, `engine/src/experts.cu` | Apache-2.0 | FP16 tensor-core prompt GEMM structure; expert directory, plan and decode GEMV structure |
| kestrel engine (crit run run-20261017T045435Z-b4d9b011) | followed in `engine/src/fp4_prompt.cu` | Apache-2.0 | NVFP4 activation quantization (UE4M3 block scales from a static global scale, Sm1xx layout) |
| NVIDIA CUTLASS v4.7.1 headers | `engine/third_party/kernels/cutlass` (retained with `crit kernels`, git cb4247394dd82148787aed73e5dc7cef33cbf862) | BSD-3-Clause | SM120 block-scaled ptr-array grouped GEMMs for the NVFP4 prompt experts (`engine/src/fp4_grouped.cu`) |
| NVIDIA cuBLASLt, CUDA runtime | system CUDA toolkit 13.1 (not redistributed) | NVIDIA CUDA Toolkit EULA | FP8 prompt GEMMs; kernel launches, graphs, memory |
| Hugging Face transformers, `models/nemotron_h` | not included | Apache-2.0 | Arithmetic specification of the text model and the FP32 quality reference (with the config's Mamba dt limit); no code copied |
| nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4 checkpoint, tokenizer and chat template | read from the user's local model folder (not redistributed) | OpenMDW 1.1 (license_name openmdw-1.1 in the model card) | Weights, tokenizer, chat template |

Everything else (the Mamba-2, MoE, MTP and speculative-round kernels, the expert cache and CPU expert path, the
Core integration and the Python adapters) was written for this engine.
