# Third-party components

| Component | Where | License | Use |
| --- | --- | --- | --- |
| crit Core 0.2.1 | built from the [crit](https://github.com/crit-labs/crit) repository (not included), tag `v0.2.1` (the run used a later checkout whose Core C++ sources match it plus config-reader fixes) | Apache-2.0 | Request runtime (Model/Session/Engine), CUDA backend services, checkpoint and safetensors loading; statically linked into `libg4.so` |
| crit Inference `fp8-kv-attention` | adapted into `engine/src/attn.cu` (the attention tile machinery, extended for 512-dim keys-equal-values heads, the sliding ring and FP8 Q·Kᵀ) | Apache-2.0 | FP8 KV cache layout and attention tiles |
| kestrel engine (crit-hub `crit-labs/mtp7-fp8attn`) | adapted into `engine/src/gemv.cu`, `lt.cpp`, `lt.h`, `encode.cu`, the sampling kernels in `net.cu` and the Core glue in `engine.cpp` | Apache-2.0 | GEMV, cuBLASLt wrappers, encoding, sampling and Core integration |
| crit Python packages (`crit_core.text`, `crit_core.serving`, crit Eval, crit serve) | from the crit checkout (not included) | Apache-2.0 | Tokenizer, chat template, the shared tool-call parser and ResidentChat for the chat adapter; the Eval and serve hosts that load the adapters |
| `tokenizers`, `jinja2` | Python environment (not included) | Apache-2.0; BSD-3-Clause | Used by `crit_core.text` |
| NVIDIA CUDA runtime and cuBLASLt | system CUDA toolkit 13.1 (not redistributed) | NVIDIA CUDA Toolkit EULA | Kernel launches, graphs, memory; prompt GEMMs (FP8, NVFP4) |
| nvidia/Gemma-4-31B-IT-NVFP4 checkpoint, tokenizer and chat template | read from the user's local model folder (not redistributed) | Apache-2.0 (Gemma 4 license) | Weights, tokenizer.json, chat template |
| google/gemma-4-31B-it-assistant drafter | read from the user's local model folder (not redistributed) | Apache-2.0 (Gemma 4 license) | Draft model for speculative decoding |

Everything else (the Gemma attention, norm, GeGLU and epilogue kernels, the speculative rounds, the prompt path, the
Gemma tool-call syntax and the Python adapters) was written for this engine. No other inference engine's source, kernels or prepared artifacts
were used.
