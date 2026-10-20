# Third-party components

| Component | Where | License | Use |
| --- | --- | --- | --- |
| crit Core 0.1.0 | built from the [crit](https://github.com/crit-labs/crit) repository (not included), tag `v0.1.0` (the run used checkout `bdaae9a`, whose Core and Inference are identical) | Apache-2.0 | Request runtime (Model/Session/Engine), CUDA backend services, checkpoint and safetensors loading; statically linked into `libkestrel.so` |
| crit Inference `fp8-kv-attention` | `engine/src/fp8_kv_attention.cu`, `engine/src/fp8_kv_attention.h`, copied from the same checkout and extended (a public `combine()` and pre-quantized query fields on `Decode`; `engine/NOTICE.md` calls the copy unchanged, which predates those additions) | Apache-2.0 | FP8 KV cache layout, cache writes and decode attention |
| crit Python packages (`crit_core.text`, crit Eval, crit serve) | from the crit checkout (not included) | Apache-2.0 | Tokenizer and chat template for the chat adapter; the Eval and serve hosts that load the adapters |
| `tokenizers`, `jinja2` | Python environment (not included) | Apache-2.0; BSD-3-Clause | Used by `crit_core.text` |
| NVIDIA CUDA runtime and cuBLASLt | system CUDA toolkit 13.1 (not redistributed) | NVIDIA CUDA Toolkit EULA | Kernel launches, graphs, memory; prompt GEMMs (FP8, NVFP4, BF16) |
| unsloth/Qwen3.8-27B-NVFP4 checkpoint, tokenizer and chat template | read from the user's local model folder (not redistributed) | Apache-2.0 | Weights, bundled MTP layer, tokenizer.json, chat template |

Everything else (the GEMV, DeltaNet, prompt-attention and epilogue kernels, the speculative
rounds, the Core Model/Session integration and the Python adapters) was written for this
engine. No other inference engine's source, kernels or prepared artifacts were used.
