# Third-party components

| Component | Where | License | Use |
| --- | --- | --- | --- |
| crit Core 0.1.0 | built from the [crit](https://github.com/crit-labs/crit) repository (not included) | Apache-2.0 | Request runtime (Model/Session/Engine), resource ledger, CUDA backend services, sampler, strict JSON; statically linked into `libfnl.so` |
| crit Python packages (`crit_core.text`, crit Eval, crit serve) | from the crit checkout (not included) | Apache-2.0 | Tokenizer and chat template for the chat adapter; the Eval and serve hosts that load the adapters |
| `tokenizers`, `jinja2` | Python environment (not included) | Apache-2.0; BSD-3-Clause | Used by `crit_core.text` |
| NVIDIA CUDA runtime | system CUDA toolkit 13.1 (not redistributed) | NVIDIA CUDA Toolkit EULA | Kernel launches, graphs, memory |
| flashinfer b12x `nvfp4_csf` codec | not included | Apache-2.0 | The NVFP4 CSF scale decoder follows its byte-window format; no flashinfer code is used |
| Hugging Face transformers, `models/qwen4_exp` | not included | Apache-2.0 | Read as the arithmetic specification of the text model and used for the FP32 quality reference; no code copied |
| nvidia/Qwen3.8-Flash-Next-FP4-Local checkpoint, tokenizer and chat template | read from the user's local model folder (not redistributed) | NVIDIA Open Model Agreement; Qwen Community License 1.0 | Weights, tokenizer.json, chat template |

Everything else (CUDA kernels, CSF decoding, the expert cache, speculative rounds, the Core
Model/Session/Backend integration and the Python adapters) was written for this engine.
No other inference engine's source, kernels or prepared artifacts were used.
