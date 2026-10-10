# Third-party notices

## llama.cpp (b11463, commit d2a79e604)

The following files under `vendor/llamacpp/b11463/` are unmodified build outputs
of llama.cpp, redistributed under the MIT License:

- `bin/llama-server.exe`, `bin/llama-server-impl.dll`, `bin/llama.dll`,
  `bin/llama-common.dll`, `bin/llama-cli-impl.dll`, `bin/ggml.dll`,
  `bin/ggml-base.dll`, `bin/ggml-cuda.dll`, `bin/ggml-rpc.dll`,
  `bin/mtmd.dll`, `bin/libomp.dll`, `bin/ggml-cpu-*.dll`
- `LICENSE-LLVM-OpenMP` is retained for the OpenMP runtime.

Upstream: https://github.com/ggml-org/llama.cpp (MIT License).

## NVIDIA CUDA runtime (`runtime/`)

`cublas64_13.dll`, `cublasLt64_13.dll` and `cudart64_13.dll` are parts of the
CUDA Toolkit redistributable, used under the NVIDIA CUDA Toolkit End User License
Agreement. They are bundled here only so the backend runs on a host without a
system CUDA installation.

## Model

`Qwen3.6-35B-A3B-UD-IQ4_XS.gguf` is quantised by Unsloth from Qwen's
`Qwen3.6-35B-A3B` model (Apache-2.0). The model bytes are not included in this
package; the frozen `inputs.json` record holds their exact identity.

