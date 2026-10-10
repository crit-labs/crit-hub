# Third-party components

This package drives a prebuilt, unmodified upstream binary. No local compiler
is used and no upstream source is vendored into this repository.

## llama.cpp

- Project: llama.cpp (https://github.com/ggml-org/llama.cpp)
- Build: b11463, commit d2a79e604
- Artifact: `llama-b11463-bin-win-cuda-13.4-x64.zip` from the project's GitHub
  releases, extracted unchanged into `vendor/llamacpp/b11463/bin`.
- Runtime: `cudart-llama-bin-win-cuda-13.4-x64.zip`, extracted unchanged into
  `vendor/llamacpp/b11463/runtime`.
- License: MIT. The upstream `LICENSE` is redistributed with the extracted
  binary as `LICENSE-LLVM-OpenMP` and in the release archive.
- CUDA runtime components (`cudart64_13`, `cublas64_13`, `cublasLt64_13`) are
  NVIDIA redistributables bundled with the upstream release. They remain
  subject to the NVIDIA CUDA Toolkit End User License Agreement.

## Model weights

Model weights are not redistributed here. The frozen run records their exact
path, byte count and SHA-256 identity at resolve time, and the engine refuses to
start against bytes that do not match.

- Model: Gemma 4 26B-A4B instruction-tuned, GGUF `UD-IQ4_XS` quantisation
- Repository: `unsloth/gemma-4-26B-A4B-it-GGUF`
- File: `gemma-4-26B-A4B-it-UD-IQ4_XS.gguf`
- License: Gemma Terms of Use, as published with the model.

## Optional OpenMP runtime

`libomp.dll` and its `LICENSE-LLVM-OpenMP` notice are redistributed as part of
the upstream prebuilt release under the Apache License 2.0 with LLVM exceptions.
