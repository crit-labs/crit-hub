# Notices

Clean-room Qwen3.8-27B NVFP4 engine. Engine source (kernels, loader, Core adapter,
chat adapter, benchmark harness) was written from scratch for this run.

| Component | Use | License |
| --- | --- | --- |
| crit Core (`core/`, built from the crit checkout) | Engine/Session runtime, CUDA backend | crit LICENSE |
| crit Forge `device-speculative-round` (`forge/device_round.*`, `forge/greedy_accept.*`, unmodified copies) | Device greedy accept and round graph controller | crit LICENSE |
| CUDA 13.1 runtime and cuBLAS (dynamically linked, not redistributed) | Runtime; BF16 GEMM for prefill after exact dequantization | NVIDIA CUDA EULA |
| tokenizers 0.22.2, Jinja2 3.1.6, MarkupSafe 3.0.3 (`vendor/`, unmodified PyPI wheel contents, RECORD-verified) | Original tokenizer.json and chat template | Apache-2.0 / BSD-3-Clause |
| Model weights `unsloth/Qwen3.8-27B-NVFP4` @ f0b7c9e722f5565102fff8481c99e4d86ae099c7 (external, not redistributed) | Read from the original Safetensors; no uploaded word changed | Apache-2.0 |
