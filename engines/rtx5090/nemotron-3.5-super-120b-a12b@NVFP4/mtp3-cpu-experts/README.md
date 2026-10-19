# Nemotron-3.5-Super-120B-A12B @ NVFP4 · RTX 5090 · mtp3-cpu-experts

> [!NOTE]
> **Contributed by** crit-labs · [crit-labs/crit](https://github.com/crit-labs/crit)

[crit-hub](../../../../README.md) › [engines](../../../README.md) › [rtx5090](../../README.md) › [nemotron-3.5-super-120b-a12b@NVFP4](../README.md) › mtp3-cpu-experts

An engine written from scratch on crit Core 0.2.1 for one RTX 5090 (32 GB, sm_120a) on WSL2 Ubuntu 24.04 with
CUDA 13.1, a Ryzen 5 9600 and about 82 GiB of host RAM visible to WSL. Text path only; context 262,144 tokens.

Model: `nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4` at revision
`59b8721646c38111a242da18218859fb2b454322` (nemotron_h: 40 Mamba-2, 40 latent-MoE layers with 512 NVFP4 experts
top-22 plus a shared expert, 8 GQA attention layers without rotary embedding, an MTP block). The checkpoint is
about 75 GB, 63 GB of it routed experts, so it cannot sit on the GPU. Weights are used as stored.
[model.json](model.json) lists every file with the size and SHA-256 the run recorded.

## Design

| Part | Where it lives |
| --- | --- |
| Mamba, attention and head projections (FP8 per tensor), router and latent projections (BF16), shared experts (NVFP4) | GPU, resident (7.2 GB) |
| Routed experts (NVFP4, 20,480 records of 3.1 MB) | Pinned host RAM, 61 GiB |
| Hot experts | GPU CLOCK slot cache (about 7,150 slots) over free VRAM, the idle prompt workspace and KV positions a request can't reach |
| KV cache | GPU, FP8 E4M3 with a scale per position and KV head, 262,144 tokens |
| Mamba state | GPU, FP32, with snapshots per speculative round position |

Greedy decoding runs speculative rounds with the checkpoint's MTP block as released (3 drafts, full head; about
3.5 tokens per round), one CUDA graph per round. Expert misses are the bottleneck: 60% of each pass's misses are
computed by 6 AVX-512 CPU threads from the pinned records with the GPU kernels' exact operation order (output does
not depend on where a miss runs); the rest are read over the host link and written through to a slot.

Prompts run in 32,768-token chunks: routed and shared experts as NVFP4 W4A4 on CUTLASS SM120 block-scaled grouped
GEMMs (the checkpoint's calibrated input scales), FP8 weights on cuBLASLt FP8, attention on E4M3 tensor cores, the
next layer's experts prefetched for chunks of 256+ tokens, and for chunks up to 384 tokens 35% of the experts
computed on the CPU instead of copied. The Mamba dt lower limit follows config.json ([0, inf], as NVIDIA's vLLM
code applies it); transformers' 0.001 clamp breaks retrieval beyond 64K tokens.

## Measured (crit Eval, crit-eval-v4, 262,144-token capacity)

| Metric | Result |
| --- | ---: |
| Decode score, greedy MTP (6 speed cases) | 42.5-44.6 tok/s |
| Prefill score | 3,035-3,152 tok/s |
| Time to first token, geometric mean | 1.78-1.83 s |
| Long-context decode, 32K-262K | 41.1 tok/s |
| Long-context first token, mean over 32K-262K / at 262K | 10.5-10.8 s / 37-38 s |
| Quality vs FP32 reference (worst path: prefill KLD mean) | 5.1e-3 (budget 0.05) |

Ranges are the two Eval runs. Run benchmarks (median of 3, 256 output tokens): 512-token prompt 38.8 tok/s decode,
2.13 s first token; 14,336-token prompt 41.6 tok/s, 2.49 s, 5,765 tok/s prefill. Every Eval gate passes (coverage,
repeatable, request history, quality, capacity, retrieval at 4K-262K, lower context). Peak use left 0.41 GB of VRAM
and 18.9 GB of host RAM free. Decode depends on the CPU: other heavy programs on the PC slow it.

## Limits

- Sampled requests (the chat default: temperature 1.0, top_p 0.95) decode one token per pass, about 22 tok/s;
  speculative rounds are greedy only.
- Short prompts are bound by copying experts over the host link (about 1.3 s).
- Speculative output is not bitwise equal to plain greedy (reported, not required).
- Needs about 70 GB free in WSL while running; loading takes 1-2 minutes.

## Build and chat

```sh
cmake -S engine -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=<crit Core 0.2.1 install>
cmake --build build
bash engine/package.sh engine build <package dir> <crit Core install>   # libnmh.so + adapters
```

Register the package as a crit run candidate, or serve the sealed candidate from the original run:

```sh
crit serve --workspace <runs root> --run run-20261018T173355Z-f764c819 --candidate candidate-0003 --project <crit checkout>
```

[engine/configuration.json](engine/configuration.json) has the applied settings; the original run keeps the full
records, Eval results and summary.
