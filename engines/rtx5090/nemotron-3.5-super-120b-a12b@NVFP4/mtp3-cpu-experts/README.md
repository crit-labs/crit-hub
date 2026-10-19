# Nemotron-3.5-Super-120B-A12B @ NVFP4 · RTX 5090 · mtp3-cpu-experts

> [!NOTE]
> **Contributed by** crit-labs · [crit-labs/crit](https://github.com/crit-labs/crit)

An engine written from scratch on crit Core 0.2.1, which it links as an installed SDK. It runs text
generation on WSL2 Ubuntu 24.04 with CUDA 13.1 on sm_120a (RTX 5090, 32 GB), a Ryzen 5 9600 and about
82 GiB of host RAM visible to WSL, at a 262,144-token context.

## Model

`nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4` at revision
`59b8721646c38111a242da18218859fb2b454322` (nemotron_h): 40 Mamba-2 layers, 40 latent-MoE layers
with 512 NVFP4 experts top-22 plus a shared expert, 8 GQA attention layers without rotary embedding,
and an MTP block. The checkpoint is about 75 GB, 63 GB of it routed experts, so it cannot sit on the
GPU. The weights are used as stored; the vision path is not used. [model.json](model.json) lists
every checkpoint file with the size and SHA-256 the run recorded; `python -B tools/verify_model.py`
checks a download against it.

## Design

| Part | How |
| --- | --- |
| Resident weights | Mamba, attention and head projections (FP8 per tensor), router and latent projections (BF16) and shared experts (NVFP4) on the GPU, 7.2 GB |
| Routed experts | NVFP4, 20,480 records of 3.1 MB in pinned host RAM, 61 GiB |
| Hot experts | GPU CLOCK slot cache (about 7,150 slots) over free VRAM, the idle prompt workspace and KV positions a request can't reach |
| Expert misses | 60% of each pass's misses computed by 6 AVX-512 CPU threads from the pinned records in the GPU kernels' exact operation order (output does not depend on where a miss runs); the rest read over the host link and written through to a slot |
| KV cache and Mamba state | FP8 E4M3 KV with a scale per position and KV head, 262,144 tokens; FP32 Mamba state with snapshots per speculative round position |
| Speculative decoding | The checkpoint's MTP block as released: 3 drafts, full head, about 3.5 tokens per round, one CUDA graph per round (greedy only) |
| Prompts | 32,768-token chunks; routed and shared experts as NVFP4 W4A4 on CUTLASS SM120 block-scaled grouped GEMMs (the checkpoint's calibrated input scales), FP8 weights on cuBLASLt FP8, attention on E4M3 tensor cores, the next layer's experts prefetched for chunks of 256+ tokens, and for chunks up to 384 tokens 35% of the experts computed on the CPU instead of copied |
| Chat | crit Core's stock native adapter (ResidentChat): the chat template with tools and tool history, tool calls in the template's format, per-request temperature, top_p, top_k and seed, streaming |

The Mamba dt lower limit follows config.json ([0, inf], as NVIDIA's vLLM code applies it);
transformers' 0.001 clamp breaks retrieval beyond 64K tokens.

## Measured

crit Eval (crit-eval-v4) of sealed candidate-0003, 6 speed cases at a 262,144-token capacity; ranges
are two Eval runs:

| Metric | Result |
| --- | ---: |
| Decode, greedy MTP | 42.5-44.6 tok/s |
| Prefill | 3,035-3,152 tok/s |
| Time to first token (geomean) | 1.78-1.83 s |
| Long-context decode (32K to 262K) | 41.1 tok/s |
| Long-context first token (mean; 37-38 s at 262K) | 10.5-10.8 s |
| Quality, worst path (prefill KLD mean) | 5.1e-3 (budget 0.05) |

| Run benchmark, 256 output tokens | Decode | Time to first token | Prefill |
| --- | ---: | ---: | ---: |
| 512-token prompt | 38.8 tok/s | 2.13 s | not recorded |
| 14,336-token prompt | 41.6 tok/s | 2.49 s | 5,765 tok/s |

Run benchmarks are medians of three repetitions, greedy, concurrency one, resident. Expert misses set
the speed, so decode depends on the CPU: other heavy programs on the PC slow it.

## Checks

- **Arithmetic:** Eval quality within budget on every path against an FP32 transformers reference
  (with the config's Mamba dt limit) over the stored checkpoint.
- **Functional:** every Eval gate passed: coverage, repeatability, request history, quality,
  capacity, retrieval at 4K to 262K and lower context.
- **Resources:** peak use left 0.41 GB of VRAM and 18.9 GB of host RAM free.

## Limits

- Text only.
- Sampled requests (the chat default: temperature 1.0, top_p 0.95) decode one token per pass, about
  22 tok/s; speculative rounds are greedy only.
- Short prompts are bound by copying experts over the host link (about 1.3 s).
- Speculative output is not bitwise equal to plain greedy (reported, not required).
- Needs about 70 GB free in WSL while running; loading takes 1-2 minutes.

## Build and serve

```sh
git clone https://github.com/crit-labs/crit && git -C crit checkout v0.2.1
cmake -S crit/core -B build/core -G Ninja -DCMAKE_BUILD_TYPE=Release -DCRIT_CORE_CUDA=ON -DBUILD_TESTING=OFF \
      -DCMAKE_CUDA_ARCHITECTURES=120a -DCMAKE_POSITION_INDEPENDENT_CODE=ON
cmake --build build/core && cmake --install build/core --prefix core-install
bash engine/package.sh engine build/nmh package core-install
```

Builds need CMake 3.28+, Ninja, GCC 13 and the CUDA 13.1 toolkit; `libcudart` must be on the loader
path. `package.sh` builds `libnmh.so` and copies it with the adapters and the CUTLASS license into the
package. Register the package as a crit run candidate, or serve the sealed candidate from the
original run:

```sh
crit serve --workspace <runs root> --run run-20261018T173355Z-f764c819 --candidate candidate-0003 --project <crit checkout>
```

[engine/configuration.json](engine/configuration.json) has the applied settings; the original run
keeps the full records, Eval results and summary.

## Credits

Credits and licenses are in [THIRD_PARTY.md](THIRD_PARTY.md) and [LICENSES.txt](LICENSES.txt).
