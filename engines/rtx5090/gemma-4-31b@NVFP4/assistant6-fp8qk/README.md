# Gemma-4-31B @ NVFP4 · RTX 5090 · assistant6-fp8qk

> [!NOTE]
> **Contributed by** crit-labs · [crit-labs/crit](https://github.com/crit-labs/crit)

A native CUDA engine on crit Core 0.2.1, which it links as an installed SDK. It runs text generation
on WSL2 Ubuntu 24.04 with CUDA 13.1 on sm_120a (RTX 5090, 32 GB) at a 262,144-token context, the
model's native maximum.

## Model

`nvidia/Gemma-4-31B-IT-NVFP4` at revision `4135a98a9b728a548947683219633b25682223ac`: 60 layers, 50
sliding-window layers with 16 KV heads of 256 and a 1,024-token window and 10 full-attention layers
with 4 KV heads of 512 whose keys equal values; NVFP4 MLPs, BF16 attention and tied output head. The
NVFP4 MLPs are used as stored; the BF16 attention projections and output head are re-encoded to FP8
E4M3 per row at load (deterministic, no calibration data), because as stored they do not fit next to
a 262K KV cache. The vision tower is not used. [model.json](model.json) lists every checkpoint file
with the size and SHA-256 the run recorded; `python -B tools/verify_model.py` checks a download
against it.

Drafter: the stock `google/gemma-4-31B-it-assistant` at revision
`627c5ec1458b9086b841a91e0512fd31fd2fbbf1`, used as released (re-encoded to NVFP4 for drafting, never
trained). Download it separately and set `draft_model` in `engine/configuration.json` to its folder
(the value there is the run machine's path).

## Design

| Part | How |
| --- | --- |
| Decode GEMVs | FP8 / E2M1 weights converted exactly to BF16 in registers, BF16 activations (Gemma's hidden states overflow FP16) on mma.m16n8k16 for 1-7 columns; sandwich norms, layer scalars, GeGLU and soft-capped logits in fused epilogues |
| Speculative decoding | 6 drafter steps and an up-to-7-column target verify in one CUDA graph per round; greedy acceptance, or for sampled requests drafts accepted with probability p(d) under the request's distribution, so sampling stays exact |
| KV cache | FP8 E4M3 with one FP32 scale per position and KV head: a 3,200-slot ring for the sliding layers; for the full layers u = v_norm(k) plus the 128 rotated key dims as their own E4M3 codes |
| Prompt GEMMs | 2,048-token chunks; attention projections on cuBLASLt FP8 and MLPs on cuBLASLt NVFP4, both with per-row activation scales |
| Prompt attention | Sliding layers: BF16 tensor-core flash attention over the FP8 ring. Full layers: Q·Kᵀ on FP8 tensor cores over the stored key codes, P·V in BF16 |
| Chat | crit Core's ResidentChat with Gemma 4's chat template and per-request temperature, top_p, top_k (up to 64) and seed (default preset temperature 1.0, top_p 0.95, top_k 64); tool calls parsed by the package's own Gemma syntax (`python/gemma_tools.py` as `ResidentChat.tool_format`, `chat-tools-v1`, streamed as tool-call deltas) |

## Measured

crit Eval of sealed candidate-0005, 6 speed cases at a 262,144-token context, decode on pinned
continuations:

| Metric | Result |
| --- | ---: |
| Decode | 260.9 tok/s |
| Prefill | 8,326 tok/s |
| Time to first token (geomean) | 116 ms |
| Long-context decode (32K to 262K) | 226.1 tok/s |
| Long-context first token (mean; about 148 s at 262K) | 26.4 s |
| Held-out decode (prompts outside the speed cases) | 180.8 tok/s |
| Quality, worst path (prefill: PPL ×1.046, KLD mean 0.025, p99 0.36) | 91.4% of the budget (PPL ×1.05, KLD mean 0.05, p99 0.5) |

| Run benchmark, 256 output tokens | Decode | Time to first token | Prefill |
| --- | ---: | ---: | ---: |
| 512-token prompt | 234.5 tok/s | 59 ms | not recorded |
| 14,336-token prompt | 233.2 tok/s | 1.70 s | not recorded |

Decode by case: chat 176, code 276, math 302, tool call 341, long summary 219, agent 328 tok/s. Run
benchmarks are medians of three repetitions after one warmup, greedy, concurrency one, resident.
Candidate-0004 runs the same default code path and measured 266.6 tok/s decode and 8,471 tok/s
prefill; the difference is run-to-run variation. The verify pass reads the weights at its bandwidth
floor (14.56 vs 15.02 ms), so decode speed is set by how many drafts get accepted: 3.05 tokens per
round on chat up to 6.4 on the agent case.

## Checks

- **Arithmetic:** Eval quality within budget on the prefill, decode, verify and prefill-then-decode
  paths against an FP32 transformers reference over the stored checkpoint.
- **Functional:** all 6 Eval speed cases ran at 262,144 tokens; coverage, repeatability, request
  history, capacity, retrieval (4K to 262K), lower-context and held-out generality gates passed; a
  fresh process repeats every greedy output. `crit serve` passed consecutive requests, a system
  message, streaming, a stop string, a tool call and its result, a streamed tool call and Ctrl+C.
  Switching was shown with candidate-0004 (same engine, before crit dropped its built-in Gemma
  parser).
- **Resources:** at least 0.65 GB of VRAM (whole GPU) and 83 GB of host RAM stayed free in every case.

## Limits

- Text only.
- Needs the whole 32 GB GPU: about 0.6 GB stays free at the full context.
- Speculative output is not bitwise equal to plain greedy decoding (reported, not required).
- The prompt path spends most of the quality budget (W4A4 MLP activations and FP8 Q·Kᵀ); decode
  drift is 7%.
- Long prompts are bound by the quadratic full attention; P·V is still BF16.
- top_k above 64 is refused.

## Build and chat

```sh
git clone https://github.com/crit-labs/crit && git -C crit checkout v0.2.1
cmake -S crit/core -B build/core -G Ninja -DCMAKE_BUILD_TYPE=Release -DCRIT_CORE_CUDA=ON -DBUILD_TESTING=OFF \
      -DCMAKE_CUDA_ARCHITECTURES=120a -DCMAKE_POSITION_INDEPENDENT_CODE=ON
cmake --build build/core && cmake --install build/core --prefix core-install
bash engine/build.sh engine build/g4 package core-install
cp engine/configuration.json package/
```

Builds need CMake 3.28+, Ninja, GCC 13 and the CUDA 13.1 toolkit (`NVCC` overrides the compiler
path); `libcudart` and `libcublasLt` must be on the loader path. The run built against a crit
checkout with changes after v0.2.1 that this package needs (Gemma's thought channel in Eval prepare,
special-token tool markers in ResidentChat and `ResidentChat.tool_format`); use a crit version that
includes them. Register the package as a crit run candidate, or serve the sealed candidate from the
original run:

```sh
crit serve --workspace <runs root> --run run-20261018T211706Z-f7e30ecd --candidate candidate-0005 --project <crit checkout>
```

[engine/configuration.json](engine/configuration.json) has the applied settings; the original run
keeps the full records, Eval results and summary.

## Credits

Credits and licenses are in [THIRD_PARTY.md](THIRD_PARTY.md) and [LICENSES.txt](LICENSES.txt).
