# Qwen3.8-27B @ NVFP4 · RTX 5090 · mtp7-fp8attn

> [!NOTE]
> **Contributed by** crit-labs · [crit-labs/crit](https://github.com/crit-labs/crit)
>
> Built with [crit](https://github.com/crit-labs/crit): a sealed crit candidate, exported with `tools/export_engine.py`.

An engine written from scratch on crit Core 0.1.0, which it links as an installed SDK. It runs text
generation on WSL2 Ubuntu 24.04 with CUDA 13.1 on sm_120a (RTX 5090, 32 GB) at a 262,144-token
context.

## Model

`unsloth/Qwen3.8-27B-NVFP4` at revision `f0b7c9e722f5565102fff8481c99e4d86ae099c7`: 48 gated DeltaNet
and 16 gated full-attention layers, NVFP4 MLPs in layers 0–55, FP8 per-channel everywhere else
including the head, and a bundled BF16 MTP layer. The weights are used as stored; the vision tower is
not used. [model.json](model.json) lists every checkpoint file with the size and SHA-256 the run
recorded; `python -B tools/verify_model.py` checks a download against it.

## Design

| Part | How |
| --- | --- |
| Decode GEMVs | Exact FP8 / E2M1 → FP16 tensor-core GEMVs (mma.m16n8k16) for 1–8 columns, residual + RMSNorm + quantization fused into the epilogues |
| DeltaNet decode | One kernel per head and step, state commit deferred until the round's accepted length is known |
| KV cache | FP8 E4M3 with one FP32 scale per position and KV head, sized for 262,144 tokens (crit `fp8-kv-attention`) |
| Speculative decoding | The bundled MTP layer, used as released (re-encoded to NVFP4 for drafting): catch-up, up to 7 drafts, an 8-column verify and accept in one CUDA graph per round, with the draft width chosen per round from the request's acceptance |
| Prompt GEMMs | cuBLASLt FP8 and NVFP4 with an algorithm table recorded once on this GPU (`gemm_table.txt`); NVFP4 activations use the checkpoint's static input scale |
| Prompt attention | Q·Kᵀ and P·V on E4M3 tensor cores over the cached codes, 64-key double-buffered tiles |
| Prompt DeltaNet | Tiled causal conv and a staged sequential gated-delta scan |
| Chat | The model card's thinking preset (temperature 1.0, top-p 0.95, top-k 20) by default, per-request temperature, top-p, top-k (up to 64) and seed, and tool calls in the chat template's format; drafts accepted with probability p(d) under that distribution, so sampling stays exact |

## Measured

crit Eval of sealed candidate-0007, 6 speed cases at a 262,144-token context, decode on pinned
continuations:

| Metric | Result |
| --- | ---: |
| Decode | 215.7 tok/s |
| Prefill | 8,917 tok/s |
| Time to first token (geomean) | 120 ms |
| Long-context decode (32K to 262K) | 273.3 tok/s |
| Long-context first token (geomean; 69.0 s at 262K) | 15.9 s |
| Held-out decode (10 prompts, 8 languages and code) | 151.1 tok/s |
| Quality, worst path | 18.8% of the budget (PPL ×1.05, KLD mean 0.05, p99 0.5) |

| Run benchmark, 256 output tokens | Decode | Time to first token | Prefill |
| --- | ---: | ---: | ---: |
| 512-token prompt | 164.8 tok/s | 65 ms | 7,903 tok/s |
| 14,336-token prompt | 147.8 tok/s | 1.68 s | 8,513 tok/s |

Decode by case: chat 150, code 203, math 259, tool call 204, long summary 178, agent 354 tok/s. Run
benchmarks are medians of three repetitions after one warmup, greedy, concurrency one, resident. The
8-column verify pass reads the weights at 96% of the bandwidth floor, so decode speed is set by how
many drafts get accepted: about 2.6 tokens per round on chat and 7.8 on the agent case.

## Checks

- **Arithmetic:** Eval quality within budget on the prefill, decode and verify paths, against an
  FP32 transformers reference over the dequantized checkpoint.
- **Functional:** all 6 Eval speed cases ran at 262,144 tokens; coverage, repeatability, capacity,
  lower-context and held-out generality gates passed; speculative output equals plain greedy
  decoding on 6 of 6 cases; a fresh process repeats every greedy output. `crit serve` passed
  consecutive requests, a system message, streaming, a stop string, switching candidates and Ctrl+C,
  a tool call and the answer after its result, a streamed tool call, and seeded sampling that
  repeats with the same seed.
- **Resources:** at least 0.77 GiB of VRAM (whole GPU) and 77 GiB of RAM stayed free in every case.

## Limits

- Text only.
- Needs the whole 32 GB GPU: under 1 GiB stays free at the full context.
- Prefill runs at 66% of its compute ceiling; the sequential DeltaNet scan and prompt epilogues take
  most of the rest.
- Decode on short chat turns is lower (about 150 tok/s) because the MTP layer's acceptance is lower
  there.
- The recorded GEMM table is for the RTX 5090; on other GPUs delete `gemm_table.txt` from the
  package (cuBLASLt's first heuristic is used) or re-measure it.

## Build and chat

```sh
git clone https://github.com/crit-labs/crit && git -C crit checkout v0.1.0
cmake -S crit/core -B build/core -G Ninja -DCMAKE_BUILD_TYPE=Release -DCRIT_CORE_CUDA=ON -DBUILD_TESTING=OFF \
      -DCMAKE_CUDA_ARCHITECTURES=120a -DCMAKE_POSITION_INDEPENDENT_CODE=ON
cmake --build build/core && cmake --install build/core --prefix core-install
bash engine/build.sh engine build/kestrel package core-install
cp engine/configuration.json package/
```

Builds need CMake 3.28+, Ninja, GCC 13 and the CUDA 13.1 toolkit (`NVCC` overrides the compiler
path); `libcudart` and `libcublasLt` must be on the loader path. Loading takes about 13 s. Register
the package as a crit run candidate, or serve the sealed candidate from the original run:

```sh
crit serve --workspace <runs root> --run run-20261017T045435Z-b4d9b011 --candidate candidate-0007 --project <crit checkout>
```

Or chat in the terminal (Python 3.12 with `tokenizers`, `jinja2` and crit's Python packages);
`/new` starts a fresh conversation, Ctrl+C stops a reply, `/quit` releases the engine:

```sh
PYTHONPATH=crit/core/python python engine/python/chat.py --model /path/to/Qwen3.8-27B-NVFP4 --package package
```

[engine/configuration.json](engine/configuration.json) has the applied settings; `chat.py` was added
at export and is not part of the sealed source.

## Credits

Credits and licenses are in [THIRD_PARTY.md](THIRD_PARTY.md) and [LICENSES.txt](LICENSES.txt).
