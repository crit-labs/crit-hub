# Qwen3.8-Flash-Next @ FP4-Local · RTX 5090 · mtp3-expert-cache

An engine written from scratch on crit Core 0.1.0, which it links as an installed SDK. It
runs text generation on WSL2 Ubuntu 24.04 with CUDA 13.1 on sm_120a (RTX 5090, 32 GB)
and a host with about 82 GiB of RAM available to WSL.

Model: `nvidia/Qwen3.8-Flash-Next-FP4-Local` at revision
`b9ce18a6887e18c089fdea8509a33e63429a3300` (qwen4_exp text path: 48 layers, 36 gated
delta-net and 12 QSA attention, 4-stream hyper-connections, a PLE n-gram embedding, 512
NVFP4 experts top-10 with CSF-encoded scales plus a shared expert, and an MTP block). The
checkpoint is about 102 GB, so it cannot sit on the GPU. Weights are used as stored.
[model.json](model.json) lists every checkpoint file with the size and SHA-256 the run
recorded; `python -B tools/verify_model.py` checks a download against it.

## Design

| Part | Where it lives |
| --- | --- |
| Attention, delta-net and shared-expert projections (MXFP8); router, hyper-connections, head (BF16) | GPU, resident |
| Routed experts (NVFP4, CSF scales kept compressed) | Pinned host RAM, about 61 GB |
| Hot experts | GPU CLOCK slot cache (about 7,800 slots), which also borrows KV positions a request can't reach and the idle prompt workspace |
| KV cache | GPU, FP8 E4M3 with an FP32 scale per 256-dim vector, sized for 262,144 tokens |
| Token embeddings | Pinned host RAM, read through the device mapping |
| PLE n-gram table | Memory-mapped checkpoint, rows gathered on the host per token |

Greedy decoding runs speculative rounds with the checkpoint's MTP block, used as released:
each round drafts 3 tokens, verifies them in one 4-token pass, accepts, commits state and
drafts again, all on the device in CUDA graphs. The drafting head is re-encoded to MXFP8;
the verify pass uses the stored BF16 head. An expert miss is read straight from pinned
host memory and written through to its cache slot. Prompts run in 16K-token chunks of
FP16 tensor-core GEMMs with one batched expert copy per layer. Sampled requests decode one
token per pass with crit Core's sampler. [engine/README.md](engine/README.md) has the
full design notes.

## Measured (sealed candidate-0003)

crit Eval, 6 speed cases at a 262,144-token context:

| Metric | Result |
| --- | ---: |
| Decode | 88.3 tok/s |
| Prefill | 1,818 tok/s |
| Time to first token (geomean) | 1.60 s |
| Long-context decode (to 262K) | 69.6 tok/s |
| Held-out decode (10 prompts, 8 languages and code) | 94.3 tok/s |
| Quality, worst path | 2.7% of the budget (PPL ×1.05, KLD mean 0.05, p99 0.5) |

| Run benchmark, 256 output tokens | Decode | Time to first token | Prefill |
| --- | ---: | ---: | ---: |
| 512-token prompt | 84.1 tok/s | 2.32 s | 221 tok/s |
| 14,336-token prompt | 77.9 tok/s | 5.17 s | 2,777 tok/s |

Medians of three measured repetitions after one warmup, greedy decoding, stop tokens
disabled, concurrency one, resident. At least 1.01 GiB of VRAM (nvidia-smi, whole GPU)
and 18.6 GiB of RAM stayed free. Loading takes about 2.5 minutes.

Decode is bound by expert misses over the host link (about 27 GB/s here): the 4-token
verify pass reads the union of four tokens' experts, and each round misses 88 (chat) to
280 (after a 14K prompt) experts.

## Checks

- **Arithmetic:** Eval quality within budget on the prefill, decode and verify paths,
  against an FP32 reference written from the Hugging Face transformers `qwen4_exp`
  definition.
- **Functional:** all 6 Eval speed cases ran at 262,144 tokens; the capacity and
  lower-context gates passed; a fresh process repeats every greedy output; the resident
  adapter passed consecutive requests, streaming, mid-stream cancel and close/reopen.
  `crit serve` answered consecutive requests, a leading system message, streaming, a
  stop string and a switch between candidates, and stopped cleanly.
- **Resources:** the free-memory figures above, in every case.

## Limits

- Text only; the vision tower is not used.
- Greedy output repeats in a fresh process but not between requests in one process: the
  512-token case diverged at token 24 on its second request. Speculative output equals
  plain greedy decoding in 3 of 6 Eval cases (FP8 KV cache); exactness is reported, not
  required.
- Prefill runs at about 5% of its compute ceiling, since every chunk streams experts over
  the host link.
- Needs about 82 GiB of RAM for WSL, an otherwise idle GPU and `vram_margin_gib` 1.0.

## Build and chat

```sh
git clone https://github.com/crit-labs/crit && git -C crit checkout v0.1.0
cmake -S crit/core -B build/core -G Ninja -DCMAKE_BUILD_TYPE=Release -DCRIT_CORE_CUDA=ON \
      -DCMAKE_CUDA_ARCHITECTURES=120a -DCMAKE_POSITION_INDEPENDENT_CODE=ON
cmake --build build/core && cmake --install build/core --prefix core-install
cmake -S engine -B build/fnl -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=$PWD/core-install
cmake --build build/fnl --target fnl
mkdir -p package && cp build/fnl/libfnl.so engine/python/*.py engine/configuration.json package/
```

`engine/python/fnl_serve.py` exports `create_engine(context)` for crit's resident chat
(`chat-json-v1`, `chat-resident-v1`, `chat-stream-v1`, `chat-config-v1`,
`chat-model-file-v1`); `engine/python/fnl_eval_adapter.py` is its crit Eval adapter. Open
the sealed run with
`crit serve --workspace <runs folder> --run run-20261016T225118Z-8d49a6ce --candidate candidate-0003`.
Serving needs crit 0.2.0 or newer: 0.1.0 bounds loading at a fixed 600 s, and this
model takes about 8 minutes (crit first copies and hashes the 102 GB checkpoint).
