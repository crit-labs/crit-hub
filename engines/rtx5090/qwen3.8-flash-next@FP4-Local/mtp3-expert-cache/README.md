# Qwen3.8-Flash-Next @ FP4-Local · RTX 5090 · mtp3-expert-cache

> [!NOTE]
> **Contributed by** crit-labs · [crit-labs/crit](https://github.com/crit-labs/crit)

[crit-hub](../../../../README.md) › [engines](../../../README.md) › [rtx5090](../../README.md) › [qwen3.8-flash-next@FP4-Local](../README.md) › mtp3-expert-cache

**Release source: candidate-0010.** It is candidate-0006 with a serve adapter built on
crit's shared `ResidentChat`: tool calls (tools and tool history go through the model's chat
template, and the adapter parses the calls), per-request `temperature`, `top_p`, `top_k` and
`seed`, and decode telemetry that counts tokens. The native library, Eval adapter and configuration are byte-identical to
candidate-0006, which repaired candidate-0003 (incorrect after long prompts) by initializing
the scales of every prefetched expert before that cache entry can be used, and reserves 2 GiB
at cache allocation for runtime graphs and desktop GPU usage. [Validation](engine/validation.json)
records tested scope and identities.

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
| Hot experts | GPU CLOCK slot cache (sized from free VRAM with a 2 GiB reserve), which also borrows KV positions a request can't reach and the idle prompt workspace |
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

## Measured

crit Eval of the byte-identical candidate-0006 engine, six speed cases, 262,144-token
capacity, greedy MTP decoding:

| Metric | Result |
| --- | ---: |
| Decode score | 84.5 tok/s |
| Prefill score | 1,801 tok/s |
| Time to first token, geometric mean | 1.61 s |
| Decode at 261,888 prompt + 256 output tokens | 63.1 tok/s |
| First token at that full capacity | 99.8 s |
| Held-out decode, ten prompts | 92.1 tok/s |

| Run benchmark: prompt tokens, 256 output | Decode tok/s | First token | Prefill tok/s |
| --- | ---: | ---: | ---: |
| 512 | 91.7 | 2.19 s | 234 |
| 14,336 | 83.3 | 4.81 s | 2,978 |

Run benchmarks are candidate-0010's own measurement; candidate-0006 measured 84.9 and 74.6
tok/s decode on the same binary in an earlier session.

Run measurements are medians of three repetitions after one warmup, concurrency one,
resident, with stop tokens disabled. The adapter has no replay support: measurements
use its own generated text. They are not a controlled same-output speedup comparison
against the corrupt original. Full-Eval physical telemetry left at least
0.85 GiB GPU memory and 17.3 GiB host RAM available.
Loading the checkpoint takes several minutes. The tested API startup, including model
preparation and verification, took 880 seconds. For a registered
run, use `crit serve --timeout 1800` to leave startup margin; the test passed with 900.

## Correctness and practical checks

- The original sealed binary fails mixed prefill-to-decode reference checks. At a
  4,094-token prompt, over 38 fixed continuation positions, its PPL ratio was
  1.17705, mean KL 0.15651 and p99 KL 3.70461.
  The repaired native binary reaches 1.00695, 0.000442 and 0.01048 on that check,
  within the unchanged limits 1.05 / 0.05 / 0.5. Graph and eager execution agree.
- The independent reference uses official Transformers 5.19.0 `qwen4_exp` layers and
  cache with exact FP32 dequantization of this stored checkpoint. Weight residency and
  query batching were adapted to fit the GPU. It correctly generated all three archive
  identifiers on the reproduced 4K failure. This is not a stock full-model deployment
  or a reference validation at 262K.
- This configuration passes all Eval gates, six fresh-process repeatability cases and
  nineteen additional generation checks using prompts outside Eval. Greedy and seeded sampled retrieval succeeds
  at 1K, 4K, 8K, 32K, 131K and 259,874 input tokens. Repeated 4K/32K requests after the
  largest request retain the same token sequences. German, French and code copying pass.
- A fresh build against the published Core SDK and the standalone launcher pass a long
  prompt-file test. Python dependencies install in a fresh virtual environment.
- Released crit 0.2.0 serves consecutive requests, leading system messages, streaming,
  a stop string, near-260K retrieval and candidate switches, and exits cleanly on Ctrl+C.
  The retained registered run was used for this API test.
- Through `crit serve` (crit main) on candidate-0010: a tool call, the
  answer after its tool result, a streamed call and a plain reply with tools offered;
  greedy `temperature: 0`, the same `seed` repeating its sampled text and another seed
  differing; decode telemetry within 20% of the wall-clock rate; SIGTERM unloading cleanly.
  Requests render to the same tokens as in candidate-0006.
- Eval gates were run under crit-eval-v3, plus the targeted prefill-then-decode,
  retrieval and request-history checks above. A crit-eval-v4 run, which adds the
  prefill_decode quality path, the retrieval gate and the request-history gate to
  Eval itself, is pending.

## Known issues

- Speculative output equals plain greedy in 3/6 Eval cases (tool-call, long-summary
  and agent-long first differ at tokens 125, 19 and 146). The cause is not yet
  established; until it is, treat it as an open bug, not rounding.

## Limits

- **262,144 means total templated prompt plus generated output.** It is a tested capacity,
  not a promise of accurate answers to every 262K document. Near-capacity retrieval was
  tested separately from the full-capacity synthetic benchmark.
- Text only, one request at a time. WSL2 Ubuntu 24.04, RTX 5090 32GB, CUDA 13.1,
  g++ 13.3, CMake 3.28 and Ninja were tested, with about 82 GiB available to WSL.
  Other GPUs, native Windows and a clean OS installation are untested.
- Default chat sampling is temperature 1, top-p .95, top-k 20, with thinking disabled.
  The speed measurements use greedy MTP. Sampled requests use one-token decoding and
  should not be advertised at the greedy benchmark rate.
- Requests to the same candidate keep weights loaded. Candidate switching reloads
  weights; tested switches took 211–248 seconds. Model allocations are not shared across switches.
- Cancellation is checked as output is emitted; it does not interrupt a long native
  prefill immediately. Allow more than 100 seconds for the first token near full context.
- Prefill and decode still stream offloaded experts over the host link. This remains
  the main performance limitation, and prompt processing is far below its compute ceiling.
- Released crit 0.2.0 also limits conversation text to 1,000,000 characters and
  internal request JSON to 1 MiB. Some token-valid near-262K documents exceed those
  transport limits. The standalone prompt-file command avoids that API JSON boundary.
- This source export is runnable terminal chat, not a portable crit run registry.
  The API adapter requires a locally registered and sealed run; the build below alone
  does not create one. The API is a tested chat-completions subset: tool calls with
  `tool_choice` auto or none and the four sampling fields above; no strict tools,
  structured output, `min_p`, penalties or `logit_bias`.

## Build and chat

Start in the crit-hub repository root. Set WORK to a writable directory on the Linux
filesystem, such as a subdirectory of your configured wsl_run_root. Keep builds and
checkpoints outside the repository.
Install the tested CUDA/compiler/CMake/Ninja prerequisites first. The native SDK is
pinned to Core 0.1.0; Python text handling and serving use crit 0.2.0.

```sh
ENGINE="$PWD/engines/rtx5090/qwen3.8-flash-next@FP4-Local/mtp3-expert-cache/engine"
WORK="/path/to/your/wsl-run-root/flash-build"
mkdir -p "$WORK"
git clone --branch v0.1.0 --depth 1 https://github.com/crit-labs/crit "$WORK/crit-sdk"
git clone --branch v0.2.0 --depth 1 https://github.com/crit-labs/crit "$WORK/crit-python"
python3 -m venv "$WORK/venv"
. "$WORK/venv/bin/activate"
python -m pip install -e "$WORK/crit-python"
export CUDACXX=/usr/local/cuda-13.1/bin/nvcc
cmake -S "$WORK/crit-sdk/core" -B "$WORK/build-core" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DCRIT_CORE_CUDA=ON -DBUILD_TESTING=OFF \
  -DCMAKE_CUDA_ARCHITECTURES=120a -DCMAKE_POSITION_INDEPENDENT_CODE=ON
cmake --build "$WORK/build-core" -j 4
cmake --install "$WORK/build-core" --prefix "$WORK/core-install"
cmake -S "$ENGINE" -B "$WORK/build-fnl" -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="$WORK/core-install"
cmake --build "$WORK/build-fnl" --target fnl -j 4
mkdir -p "$WORK/package"
cp "$WORK/build-fnl/libfnl.so" "$ENGINE"/python/*.py "$ENGINE/configuration.json" "$WORK/package/"
```

Download the exact checkpoint revision in [model.json](model.json), then verify it
from the crit-hub root and start the exported command:

```sh
MODEL="/path/to/downloaded/checkpoint"
python -B tools/verify_model.py "$ENGINE/../model.json" "$MODEL"
python "$WORK/package/chat.py" --model "$MODEL" --package "$WORK/package"
# One request from a UTF-8 document, including multiline/long prompts:
python "$WORK/package/chat.py" --model "$MODEL" --package "$WORK/package" \
  --prompt-file /path/to/prompt.txt --output-tokens 256
```

Interactive commands: `/new` clears the conversation; `/quit` exits. The default
capacity is 262144, including the chat template and requested output allowance.

For a registered run, `fnl_serve.py` exports `create_engine(context)` with
`chat-json-v1`, `chat-resident-v1`, `chat-stream-v1`, `chat-config-v1`,
`chat-model-file-v1`, `chat-tools-v1`, `chat-tool-stream-v1` and per-request `temperature`, `top_p`, `top_k`
and `seed` (crit main). `temperature: 0` decodes greedily in MTP rounds; omitted
fields keep the engine's defaults. Follow crit's [run registration and packaging documentation](https://github.com/crit-labs/crit/blob/v0.2.0/docs/BUILDING_ENGINES.md)
before using `crit serve`; the historical author's run ID is not part of this export.

## Benchmark your machine

`engine/tools/flashnext_bench.py` measures this engine on your PC, first with every expert miss
copied over PCIe (CPU share 0, the default), then with 25%, 50% and 75% of misses computed on the
CPU (`FNL_CPU_SHARE`). The CPU path lost on the 6-core PC this engine was built on; CPUs with
more AVX-512 cores may gain. It loads the engine once per share (several minutes each), runs
greedy MTP on a chat, a code and a ~14K-token prompt (median of 3 after a warm-up), and prints a
Markdown table. After the build and checkpoint check above, from the same shell:

```sh
python "$ENGINE/tools/flashnext_bench.py" --model "$MODEL" --package "$WORK/package"
# Fewer shares or a different CPU thread count (default: about one per physical core, minus one):
python "$ENGINE/tools/flashnext_bench.py" --model "$MODEL" --package "$WORK/package" --shares 0,0.5 --threads 14
```

It needs about 82 GiB visible to Linux and stops before a load when less than 68 GiB is free. On
a 96 GB Windows PC set `memory=88GB` under `[wsl2]` in `%UserProfile%\.wslconfig`, run
`wsl --shutdown`, and close large apps. To share results, open an issue with the printed table
and attach `flashnext-bench.json` if a share failed (it holds the error).
