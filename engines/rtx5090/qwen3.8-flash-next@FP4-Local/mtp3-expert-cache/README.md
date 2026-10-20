# Qwen3.8-Flash-Next @ FP4-Local · RTX 5090 · mtp3-expert-cache

> [!NOTE]
> **Contributed by** crit-labs · [crit-labs/crit](https://github.com/crit-labs/crit)
>
> Built with [crit](https://github.com/crit-labs/crit): a sealed crit candidate, exported with `tools/export_engine.py`.

An engine written from scratch on crit Core 0.1.0, which it links as an installed SDK. It runs text
generation on WSL2 Ubuntu 24.04 with CUDA 13.1 on sm_120a (RTX 5090, 32 GB) and a host with about
82 GiB of RAM available to WSL, at a 262,144-token context. The release is candidate-0010;
[engine/VALIDATION.md](engine/VALIDATION.md) has its history and the full validation detail.

## Model

`nvidia/Qwen3.8-Flash-Next-FP4-Local` at revision `b9ce18a6887e18c089fdea8509a33e63429a3300`
(qwen4_exp text path): 48 layers, 36 gated delta-net and 12 QSA attention, 4-stream
hyper-connections, a PLE n-gram embedding, 512 NVFP4 experts top-10 with CSF-encoded scales plus a
shared expert, and an MTP block. The checkpoint is about 102 GB, so it cannot sit on the GPU. The
weights are used as stored. [model.json](model.json) lists every checkpoint file with the size and
SHA-256 the run recorded; `python -B tools/verify_model.py` checks a download against it.

## Design

| Part | How |
| --- | --- |
| Resident weights | Attention, delta-net and shared-expert projections (MXFP8); router, hyper-connections and head (BF16) on the GPU |
| Routed experts | NVFP4 with CSF scales kept compressed, in pinned host RAM, about 61 GB |
| Hot experts | GPU CLOCK slot cache (sized from free VRAM with a 2 GiB reserve), which also borrows KV positions a request can't reach and the idle prompt workspace; a miss is read straight from pinned host memory and written through to its slot |
| KV cache | FP8 E4M3 with an FP32 scale per 256-dim vector, sized for 262,144 tokens |
| Embeddings | Token embeddings in pinned host RAM, read through the device mapping; the PLE n-gram table memory-mapped, rows gathered on the host per token |
| Speculative decoding | The checkpoint's MTP block as released: 3 drafts verified in one 4-token pass, accepted and committed on the device in CUDA graphs; the drafting head re-encoded to MXFP8, the verify pass on the stored BF16 head (greedy only) |
| Prompts | 16K-token chunks of FP16 tensor-core GEMMs with one batched expert copy per layer |
| Chat | crit's shared ResidentChat: tool calls through the model's chat template, per-request temperature, top_p, top_k and seed (default temperature 1, top-p 0.95, top-k 20, thinking off); sampled requests decode one token per pass with crit Core's sampler |

[engine/README.md](engine/README.md) has the full design notes.

## Measured

crit Eval of the byte-identical candidate-0006 engine, 6 speed cases at a 262,144-token capacity,
greedy MTP decoding:

| Metric | Result |
| --- | ---: |
| Decode | 84.5 tok/s |
| Prefill | 1,801 tok/s |
| Time to first token (geomean) | 1.61 s |
| Decode at 261,888 prompt + 256 output tokens | 63.1 tok/s |
| First token at that full capacity | 99.8 s |
| Held-out decode (10 prompts) | 92.1 tok/s |

| Run benchmark, 256 output tokens | Decode | Time to first token | Prefill |
| --- | ---: | ---: | ---: |
| 512-token prompt | 91.7 tok/s | 2.19 s | 234 tok/s |
| 14,336-token prompt | 83.3 tok/s | 4.81 s | 2,978 tok/s |

Run benchmarks are candidate-0010's medians of three repetitions after one warmup, concurrency one,
resident, stop tokens disabled. Expert misses streamed over the host link set the speed, and prompt
processing is far below its compute ceiling.

## Checks

- **Arithmetic:** the repaired binary passes the prefill-to-decode reference check that the original
  candidate-0003 failed (at 4,094 tokens: PPL ×1.007, KL mean 0.00044, p99 0.010 against limits
  1.05 / 0.05 / 0.5), against an independent FP32 Transformers `qwen4_exp` reference over the stored
  checkpoint. Eval quality gates ran under crit-eval-v3; a crit-eval-v4 run is pending.
- **Functional:** all Eval gates, six fresh-process repeatability cases and nineteen generation
  checks outside Eval passed; retrieval succeeds from 1K to 259,874 input tokens. `crit serve` passed
  consecutive requests, system messages, streaming, a stop string, candidate switches, Ctrl+C, a tool
  call and its result, a streamed call, greedy `temperature: 0` and seeded sampling.
- **Resources:** full-Eval telemetry left at least 0.85 GiB of GPU memory and 17.3 GiB of host RAM
  free.

## Limits

- Known issue: speculative output equals plain greedy in 3/6 Eval cases (tool-call, long-summary and
  agent-long first differ at tokens 125, 19 and 146). The cause is not yet established; until it is,
  treat it as an open bug, not rounding.
- Text only, one request at a time; other GPUs and native Windows are untested.
- Sampled requests decode one token per pass and are slower than the greedy figures above.
- Loading takes several minutes: the tested API startup took 880 s, so use `crit serve --timeout 1800`.
  Candidate switches reload the weights (211–248 s).
- Allow more than 100 s for the first token near full context; cancellation does not interrupt a long
  prefill immediately.
- Released crit 0.2.0 caps conversation text at 1,000,000 characters; the standalone prompt-file
  command avoids that limit. [engine/VALIDATION.md](engine/VALIDATION.md) lists the remaining limits.

## Build and chat

Start in the crit-hub root. Set WORK to a writable directory on the Linux filesystem and keep builds
and checkpoints outside the repository. The native SDK is pinned to Core 0.1.0; Python text handling
and serving use crit 0.2.0.

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

Download the exact checkpoint revision in [model.json](model.json), verify it and chat in the
terminal (`/new` clears the conversation, `/quit` exits):

```sh
MODEL="/path/to/downloaded/checkpoint"
python -B tools/verify_model.py "$ENGINE/../model.json" "$MODEL"
python "$WORK/package/chat.py" --model "$MODEL" --package "$WORK/package"
# One request from a UTF-8 document, including multiline/long prompts:
python "$WORK/package/chat.py" --model "$MODEL" --package "$WORK/package" \
  --prompt-file /path/to/prompt.txt --output-tokens 256
```

To serve it, register the package as a crit run candidate following crit's
[run registration and packaging documentation](https://github.com/crit-labs/crit/blob/v0.2.0/docs/BUILDING_ENGINES.md)
and run `crit serve --timeout 1800` (`fnl_serve.py` is the serve adapter); the original run ID is not
part of this export.

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

## Credits

Credits and licenses are in [THIRD_PARTY.md](THIRD_PARTY.md) and [LICENSES.txt](LICENSES.txt).
