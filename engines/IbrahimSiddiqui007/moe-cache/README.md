<!-- Rendered by tools/card.py from card.yaml: edit card.yaml, then run python -B tools/card.py on this folder. -->

# moe-cache

### rtx4060-laptop · qwen3.8-flash-next@GSQ-IQ2-XS

> [!WARNING]
> **Contributed by** IbrahimSiddiqui007 · [IbrahimSiddiqui007/moe-cache](https://github.com/IbrahimSiddiqui007/moe-cache)
>
> External engine: crit-hub has not reviewed its source. Read it before you build or run it.

| Decode (sampled) | Decode (greedy) | Prefill | TTFT | Perplexity | Mean KL divergence | Context |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| – | **12.5 tok/s** | **90–103 tok/s** | – | – | – | – |

- **Measured with:** self-reported (moe-cache clean-run gate) · [self-reported.json](evidence/self-reported.json)
- **Hardware:** Laptop: Intel i7-13620H; 31 GB RAM with a 24 GB cgroup limit; RTX 4060 8 GB; NVMe SSD
- **Requires:** Linux · sm_89 · 8 GB GPU memory · CUDA 13.0
- **Checkpoint** (`qwen3.8-flash-next@GSQ-IQ2-XS`): [`ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF`](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/tree/ed59f92082b1e93c0e96d60a8b11aab089b52f09) at `ed59f920`
- **Source:** [IbrahimSiddiqui007/moe-cache @ `bb931534`](https://github.com/IbrahimSiddiqui007/moe-cache/tree/bb9315345eea6fbed5c7826a452c5864788ef758)
- **License:** MIT

moe-cache runs mixture-of-experts models that are bigger than your RAM on an unmodified
`llama-server`, as a ggml backend plugin: no fork, no patch. It manages the CPU-side experts itself,
keeping the hot ones in RAM and reading the rest from the SSD with large parallel direct reads, sizes
its cache from the real memory limit and learns which experts you use. Output is token-identical to
stock llama.cpp unless an approximate mode is switched on.

---

## Design

| Part | How |
| --- | --- |
| Plugin | A ggml backend that takes the CPU-side expert matrix multiplications |
| Cache | Hot experts in RAM (LRU), the rest read from SSD in the background; warm start from a usage profile |
| Executor | Computes the experts already in RAM while missing ones load, optionally on efficiency cores |
| Kernels | llama.cpp's own dot products, plus a SIMD Q2_0 kernel (about 3x, bit-identical) |
| Opt-in | `--fast` skips a low-ranked expert not in RAM; `--hybrid` computes a bank of hot experts on the GPU |

---

## Measured

Intel i7-13620H laptop, 31 GB RAM with a cgroup limit, RTX 4060 8 GB, NVMe; stock runs first drop the
model from the page cache:

| Situation | Stock llama.cpp | moe-cache |
| --- | ---: | ---: |
| Qwen3.8-Flash-Next IQ2_XS (68 GB), 24 GB RAM, 8 GB GPU | 5.2 tok/s | **12.5 tok/s** (17.6 with `--hybrid`) |
| Same, prefill of 2-5K-token prompts | 65.9 tok/s | **90-103 tok/s** |
| GLM-5.3-Flash (102 GB), 24 GB RAM | 0.32 tok/s | 1.75 tok/s |
| Qwen3.6-35B, 12 GB RAM | 6.9 tok/s | 22.6 tok/s |
| Qwen3.6-35B fits in RAM (24 GB) | **42.9 tok/s** | 36.6 tok/s |

When the model fits in RAM, stock llama.cpp is faster and the launcher runs it instead.

---

## Checks

- **Identity:** `tests/identity.sh` compares tokens with stock llama.cpp; checked on every model in RESULTS.
- **Executor:** every output compared byte for byte with llama.cpp's own on the test models.

---

## Limits

- Linux only; Windows code compiles but has never run with llama.cpp.
- The checkpoint is identified in RESULTS only as ISTA's IQ2_XS file; the pinned repository is assumed to be ISTA-DASLab's GSQ-RCO GGUF.
- The CUDA version is not stated upstream; one run per row unless a section says paired.

---

## Build and chat

```sh
git clone https://github.com/IbrahimSiddiqui007/moe-cache.git
git -C moe-cache checkout bb9315345eea6fbed5c7826a452c5864788ef758
cd moe-cache && ./build.sh
bin/moe-cache-server /path/to/model.gguf --ram 24 --llama-server /path/to/llama-server
```

`./install.sh` builds a matching llama.cpp as well. Check the model with
`python -B tools/verify_model.py <this engine folder> <model folder>`; the server and its OpenAI API
are at `http://localhost:8080`.

---

## Credits

MIT, [LICENSE](https://github.com/IbrahimSiddiqui007/moe-cache/blob/bb9315345eea6fbed5c7826a452c5864788ef758/LICENSE)
at the pinned commit; ggml interface headers under ggml's MIT license. A plugin for llama.cpp and
ggml; the GSQ-RCO quants are IST Austria DAS Lab's and the model is Qwen's.
