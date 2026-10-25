<!-- Rendered by tools/card.py from card.yaml: edit card.yaml, then run python -B tools/card.py on this folder. -->

# ninfer-4090

### rtx4090 · qwen3.8-27b@groupwise-int

> [!WARNING]
> **Contributed by** sergiuszm · [sergiuszm/ninfer-4090](https://github.com/sergiuszm/ninfer-4090)
>
> External engine: crit-hub has not reviewed its source. Read it before you build or run it.

| Decode (sampled) | Decode (greedy) | Prefill | TTFT | Perplexity | Mean KL divergence | Context |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| – | **148.6 tok/s** | **1,849 tok/s** | – | – | – | **262,144 tok** |

- **Measured with:** self-reported (ninfer CLI and ninfer-serve metrics) · [self-reported.json](evidence/self-reported.json)
- **Hardware:** RTX 4090 24 GB
- **Requires:** Linux, Windows · sm_89 · 24 GB GPU memory · CUDA 13.3
- **Checkpoint** (`qwen3.8-27b@groupwise-int`): [`neroued/Qwen3.8-27B-NInfer`](https://huggingface.co/neroued/Qwen3.8-27B-NInfer/tree/1cbd84e7221e51186bd7f093a149912d2489625b) at `1cbd84e7`
- **Source:** [sergiuszm/ninfer-4090 @ `fdb7a6cf`](https://github.com/sergiuszm/ninfer-4090/tree/fdb7a6cf3284770b9609ef7b3eac873f84e3849f)
- **License:** Apache-2.0

NInfer-4090 runs Qwen3.8-27B on one 24 GB RTX 4090. It is an `sm_89` port of
[NInfer-3090](https://github.com/Don-Chad/ninfer-3090), which derives from
[Neroued/ninfer](https://github.com/Neroued/ninfer); Blackwell-only NVFP4 is unavailable, so it uses
the official groupwise `.ninfer` artifact. Its E8 lattice 4-bit KV mode fits the model's full native
262,144-token context on 24 GB.

---

## Design

| Part | How |
| --- | --- |
| Weights | The official groupwise-int Qwen3.8-27B artifact (16.96 GiB) |
| Speculation | MTP3 with `--lm-head-draft` |
| KV cache | `rk4v4-e8` (E8 Conway-Sloane lattice, 4-bit) by default for 262,144 tokens; INT8 for the headline table |
| State | Paged KV, compatible-prefix reuse, ReplaySSM state transactions, long anchors at message boundaries |
| Serving | OpenAI- and Anthropic-compatible APIs, reasoning-effort control |

---

## Measured

Single request, greedy, CUDA Graphs, INT8 KV, `--prefill-chunk 1024`:

| Test | Result |
| --- | --- |
| Decode, code generation, MTP3 | **148.6 tok/s** at 81.0% acceptance |
| Decode, bench corpus, MTP3 | 106.5 tok/s at 48.7% |
| Decode, no speculation / at 128K depth | 50.5 / 39.6 tok/s |
| 64K / 128K needle-in-a-haystack | exact, 1,849 / 1,561 tok/s prefill |

Against llama.cpp on the same card (UD-Q4_K_XL GGUF): NInfer decodes 10% faster shallow and 27%
faster at 128K without speculation, and serves past llama.cpp's 144K ceiling; llama.cpp prefills
4-10% faster up to 128K.

---

## Checks

- **Ops:** 78 of 78 runnable op tests pass on `sm_89`.
- **Retrieval:** single-needle, 5-needle and exact-code-detail probes exact through 260K with E8 KV.
- **Vision:** 3 of 3 oracle facts on a chart-reading probe.

---

## Limits

- The E8 KV default costs 5.7% decode and 1-2% prefill against the INT8 figures above.
- Prefill serializes across lanes; a deep cold prefill delays the other lane's first token.
- The Qwen3.6-35B-A3B target is inherited but untested on the RTX 4090.

---

## Build and chat

```sh
git clone https://github.com/sergiuszm/ninfer-4090.git
git -C ninfer-4090 checkout fdb7a6cf3284770b9609ef7b3eac873f84e3849f
cd ninfer-4090
docker build --tag ninfer-4090:sm89 .
NINFER_MODEL_DIR="$PWD/models" bash scripts/download-qwen38.sh
docker run --rm --gpus all --publish 8080:8080 --volume "$PWD/models:/workspace/models:ro" ninfer-4090:sm89 \
  ninfer-serve models/qwen3_8_27b.ninfer --host 0.0.0.0 --port 8080 --max-context 262144 --kv-capacity 262144 \
  --max-concurrency 1 --prefill-chunk 1024 --kv-dtype rk4v4-e8 --spec mtp --draft-tokens 3 --lm-head-draft --preserve-thinking
```

The API is on `http://127.0.0.1:8080/v1`. Check the artifact with
`python -B tools/verify_model.py <this engine folder> models` from a crit-hub checkout.

---

## Credits

Apache License 2.0,
[LICENSE](https://github.com/sergiuszm/ninfer-4090/blob/fdb7a6cf3284770b9609ef7b3eac873f84e3849f/LICENSE)
at the pinned commit. Derived from [NInfer-3090](https://github.com/Don-Chad/ninfer-3090) and
[Neroued/ninfer](https://github.com/Neroued/ninfer); the E8 KV mode is ported from
[UDPSendToFailed/ninfer-4090](https://github.com/UDPSendToFailed/ninfer-4090).
