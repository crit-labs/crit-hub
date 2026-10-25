<!-- Rendered by tools/card.py from card.yaml: edit card.yaml, then run python -B tools/card.py on this folder. -->

# ignis

### rtx5090 · qwen3.8-27b@NVFP4

> [!WARNING]
> **Contributed by** gpillon · [gpillon/ignis](https://github.com/gpillon/ignis)
>
> External engine: crit-hub has not reviewed its source. Read it before you build or run it.

| Decode (sampled) | Decode (greedy) | Prefill | TTFT | Perplexity | Mean KL divergence | Context |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| – | **367 tok/s** | – | – | – | – | **262,144 tok** |

- **Measured with:** self-reported (Ignis findings A/B) · [self-reported.json](evidence/self-reported.json)
- **Hardware:** RTX 5090 32 GB; 64 GB DDR4-3200; PCIe 3.0
- **Requires:** Linux, Windows · sm_120a · 32 GB GPU memory · CUDA 13.1
- **Checkpoint** (`qwen3.8-27b@NVFP4`): [`gpillon/Qwen3.8-27B-nvfp4full-dflash2-NInfer`](https://huggingface.co/gpillon/Qwen3.8-27B-nvfp4full-dflash2-NInfer/tree/86d9a4bf76997639dcd64f78fde2d73740ad9a3e) at `86d9a4bf`
- **Checkpoint** (`qwen3.8-flash-next@trellis-2.5bpw`): [`gpillon/Qwen3.8-Flash-Next-trellis-a25-ignis`](https://huggingface.co/gpillon/Qwen3.8-Flash-Next-trellis-a25-ignis/tree/258a3d43d6dbd682c1a31e6c7efbf248ff72bef0) at `258a3d43`
- **Source:** [gpillon/ignis @ `75c9fdc8`](https://github.com/gpillon/ignis/tree/75c9fdc838ada96ff84129f4e6c70058c690ccd0)
- **License:** Apache-2.0

Ignis is a deliberately specialized engine: one model family on one class of card. It loads an
NVFP4 Qwen3.8-27B onto a single RTX 5090 or RTX PRO 6000 and is shaped around one main agent plus a
handful of subagents hitting the same card: eight resident decode lanes run as one batch-wide round,
the paged KV cache is budgeted in bytes, conversation state outlives the request that built it, and
a reasoning loop is caught while it streams. It also serves Qwen3.8-Flash-Next (trellis 2.5 bpw
experts) and Qwen-Image-2.1-Turbo, switching models without a restart.

---

## Design

| Part | How |
| --- | --- |
| Runtime | Rust above the step (scheduling, admission, KV accounting, serving); a C++/CUDA static library for the forward pass |
| Weights | Qwen3.8-27B NVFP4 with a DFlash2 drafter in the artifact; Flash-Next experts re-quantized to a 2-4 bit trellis code (2.5 bpw average) |
| Speculation | DFlash2, 7 draft tokens (27B); the model's MTP head on the `rtxpro6000` profile (Flash-Next) |
| KV cache | Paged, byte-budgeted, `hq-e8-2b` (7.11x the capacity of BF16); 512K context with YaRN x2 |
| Serving | OpenAI v1 API, Playground, live Monitor and Prometheus metrics in the binary |

---

## Measured

One RTX 5090 (64 GB DDR4-3200, PCIe 3.0), Qwen3.8-27B NVFP4, hq-e8-2b KV, DFlash2 with 7 draft
tokens, greedy, short prompts, 262,144-token maximum:

| Load | Throughput |
| --- | ---: |
| One lane, eight coding prompts, geometric mean | **367 tok/s** |
| One lane, predictable code / free prose | 491 / 167 tok/s |
| Eight lanes, aggregate | 1,065 tok/s |

Flash-Next on the same 32 GB card runs about 100 tok/s at one lane on a PCIe 3.0 board; on a 96 GB
RTX PRO 6000 with every expert in VRAM, about 258 tok/s on code at one lane. The README notes that
prefill is where Ignis is not ahead.

---

## Checks

- **Findings:** each figure links a dated findings document under `docs/findings/` upstream.

---

## Limits

- One model family and one class of card (SM120a); no DGX Spark (see `docs/user/dgx-spark.md` upstream).
- Prefill trails other engines on the same card.
- The image model's weights are for non-commercial use only.

---

## Build and chat

```sh
git clone https://github.com/gpillon/ignis.git
git -C ignis checkout 75c9fdc838ada96ff84129f4e6c70058c690ccd0
```

Build from the checkout as [Quick start](https://github.com/gpillon/ignis/blob/75c9fdc838ada96ff84129f4e6c70058c690ccd0/README.md#quick-start)
describes, or take a [release](https://github.com/gpillon/ignis/releases). Then:

```sh
ignis-server --model-max-context 524288 --spec-backend dflash2 --spec-draft-tokens 7 --vision-enabled
```

The API is on `http://127.0.0.1:8000/v1` and the Playground on `/ui/`. Check the weights with
`python -B tools/verify_model.py <this engine folder> <model folder>`.

---

## Credits

Apache License 2.0,
[LICENSE](https://github.com/gpillon/ignis/blob/75c9fdc838ada96ff84129f4e6c70058c690ccd0/LICENSE)
at the pinned commit. The weights keep their own licenses, given on their Hugging Face cards.
