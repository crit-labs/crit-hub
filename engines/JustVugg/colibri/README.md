<!-- Rendered by tools/card.py from card.yaml: edit card.yaml, then run python -B tools/card.py on this folder. -->

# colibri

### rtx3070 · qwen3.6-35b-a3b@colibri-int4

> [!WARNING]
> **Contributed by** JustVugg · [JustVugg/colibri](https://github.com/JustVugg/colibri)
>
> External engine: crit-hub has not reviewed its source. Read it before you build or run it.

| Decode (sampled) | Decode (greedy) | Prefill | TTFT | Perplexity | Mean KL divergence | Context |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| – | **30.0 tok/s** | – | – | – | – | – |

- **Measured with:** self-reported (colibri qwen36 COLI_TIMERS) · [self-reported.json](evidence/self-reported.json)
- **Hardware:** Threadripper 3945WX 12C; RTX 3070 8 GB
- **Requires:** Linux, Windows, macOS · sm_86, sm_89 · 8 GB GPU memory · CUDA 12.9
- **Checkpoint** (`qwen3.6-35b-a3b@colibri-int4`): [`Qwen/Qwen3.6-35B-A3B`](https://huggingface.co/Qwen/Qwen3.6-35B-A3B/tree/995ad96eacd98c81ed38be0c5b274b04031597b0) at `995ad96e`
- **Source:** [JustVugg/colibri @ `e0db083c`](https://github.com/JustVugg/colibri/tree/e0db083cdc9e563e0766cc6fc661ef23c32298e0)
- **License:** Apache-2.0

colibri runs very large mixture-of-experts models on ordinary machines: pure C, one file per model
family, no GPU required. It keeps the part of the model every token uses in RAM and reads the
experts from disk when the router asks for them. Thirteen engines run today (GLM-5.x, DeepSeek V4 and
V4.1 Flash, Kimi K3, Qwen3.8-Flash-Next, Qwen3.6 and others). The card's figure is the Qwen3.6 engine
with its CUDA tier, which promotes hot experts into VRAM on one or more GPUs.

---

## Design

| Part | How |
| --- | --- |
| Engines | One C file per model family, built with `make -C c <family>` |
| Experts | Routed experts streamed from an int4/int8 container on disk, the hot set kept in RAM |
| CUDA tier | Hot experts promoted to VRAM by routing heat (LFRU), misses on the CPU int8 path overlapping the GPU groups; a heat file warm-starts the next run |
| DeltaNet | `Q36_DN_GPU=1` runs Qwen3.6's 30 DeltaNet layers on the card with their recurrent state in VRAM |
| Backends | CPU, CUDA, HIP/ROCm and Vulkan; Windows through a CUDA DLL |

---

## Measured

RTX 3070 8 GB alone, per-row int4 Qwen3.6-35B-A3B, 49-token prompt, 200 decoded tokens, dense layers
in VRAM:

| | `Q36_DN_GPU=0` | `Q36_DN_GPU=1` |
| --- | ---: | ---: |
| Decode | 25.4 tok/s | **30.0 tok/s** |
| As printed, time to first token included | 21.2 tok/s | 24.1 tok/s |
| VRAM hit rate | 82% | 82% |

For scale, the same engine on the CPU alone ran 0.35 tok/s before the tier, and Ollama on the same
card shows 38.1 tok/s.

---

## Checks

- **Placement:** promoting experts never changes routing or precision; GPU output is not byte-identical
  to the CPU path (different summation order) but two GPU runs are identical.

---

## Limits

- No measurement date or context length is given for the figure.
- Ollama is faster on this card for a model this size; colibri's case is models larger than RAM.
- CUDA 13 no longer compiles for Volta; such cards fall back to Vulkan.

---

## Build and chat

```sh
git clone https://github.com/JustVugg/colibri.git
git -C colibri checkout e0db083cdc9e563e0766cc6fc661ef23c32298e0
make -C colibri/c qwen36 CUDA=1 CUDA_ARCH=native
```

Download `Qwen/Qwen3.6-35B-A3B`, check it with
`python -B tools/verify_model.py <this engine folder> <model folder>`, convert it to a container and
run with `COLI_CUDA=1 Q36_DN_GPU=1`; the steps are in
[docs/qwen36.md](https://github.com/JustVugg/colibri/blob/e0db083cdc9e563e0766cc6fc661ef23c32298e0/docs/qwen36.md)
and [docs/qwen36-cuda-tier.md](https://github.com/JustVugg/colibri/blob/e0db083cdc9e563e0766cc6fc661ef23c32298e0/docs/qwen36-cuda-tier.md).

---

## Credits

Apache License 2.0,
[LICENSE](https://github.com/JustVugg/colibri/blob/e0db083cdc9e563e0766cc6fc661ef23c32298e0/LICENSE)
at the pinned commit; third-party notices in THIRD_PARTY_NOTICES.md. The model is Qwen's.
