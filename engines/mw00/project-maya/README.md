<!-- Rendered by tools/card.py from card.yaml: edit card.yaml, then run python -B tools/card.py on this folder. -->

# project-maya

### rtxpro6000 · glm-5.3-flash@Maya-M-IQ2-S

> [!WARNING]
> **Contributed by** mw00 · [mw00/project-maya](https://github.com/mw00/project-maya)
>
> External engine: crit-hub has not reviewed its source. Read it before you build or run it.

| Decode (sampled) | Decode (greedy) | Prefill | TTFT | Perplexity | Mean KL divergence | Context |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| – | **81.0 tok/s** | **3,102 tok/s** | – | – | – | **262,144 tok** |

- **Measured with:** self-reported (maya.sh --bench (user report)) · [self-reported.json](evidence/self-reported.json)
- **Hardware:** RTX PRO 6000 Blackwell 96 GB; Ryzen 7 9800X3D; 61 GB RAM
- **Requires:** Linux, Windows · sm_70, sm_75, sm_80, sm_86, sm_89, sm_120 · 16 GB GPU memory · CUDA 12.8
- **Checkpoint** (`glm-5.3-flash@Maya-M-IQ2-S`): [`peasantsmith/GLM-5.3-Flash-Maya-GGUF`](https://huggingface.co/peasantsmith/GLM-5.3-Flash-Maya-GGUF/tree/f16ee4ba50d8fd2bb726ea1862ca6ca14ba188da) at `f16ee4ba`
- **Source:** [mw00/project-maya @ `8d9a9b5d`](https://github.com/mw00/project-maya/tree/8d9a9b5d705049a87da853717cdc59440252de30)
- **License:** MIT

Project Maya runs GLM-5.3-Flash (321B) and DeepSeek-V4.1-Flash-1017 (748B) on hardware people
already have: one NVIDIA GPU or up to 16, with AMD and Windows experimental. Each model has its own
engine, which keeps the experts a conversation uses most on the GPU, the next ones in RAM and the
rest on the NVMe SSD, and tunes itself to the machine. It serves OpenAI- and Anthropic-compatible
APIs with tool calls, pictures and up to eight requests at once. It started from Strata's engine.

---

## Design

| Part | How |
| --- | --- |
| Experts | Tiered across VRAM, RAM and SSD and moved as you chat; layers split across GPUs |
| Weights | Maya's own GLM-5.3-Flash quants from Z.ai's FP8 release (S 96.5 GB, S24, M 116 GB, L 156 GB), MTP block kept; EXL3 and unsloth GGUFs for DeepSeek |
| Speculation | The model's MTP block, or an optional DFlash2 drafter (non-commercial license) kept only where it is faster |
| KV cache | Streaming: each layer's cache in RAM with a window in VRAM (`--kv-streaming on`) |
| Serving | OpenAI and Anthropic APIs, chat and monitor in the browser, `--parallel` up to 8 |

---

## Measured

A selection of the community table (`./maya.sh --bench`, prefill of an 8K-token prompt):

| Machine | Model | Decode | Prefill |
| --- | --- | ---: | ---: |
| 4x RTX 4090 24 GB | Maya-S24 | 118 tok/s | 4,129 tok/s |
| **RTX PRO 6000 Blackwell 96 GB** | Maya-M, 256K context | **81.0 tok/s** | 3,102 tok/s |
| 2x RTX 3090 24 GB (225 W) | Maya-S24 | 53.0 tok/s | 885 tok/s |
| RTX 5090 32 GB | Maya-M | 30+ tok/s | ~2,000 tok/s |
| 1x RTX 3090 24 GB | Maya-M, 128K context | 27.0 tok/s | 1,069 tok/s |

The card records the RTX PRO 6000 row.

---

## Checks

- **Quants:** Maya's GLM-5.3-Flash quants keep 97.7-99.2% of the FP8 model's zero-shot accuracy (Maya-M: 97.9%, the same next token 86% of the time).
- **DeepSeek:** Maya-S of DeepSeek-V4.1-Flash-1017 picks the original's next token 92.6% of the time.

---

## Limits

- The figure is a user report on v1.1.7 with no date; the pinned commit is v1.1.17.
- The engine pins all but about 6 GB of free RAM while it runs.
- AMD, Windows and Pascal support are experimental.

---

## Build and chat

```sh
git clone https://github.com/mw00/project-maya.git
git -C project-maya checkout 8d9a9b5d705049a87da853717cdc59440252de30
cd project-maya && ./setup.sh
```

The setup checks the machine, asks which quant to download (Maya-M for this card), builds everything
once and starts the model at `http://127.0.0.1:8080`. Check the download with
`python -B tools/verify_model.py <this engine folder> <model folder>`. Every option:
[docs/SETUP.md](https://github.com/mw00/project-maya/blob/8d9a9b5d705049a87da853717cdc59440252de30/docs/SETUP.md).

---

## Credits

MIT, [LICENSE](https://github.com/mw00/project-maya/blob/8d9a9b5d705049a87da853717cdc59440252de30/LICENSE)
at the pinned commit. Created and led by Peasant Smith (mw00), co-maintained with needmorevram; built
on [Strata](https://github.com/Niko1221/Strata) and ggml/llama.cpp, with exllamav3 and beellama.cpp
parts. GLM-5.3-Flash is zai-org's (MIT).
