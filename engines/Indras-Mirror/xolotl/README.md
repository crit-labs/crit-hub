<!-- Rendered by tools/card.py from card.yaml: edit card.yaml, then run python -B tools/card.py on this folder. -->

# xolotl

### rtx4090 · deepseek-v4.1-flash@GSQ

> [!WARNING]
> **Contributed by** Indras-Mirror · [Indras-Mirror/Xolotl](https://github.com/Indras-Mirror/Xolotl)
>
> External engine: crit-hub has not reviewed its source. Read it before you build or run it.

| Decode (sampled) | Decode (greedy) | Prefill | TTFT | Perplexity | Mean KL divergence | Context |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| – | **26.8 tok/s** | **180–320 tok/s** | – | – | – | **524,288 tok** |

- **Measured with:** self-reported (Xolotl task runs through the server) · [self-reported.json](evidence/self-reported.json)
- **Hardware:** Linux; Ryzen 7 5700X; 90 GB DDR4; one NVMe drive; RTX 4090 24 GB
- **Requires:** Linux · sm_89 · 24 GB GPU memory · CUDA 13.0
- **Checkpoint** (`deepseek-v4.1-flash@GSQ`): [`IndrasMirror/DeepSeek-V4.1-Flash-GSQ-Uncensored`](https://huggingface.co/IndrasMirror/DeepSeek-V4.1-Flash-GSQ-Uncensored/tree/0ae18f838e103fb03d952fa8ed73c7aa9c34002f) at `0ae18f83`
- **Source:** [Indras-Mirror/Xolotl @ `9747b9e3`](https://github.com/Indras-Mirror/Xolotl/tree/9747b9e3db0b499242434e7d44da60f4d4a96a01)
- **License:** MIT

Xolotl is a Strata fork that keeps [Strata](https://github.com/Niko1221/Strata)'s engine, server and
kernels at its core and adds DeepSeek-V4.1 and V4-Flash engines, a GLM-5.3-Flash engine and a
Qwen3.8-27B dense port, each with its own quantization and expert-tiering work. Its headline is
DeepSeek-V4.1-Flash, all 552 billion parameters, on one gaming PC: weights live across VRAM, pinned
RAM and NVMe, and the engine carries up the experts each token needs.

---

## Design

| Part | How |
| --- | --- |
| Weights | A GSQ mixed ternary / 2-bit / 3-bit quant of DeepSeek-V4.1-Flash (152 GiB) |
| Experts | Tiered across VRAM, pinned RAM and NVMe; the engine learns a conversation's experts between turns |
| Speculation | A speculative-decode draft, kept off the card during long prompts by default |
| Serving | One OpenAI- and Anthropic-compatible server per model, prefix reuse across agent turns, a thinking budget |

---

## Measured

One RTX 4090 (24 GB), Ryzen 7 5700X, 90 GB DDR4, one NVMe drive; every number measured through the
server while it solved real tasks, greedy, thinking on with a 32K budget:

| Model | Prefill | Decode | HumanEval | Tool calls | Agentic | Context |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| DeepSeek-V4.1-Flash | 180-320 tok/s | **26.8 tok/s** | 21/21 | 10/10 | 3/3 | 512K |
| GLM-5.3-Flash | 290-375 tok/s | 25.3 tok/s | 19/21 | 10/10 | 3/3 | 512K |
| Qwen3.8-Flash-Next | 3,000-3,430 tok/s | 127.8 tok/s | 21/21 | 10/10 | 3/3 | 200K |
| Qwen3.8-27B | 1,240-1,390 tok/s | 98.5 tok/s | 21/21 | 10/10 | 3/3 | 200K |

Only the DeepSeek row is listed as a result here, since the README names its checkpoint; the other
models' configurations point at local files.

---

## Checks

- **Tasks:** 21 HumanEval problems, 10 tool-call tasks and 3 agentic bug-fix loops per model, through the server.
- **Against the API:** how close the DeepSeek quant gets to DeepSeek's own API is in `docs/xolotl/BENCHMARKS.md` upstream.

---

## Limits

- One measured setup (the hardware above); Linux only for the DeepSeek engine.
- The DeepSeek checkpoint is an uncensored GSQ quant, not DeepSeek's release.

---

## Build and chat

```sh
git clone https://github.com/Indras-Mirror/Xolotl.git xolotl
git -C xolotl checkout 9747b9e3db0b499242434e7d44da60f4d4a96a01
cd xolotl
bash tools/xolotl/fetch-llama.sh
cmake -S . -B build-xolotl -G Ninja -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON -DSTRATA_GGML_CUDA=ON \
  -DSTRATA_MMQ_KQUANTS=ON -DCMAKE_CUDA_ARCHITECTURES=89 -DSTRATA_GGML_DIR=$PWD/third_party/llama.cpp \
  -DSTRATA_DSV41_FQ_OP=ON -DSTRATA_DSV41_HC_MIX=ON
ninja -C build-xolotl
python3 tools/v41/serve/serve_ds41.py --engine strata --config tools/v41/serve/strata-ds41.json --port 8150
```

Download the checkpoint (its card has the Engram steps), check it with
`python -B tools/verify_model.py <this engine folder> <model folder>`, and edit the model paths in the
config first.

---

## Credits

MIT, [LICENSE](https://github.com/Indras-Mirror/Xolotl/blob/9747b9e3db0b499242434e7d44da60f4d4a96a01/LICENSE)
at the pinned commit (Niko1221 and the Strata contributors). Built on Strata and llama.cpp's ggml.
