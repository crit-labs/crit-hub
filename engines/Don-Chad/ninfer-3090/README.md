<!-- Rendered by tools/card.py from card.yaml: edit card.yaml, then run python -B tools/card.py on this folder. -->

# ninfer-3090

### rtx3090 · qwen3.8-27b@groupwise-int

> [!WARNING]
> **Contributed by** Don-Chad · [Don-Chad/ninfer-3090](https://github.com/Don-Chad/ninfer-3090)
>
> External engine: crit-hub has not reviewed its source. Read it before you build or run it.

| Decode (sampled) | Decode (greedy) | Prefill | TTFT | Perplexity | Mean KL divergence | Context |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| – | **116.7 tok/s** | – | **127 ms** | – | – | **16,384 tok** |

- **Measured with:** self-reported (local qualification run) · [self-reported.json](evidence/self-reported.json)
- **Hardware:** Windows; RTX 3090 24 GB
- **Requires:** Windows, Linux · sm_86 · 24 GB GPU memory · CUDA 13.2
- **Checkpoint** (`qwen3.8-27b@groupwise-int`): [`WarlaxZ/Qwen3.8-27B-NInfer-3090`](https://huggingface.co/WarlaxZ/Qwen3.8-27B-NInfer-3090/tree/ea90e8766354c723e6bd0cf0ed731d5d71ce5fa1) at `ea90e876`
- **Source:** [Don-Chad/ninfer-3090 @ `d8c29253`](https://github.com/Don-Chad/ninfer-3090/tree/d8c29253f2c0845611a794d87057950107a4956b)
- **License:** Apache-2.0

NInfer-3090 is a port of [NInfer](https://github.com/Neroued/ninfer) to Ampere: a native C++20/CUDA
runtime with specialized SM86 kernels, quantized execution, CUDA Graphs and ReplaySSM state
management, built around the RTX 3090's compute and memory. It serves Qwen3.8-27B and Qwen3.6, and
Qwen3.8-Flash-Next as a Windows preview, with no Python or PyTorch at inference time.

---

## Design

| Part | How |
| --- | --- |
| Weights | The RTX 3090 `groupwise-int` recipe of Qwen3.8-27B (about 19 GB) |
| Execution | SM86 kernels, CUDA Graph capture and replay, up to 8 lanes with compact decode batches rebuilt each round |
| Speculation | DFlash2 K=7 (the default 27B profile, one lane); MTP on compatible artifacts |
| KV cache | Paged, quantized formats (`rk4v4` in the measured run), prefix reuse, ReplaySSM for the recurrent state |
| Serving | OpenAI Chat Completions/Responses and Anthropic Messages; JSON Schema, GBNF, regex and tool-call constraints |

---

## Measured

Local qualification at the pinned README (16 October 2026): median decode **116.7 tok/s**, median
first streamed text **127 ms**, six 512-token requests after three warm-ups (range 116.5-117.4 tok/s),
all six greedy outputs identical. Windows, CUDA 13.2, DFlash2 K=7, one lane, 16K context cap, `rk4v4`
KV, thinking and prefix reuse off, a fixed 52-token prompt; cold startup (about 42 s) excluded.

Contributor-reported figures, not re-run here: 140 tok/s single stream with DFlash2 K=7 (eight
thinking-off prompts, 1,024 tokens, greedy, INT8 KV), and v0.16.0 Windows HTTP measurements of
126.3 / 125.6 / 110.2 tok/s after about 1.3K / 8.8K / 29K prompt tokens.

---

## Checks

- **Determinism:** the six greedy outputs of the qualification run matched.
- **Host checks:** CI runs host checks on every push (`host-checks.yml` upstream).

---

## Limits

- A small reproducible smoke benchmark, not a quality evaluation or a matched comparison.
- The published DFlash2 profile caps context at 188,416 tokens on Windows; the MTP profile at 262,144.
- Flash-Next is a Windows preview that needs substantial host RAM and fast storage.

---

## Build and chat

```sh
git clone https://github.com/Don-Chad/ninfer-3090.git
git -C ninfer-3090 checkout d8c29253f2c0845611a794d87057950107a4956b
```

Build and launch as the repository's README and `docs/` describe for Windows (MSVC + CUDA) or Linux;
the [Windows release](https://github.com/Don-Chad/ninfer-3090/releases/tag/v0.16.1-rtx3090) ships a
prebuilt archive with a model download script. Check the artifact against
[card.lock.json](card.lock.json) with `python -B tools/verify_model.py <this engine folder> <model folder>`.

---

## Credits

Apache License 2.0,
[LICENSE](https://github.com/Don-Chad/ninfer-3090/blob/d8c29253f2c0845611a794d87057950107a4956b/LICENSE)
at the pinned commit. A fork of [Neroued/ninfer](https://github.com/Neroued/ninfer); contributors are
listed in its `CONTRIBUTORS.md`, and the 140 tok/s benchmark comes from
[ashalliants](https://github.com/ashalliants/ninfer-3090).
