<!-- Rendered by tools/card.py from card.yaml: edit card.yaml, then run python -B tools/card.py on this folder. -->

# infernix

### rtx5090 · qwen3.8-flash-next@NVFP4-Dense8

> [!WARNING]
> **Contributed by** Wallawalla47 · [Wallawalla47/Infernix](https://github.com/Wallawalla47/Infernix)
>
> External engine: crit-hub has not reviewed its source. Read it before you build or run it.

| Decode (sampled) | Decode (greedy) | Prefill | TTFT | Perplexity | Mean KL divergence | Context |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| – | **157.2 tok/s** | – | **2.29 s** | – | – | **262,144 tok** |

- **Measured with:** self-reported (Infernix context-length benchmark) · [self-reported.json](evidence/self-reported.json)
- **Hardware:** Windows 11; Intel Core i9-13900K; 96 GB DDR5; RTX 5090 32 GB, PCIe 5.0 x8
- **Requires:** Windows, Linux, WSL2 · sm_120a · 32 GB GPU memory · CUDA 13.4
- **Checkpoint** (`qwen3.8-flash-next@NVFP4-Dense8`): [`Wallawalla47/Qwen3.8-Flash-Next-NVIDIA-NVFP4-Dense8-Infernix`](https://huggingface.co/Wallawalla47/Qwen3.8-Flash-Next-NVIDIA-NVFP4-Dense8-Infernix/tree/bb9a48f68c4a183cf04be9ef970be25e608c0c64) at `bb9a48f6`
- **Checkpoint** (`qwen3.8-27b@NVFP4`): [`Wallawalla47/Qwen3.8-27B-NVIDIA-NVFP4-NInferV3`](https://huggingface.co/Wallawalla47/Qwen3.8-27B-NVIDIA-NVFP4-NInferV3/tree/016955102d4974784351eea8f9975fc9a8a01b18) at `01695510`
- **Source:** [Wallawalla47/Infernix @ `7dbe99a3`](https://github.com/Wallawalla47/Infernix/tree/7dbe99a3ef7cf334142ad0bb6da5a8aacb06af84)
- **License:** Apache-2.0

Infernix is a C++/CUDA engine that serves Qwen models on one RTX 50-series GPU, tuned on the RTX
5090. It began as a fork of [NInfer](https://github.com/Neroued/ninfer) and reads `.ninfer`
artifacts as well as its own `.infernix` files. For Qwen3.8-Flash-Next it keeps the 24,576 routed
experts in pinned system RAM, caches the hot ones in VRAM, computes part of each layer's misses on
the CPU while the rest cross PCIe, and reads the n-gram table from NVMe. The author notes that most
of what was added since the fork, and most of its README, was written with AI.

---

## Design

| Part | How |
| --- | --- |
| Weights | NVIDIA's NVFP4 checkpoint in the Dense8 conversion (8-bit dense weights); IQ3_S GGUF, FP8 and uncensored conversions also published |
| Routed experts | Pinned host RAM, a VRAM cache of hot experts, part of each layer's misses computed on the CPU; an optional SSD tier |
| Speculation | MTP, DFlash, DFlash2 (a separate 776 MB drafter) and n-gram drafting, with exact sampling |
| KV cache | `int8`, `bf16`, `fp8`, `nvfp4`, `k8v4`, `vq2`, `k4v2`; a hybrid prefix cache in GPU, host RAM and a file that survives restarts |
| Serving | OpenAI Chat Completions and Responses, Anthropic Messages, constrained decoding, `/v1/decide`, 1-8 concurrent requests |

---

## Measured

Release `windows-2026.10.18` on the machine above, 2026-10-19, one request at a time, greedy, 1,024
output tokens, MTP drafter, INT8 KV at 262,144 tokens:

| Context | TTFT: Infernix | Strata 0.1.45 (UD-Q4_K_XL) | strata-nvfp4 | Decode: Infernix | Strata | strata-nvfp4 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ~8K | **2.29 s** | 10.51 s | 2.54 s | **157.2** | 103.9 | 143.6 |
| ~128K | **12.58 s** | 93.59 s | 15.22 s | **142.7** | 92.1 | 134.9 |
| ~250K | **25.17 s** | 139.64 s | 31.22 s | 155.2 | 93.0 | **161.0** |

In an agentic replay (three coding-agent sessions and eleven subagents, two requests at once) the
average time to first token was 4.9 s against 17.0 s (Strata) and 10.7 s (strata-nvfp4), with 84.9%
of prompt tokens served from cache. Each engine ran its own quantization and drafter.

---

## Checks

- **Quality:** perplexity 4.654 against Strata's 4.844 on the same token ids, as the README reports.
- **Resilience:** a failed expert read on Flash-Next fails only the requests running at the time.

---

## Limits

- Measured at PCIe 5.0 x8; expert-cache misses cross PCIe on every token, so the link sets part of the speed.
- Hobby development; the author warns the README may be incomplete or inaccurate.
- Qwen3.8-27B is listed as supported; its README figures are relative to NInfer, not absolute.

---

## Build and chat

```sh
git clone https://github.com/Wallawalla47/Infernix.git
git -C Infernix checkout 7dbe99a3ef7cf334142ad0bb6da5a8aacb06af84
```

Build natively with MSVC or GCC, or take the
[windows-2026.10.18 release](https://github.com/Wallawalla47/Infernix/releases/tag/windows-2026.10.18)
that was measured; setup and flags are in the
[Qwen3.8-Flash-Next guide](https://github.com/Wallawalla47/Infernix/blob/7dbe99a3ef7cf334142ad0bb6da5a8aacb06af84/docs/qwen3_8-flash-next.md).
Check a download with `python -B tools/verify_model.py <this engine folder> <model folder>`.

---

## Credits

Apache License 2.0,
[LICENSE](https://github.com/Wallawalla47/Infernix/blob/7dbe99a3ef7cf334142ad0bb6da5a8aacb06af84/LICENSE)
at the pinned commit. A fork of [Neroued/ninfer](https://github.com/Neroued/ninfer); the DFlash2
drafter is Inco AI's, and the weights keep their own licenses.
