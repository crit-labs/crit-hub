<!-- Rendered by tools/card.py from card.yaml: edit card.yaml, then run python -B tools/card.py on this folder. -->

# ninfer

### rtx5090 · qwen3.8-27b@NVFP4, rtx5090 · qwen3.8-27b@groupwise-int, rtx5090 · qwen3.6-35b-a3b@groupwise-int

> [!WARNING]
> **Contributed by** Neroued · [Neroued/ninfer](https://github.com/Neroued/ninfer)
>
> External engine: crit-hub has not reviewed its source. Read it before you build or run it.

| GPU · model | Decode (sampled) | Decode (greedy) | Prefill | TTFT | Perplexity | Mean KL divergence | Context |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| rtx5090 · qwen3.8-27b@NVFP4 | – | **147.7 tok/s** | **12,819.1 tok/s** | – | – | – | **260,096 tok** |
| rtx5090 · qwen3.8-27b@groupwise-int | – | **136.5 tok/s** | **3,331.9 tok/s** | – | – | – | **260,096 tok** |
| rtx5090 · qwen3.6-35b-a3b@groupwise-int | – | **642.5 tok/s** | **17,705.4 tok/s** | – | – | – | **260,096 tok** |

- **Measured with** (rtx5090 · qwen3.8-27b@NVFP4): self-reported (NInfer performance suite) · [self-reported.json](evidence/self-reported.json)
- **Hardware** (rtx5090 · qwen3.8-27b@NVFP4): Linux; RTX 5090 32 GB
- **Measured with** (rtx5090 · qwen3.8-27b@groupwise-int): self-reported (NInfer performance suite) · [self-reported.json](evidence/self-reported.json)
- **Hardware** (rtx5090 · qwen3.8-27b@groupwise-int): Linux; RTX 5090 32 GB
- **Measured with** (rtx5090 · qwen3.6-35b-a3b@groupwise-int): self-reported (NInfer performance suite) · [self-reported.json](evidence/self-reported.json)
- **Hardware** (rtx5090 · qwen3.6-35b-a3b@groupwise-int): Linux; RTX 5090 32 GB
- **Requires:** Linux · sm_120a · 32 GB GPU memory · CUDA 13.4
- **Checkpoint** (`qwen3.8-27b@NVFP4`): [`neroued/Qwen3.8-27B-nvfp4-NInfer`](https://huggingface.co/neroued/Qwen3.8-27B-nvfp4-NInfer/tree/a107ba1b5b0609d9ca90d5ed61439f5f5cc7d64d) at `a107ba1b`
- **Checkpoint** (`qwen3.8-27b@groupwise-int`): [`neroued/Qwen3.8-27B-NInfer`](https://huggingface.co/neroued/Qwen3.8-27B-NInfer/tree/1cbd84e7221e51186bd7f093a149912d2489625b) at `1cbd84e7`
- **Checkpoint** (`qwen3.6-35b-a3b@groupwise-int`): [`neroued/Qwen3.6-35B-A3B-NInfer`](https://huggingface.co/neroued/Qwen3.6-35B-A3B-NInfer/tree/ee4495803bc4f8015b8a7e22d4cf9b67de8e27c6) at `ee449580`
- **Checkpoint** (`qwen3.6-27b@NVFP4`): [`neroued/Qwen3.6-27B-nvfp4-NInfer`](https://huggingface.co/neroued/Qwen3.6-27B-nvfp4-NInfer/tree/284521801f80d9c3f4a3f8389e2c68048ba90edb) at `28452180`
- **Checkpoint** (`qwen3.6-27b@groupwise-int`): [`neroued/Qwen3.6-27B-NInfer`](https://huggingface.co/neroued/Qwen3.6-27B-NInfer/tree/3e3d9a3951c452c1ca80bd7a2860c7f3bfc5a829) at `3e3d9a39`
- **Source:** [Neroued/ninfer @ `64d10fbf`](https://github.com/Neroued/ninfer/tree/64d10fbf9f8e6bfc79ab0570b94674c5b9997929)
- **License:** Apache-2.0

NInfer is a from-scratch C++/CUDA inference engine for a single NVIDIA GeForce RTX 5090. It serves
Qwen3.5/3.6 dense and MoE models and Qwen4Exp (Qwen3.8-Flash-Next) with text, image and video prompts
and native MTP, from official v3 `.ninfer` artifacts that carry the model's configuration, encoded
weights and frontend resources. One GPU, one resident model and one to eight execution lanes fixed at
startup. Several engines in this registry (NInfer-3090, NInfer-4090, NInfer V100X2, NInfer 6000,
capy-flash, Infernix, VoidInfer and giveen's ninfer-ext) started from it.

---

## Design

| Part | How |
| --- | --- |
| Weights | Official artifacts in `groupwise-int` or `nvfp4`; your own conversions through the recipes in `docs/weight-conversion.md` |
| Execution | One GPU, one resident model, 1–8 lanes fixed at startup, CUDA Graphs |
| Speculation | Native MTP (`--spec mtp --draft-tokens 3`), with an optional smaller quantized proposal head (`--lm-head-draft`) |
| KV cache | FP8 E4M3 row-256 for Qwen3.8 and INT8 group-64 for Qwen3.6 in the published runs; a shared device KV pool with pause and resume under pressure |
| Context reuse | Checkpoints of KV plus the complete continuation state at exact token frontiers, retained across turns in device and pinned host memory |
| Serving | Local CLI and OpenAI- and Anthropic-compatible HTTP APIs |

---

## Measured

From the README at the pinned commit (excerpts of the [performance index](https://github.com/Neroued/ninfer/blob/64d10fbf9f8e6bfc79ab0570b94674c5b9997929/docs/performance.md)), one RTX 5090:

| Model profile | Decode, C=1 / C=8, MTP3 | Prefill, 7,680 / 260,096 tokens | Structured MTP3 decode |
| --- | ---: | ---: | ---: |
| Qwen3.8-27B `nvfp4` | 147.7 / 922.4 tok/s | 12,819.1 / 4,016.4 tok/s | 231.7 tok/s |
| Qwen3.8-27B `groupwise-int` | 136.5 / 582.4 tok/s | 3,331.9 / 2,139.4 tok/s | 214.7 tok/s |
| Qwen3.6-35B-A3B `groupwise-int` | 642.5 / 1,380.7 tok/s | 17,705.4 / 5,247.0 tok/s | 779.6 tok/s |

Decode is the saturated MTP3 suite: one 8,192-token generation per active request, aggregate
committed tokens over intervals whose decode batch equaled the concurrency. Prefill is the serial
serving corpus (1,024-token chunks, five seeds after warm-up). Qwen3.6-27B (`groupwise-int` and
`nvfp4`) has the same tables upstream and is listed as supported here.

---

## Checks

- **Capability:** AIME 2025/2026, GPQA-Diamond, ERQA and RealWorldQA through the serving route with
  thinking on and MTP3 (EvalScope 1.9.0); Qwen3.8-27B `nvfp4` scores 96.67% / 96.67% / 90.40% on the
  first three.
- **Method:** the measurement rules are in `docs/performance/methodology.md` upstream; this entry does
  not re-run them.

---

## Limits

- Linux on x86-64 and an RTX 5090 only: the build rejects CUDA architectures other than `sm_120a`.
- v3 artifacts only; older v2 downloads must be upgraded locally first.
- No install target or packaged binaries; run from the build tree.

---

## Build and chat

```sh
git clone https://github.com/Neroued/ninfer.git
git -C ninfer checkout 64d10fbf9f8e6bfc79ab0570b94674c5b9997929
cd ninfer
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
hf download neroued/Qwen3.8-27B-nvfp4-NInfer qwen3_8_27b_nvfp4.ninfer --local-dir models
```

Check a download against [card.lock.json](card.lock.json) with
`python -B tools/verify_model.py <this engine folder> <model folder>` from a crit-hub checkout. Serve:

```sh
./build/apps/ninfer-serve models/qwen3_8_27b_nvfp4.ninfer --max-context 240000 --kv-capacity 240000 \
  --max-concurrency 2 --kv-dtype fp8 --device-state-slots 2 --spec mtp --draft-tokens 3 \
  --lm-head-draft --preserve-thinking
```

The server listens on `http://127.0.0.1:8080` (`/v1/chat/completions`). One-shot chat:
`./build/apps/ninfer models/qwen3_8_27b_nvfp4.ninfer --prompt "..." --max-context 32768 --kv-dtype fp8 --spec mtp --draft-tokens 3 --lm-head-draft`.

---

## Credits

NInfer is licensed under the Apache License 2.0,
[LICENSE](https://github.com/Neroued/ninfer/blob/64d10fbf9f8e6bfc79ab0570b94674c5b9997929/LICENSE) at the
pinned commit. The model weights keep their own licenses, given on each artifact's Hugging Face card.
