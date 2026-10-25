<!-- Rendered by tools/card.py from card.yaml: edit card.yaml, then run python -B tools/card.py on this folder. -->

# qwentin

### rtx5090 · qwen3.8-27b@EXL3-5.75bpw

> [!WARNING]
> **Contributed by** kacper-daftcode · [kacper-daftcode/qwentin](https://github.com/kacper-daftcode/qwentin)
>
> External engine: crit-hub has not reviewed its source. Read it before you build or run it.

| Decode (sampled) | Decode (greedy) | Prefill | TTFT | Perplexity | Mean KL divergence | Context |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| – | **177 tok/s** | – | **11.6 s** | – | **0.0027** | **262,144 tok** |

- **Measured with:** self-reported (qwentin gate logs) · [self-reported.json](evidence/self-reported.json)
- **Hardware:** RTX 5090 32 GB at 550 W; CUDA 13.1
- **Requires:** Linux · sm_120 · 32 GB GPU memory · CUDA 13.1
- **Checkpoint** (`qwen3.8-27b@EXL3-5.75bpw`): [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B/tree/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0) at `1d4bf0f2`
- **Source:** [kacper-daftcode/qwentin @ `6a15ea39`](https://github.com/kacper-daftcode/qwentin/tree/6a15ea399c2ebd9408268de9d3a20e9ecee108a8)
- **License:** MIT

qwentin is a from-scratch CUDA inference engine for the 27B hybrid-attention Qwen3.6/3.8 model on
one consumer Blackwell GPU: hand-written tensor-core kernels behind an OpenAI-compatible server with
paged continuous batching. Its memory comes from EXL3 trellis-quantized weights at 5.75 bpw (made
with turboderp's exllamav3), its own 4-bit trellis-coded KV cache (16.5 KiB per token) and the
model's own MTP head as the drafter, which together pool 576K tokens of context across 12 sessions.
It was built by AI coding agents directed by the maintainer, and every number is backed by a log in
`docs/evidence/`.

---

## Design

| Part | How |
| --- | --- |
| Weights | EXL3 5.75 bpw, quantized from Qwen's BF16 checkpoint with the served recipe, packed into a `.tqf` file |
| KV cache | 4-bit trellis-coded K and V, 16.5 KiB per token; 12 slots and 576K pooled tokens in 29,894 MiB |
| Speculation | The model's MTP head on every stream |
| Serving | `tools/serve_batched.py`, OpenAI-compatible, paged continuous batching, `/health` with pool and spec stats |

---

## Measured

One RTX 5090, v0.11 (2026-10-20), against vLLM on the same cards:

| | qwentin, 1x RTX 5090 | vLLM NVFP4, 1x RTX 5090 (bf16 ; fp8 KV) | vLLM FP8, 2x RTX 5090 |
| --- | --- | --- | --- |
| Pooled KV capacity | **576K tokens** | 147-179K ; 283K | 493K |
| KL / top-1 vs BF16 | **0.0022-0.0027 / 99.0-99.1%** | about 0.04 KL | 0.0037 / 98.7% |
| Decode per stream @32K, N = 1 / 4 / 8 | **177 / 106 / 79 tok/s** | 71.5 / 51.4 / – ; 75.3 / 58.8 / – | 76.8 / 60.3 / 50.4 |
| Decode per stream @128K, N = 1 | **115 tok/s** | 55.5 ; 66.0 | 65.8 |
| Cold TTFT, 32K / 128K | 11.6 / 59.7 s | **3.1 / 25.2 s** ; 3.3 / 30.0 s | **4.4 / 24.1 s** |

Each spec-on cell is one prompt, and its speed follows the accepted draft length (3.78 tokens per
round at 32K, N = 1). `bench/reproduce.sh` reads 176 / 79 tok/s at N = 1 / 8 with the vLLM checkout
as corpus.

---

## Checks

- **Quality:** KL and top-1 against BF16 (`docs/QUALITY.md`); EvalPlus equal to FP8.
- **Retrieval:** needle in a repository 6/6 at 32K and 128K.
- **Reproduction:** `bench/reproduce.sh` builds, checks the model, runs the served line, needle,
  spec A/B and speed, and writes a results file.

---

## Limits

- Cold time to first token is slower than vLLM's (11.6 s against 3.1 s at 32K).
- No converted weights are published: conversion takes about 40 minutes on one 5090.
- Spec-on speeds are single prompts and move with the accepted draft length.

---

## Build and chat

```sh
git clone https://github.com/kacper-daftcode/qwentin.git
git -C qwentin checkout 6a15ea399c2ebd9408268de9d3a20e9ecee108a8
cd qwentin
cmake -S . -B build-qwen -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.1/bin/nvcc && make -C build-qwen -j8
pip install -r requirements.txt
```

Check the BF16 checkpoint with `python -B tools/verify_model.py <this engine folder> /path/to/Qwen3.8-27B`,
convert it with exllamav3 and `tools/convert_qwen_tqf.py` as the README shows, then serve:

```sh
CUDA_VISIBLE_DEVICES=0 TQ_CTX=262144 TQ_KV_Q4=1 TQ_EMBED_FP8=2 TQ_PAGED_ATTN_MMA=1 python3 -u tools/serve_batched.py \
  --port 8100 --max-slots 12 --num-blocks 4500 --timeout 1800 --model-name qwen3.8-27b \
  --tqf /path/to/qwen3_8-27b-exl3-opt5.75-mtp4.tqf --model-dir /path/to/Qwen3.8-27B --lib build-qwen/libforward_qwen.so
```

---

## Credits

MIT, [LICENSE](https://github.com/kacper-daftcode/qwentin/blob/6a15ea399c2ebd9408268de9d3a20e9ecee108a8/LICENSE)
at the pinned commit. The quality numbers stand on turboderp's
[exllamav3](https://github.com/turboderp-org/exllamav3) EXL3 quantizer.
