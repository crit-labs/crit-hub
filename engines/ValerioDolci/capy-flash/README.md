<!-- Rendered by tools/card.py from card.yaml: edit card.yaml, then run python -B tools/card.py on this folder. -->

# capy-flash

### 2x-rtx5070ti · qwen3.8-flash-next@NVFP4

> [!WARNING]
> **Contributed by** ValerioDolci · [ValerioDolci/capy-flash](https://github.com/ValerioDolci/capy-flash)
>
> External engine: crit-hub has not reviewed its source. Read it before you build or run it.

| Decode (sampled) | Decode (greedy) | Prefill | TTFT | Perplexity | Mean KL divergence | Context |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| – | **155 tok/s** | **4,615 tok/s** | **1.48 s** | – | – | **262,144 tok** |

- **Measured with:** self-reported (capy-flash agentic replay) · [self-reported.json](evidence/self-reported.json)
- **Hardware:** Ubuntu; AMD Ryzen 9 9900X; 96 GB DDR5-5600; 2x RTX 5070 Ti 16 GB, PCIe 5.0 x8 + x8, no P2P
- **Requires:** Linux · sm_120a · 32 GB GPU memory · CUDA 13.1
- **Checkpoint** (`qwen3.8-flash-next@NVFP4`): [`Feyd89/Qwen3.8-Flash-Next-NVFP4-capy-flash`](https://huggingface.co/Feyd89/Qwen3.8-Flash-Next-NVFP4-capy-flash/tree/74a0d3ffe75d36508b65c0f650dc030197e28e27) at `74a0d3ff`
- **Source:** [ValerioDolci/capy-flash @ `60a4153b`](https://github.com/ValerioDolci/capy-flash/tree/60a4153be06d33bddc410917c3214e5fb9664cd2)
- **License:** Apache-2.0

capy-flash is the two-GPU tensor-parallel fork of [NInfer](https://github.com/Neroued/ninfer)
(ninfer-tp2, renamed at v0.5.0). It runs Qwen3.8-Flash-Next with NVIDIA's NVFP4 experts as
published on two 16 GB consumer cards: routed experts that do not fit in VRAM live in host RAM and
are computed on the CPU, everything else runs tensor-parallel over PCIe without peer-to-peer. It is
built for agents: idle conversations move to RAM and give their VRAM back to the expert cache, and a
continued conversation keeps the experts its next answer needs.

---

## Design

| Part | How |
| --- | --- |
| Weights | NVIDIA's NVFP4 routed experts as published; indexer BF16, GDN Q8 (`qwen3_8_flash_next_nvfp4_idxbf16_gdnq8`) |
| Parallelism | Tensor parallel on two GPUs through a pinned-host mailbox with FP32 partial sums; direct copies when P2P is available |
| Routed experts | A VRAM cache partitioned by expert id across the GPUs; misses computed on the CPU (AVX-512 NVFP4 kernel, W4A4 activation quantization like the GPU path) inside the captured CUDA graph |
| Speculation | MTP K=3 with a draft vocabulary shortlist; the MTP head re-distilled on the engine's own output (v0.6) |
| KV cache | INT8 with an elastic KV / expert-slot budget up to 262,144 tokens; idle conversations' KV moves to RAM (`--profile flash-next`: 4 GiB) |
| Serving | `ninfer-serve`, OpenAI- and Anthropic-compatible, prefix cache across turns, optional vision (`--vision`) |

---

## Measured

One machine (hardware above), same day against [Strata](https://github.com/Niko1221/Strata) 0.1.45:

| | capy-flash, NVFP4 | Strata, same NVFP4 weights | Strata, IQ3_S |
| --- | ---: | ---: | ---: |
| Agentic replay, total time | **177 s** | 268 s | 197 s |
| Agentic replay, decode (median) | **155 tok/s** | 98 tok/s | 151 tok/s |
| Agentic replay, TTFT median (p90) | **1.48 s (2.62)** | 2.35 s (4.17) | 1.60 s (3.69) |
| Prefill, ~15k-token prompt | **4,615 tok/s** | 3,513 tok/s | 3,811 tok/s |
| LocalMaxxing reasoning-v1 (512 tokens) | 178 tok/s | 119 tok/s | **195 tok/s** |

Decode stays flat with context: 168 / 165 / 171 / 165 tok/s after 0 / 14k / 61k / 118k tokens of
prompt. The same binary serves the dense Qwen3.8-27B (`--profile 27b`): 175 / 326 / 560 / 927 tok/s
aggregate at 1 / 2 / 4 / 8 requests.

---

## Checks

- **Quality** against NVIDIA's NVFP4 checkpoint served by vLLM on a GB300, same prompts, paired:
  GSM8K 0.9750 vs 0.9810 (1,319 questions), MMLU-Pro 0.8413 vs 0.8452 (1,008 questions, non-inferior
  at a 2.5-point margin).
- **Agent tasks:** 20/20 on five multi-step coding tasks (OpenCode, four runs each), no malformed tool
  call in 1,124.
- **Tests:** 187 unit tests (`ctest`); engine integration tests run against synthetic or real artifacts.

---

## Limits

- One request at a time (`--max-concurrency 1`); tested only on the machine above.
- Decode is bound by host-RAM bandwidth for the experts computed on the CPU.
- Output is not bit-reproducible run to run: which experts sit in VRAM depends on timing.
- Needs about 75 GiB of free RAM and an AVX-512 CPU; the server refuses to start when they do not fit.
- Generated tokens per agent task are about 1.6x the GB300 reference.

---

## Build and chat

```sh
git clone https://github.com/ValerioDolci/capy-flash.git
git -C capy-flash checkout 60a4153be06d33bddc410917c3214e5fb9664cd2
cd capy-flash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=120a \
  -DNINFER_BUILD_APPS=ON -DNINFER_BUILD_BENCHMARKS=OFF -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.1/bin/nvcc
ninja -C build apps/ninfer-serve
hf download Feyd89/Qwen3.8-Flash-Next-NVFP4-capy-flash --local-dir models/flash-next
build/apps/ninfer-serve models/flash-next/qwen3_8_flash_next_nvfp4_idxgdnq8.ninfer --profile flash-next --port 8098 --model-id capy-flash
```

Check the download against [card.lock.json](card.lock.json) with
`python -B tools/verify_model.py <this engine folder> models/flash-next` from a crit-hub checkout.
Releases also ship a prebuilt tarball and `install.sh`.

---

## Credits

Apache License 2.0,
[LICENSE](https://github.com/ValerioDolci/capy-flash/blob/60a4153be06d33bddc410917c3214e5fb9664cd2/LICENSE)
and [NOTICE](https://github.com/ValerioDolci/capy-flash/blob/60a4153be06d33bddc410917c3214e5fb9664cd2/NOTICE)
at the pinned commit, which lists the prior work it builds on, starting with
[Neroued/ninfer](https://github.com/Neroued/ninfer). The artifact is under the NVIDIA Open Model
License together with the Qwen Community License of Qwen3.8-Flash-Next.
