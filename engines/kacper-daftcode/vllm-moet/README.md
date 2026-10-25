<!-- Rendered by tools/card.py from card.yaml: edit card.yaml, then run python -B tools/card.py on this folder. -->

# vllm-moet

### 4x-rtxpro6000 · deepseek-v4.1-flash@MXFP4

> [!WARNING]
> **Contributed by** kacper-daftcode · [kacper-daftcode/vLLM-Moet](https://github.com/kacper-daftcode/vLLM-Moet)
>
> External engine: crit-hub has not reviewed its source. Read it before you build or run it.

| Decode (sampled) | Decode (greedy) | Prefill | TTFT | Perplexity | Mean KL divergence | Context |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| – | **190–224 tok/s** | – | – | – | – | **524,288 tok** |

- **Measured with:** self-reported (vLLM-Moet spec_matrix) · [self-reported.json](evidence/self-reported.json)
- **Hardware:** 4x RTX PRO 6000 96 GB, PCIe peer-to-peer, TP4
- **Requires:** Linux · sm_120a · 384 GB GPU memory · CUDA 13.0
- **Checkpoint** (`deepseek-v4.1-flash@MXFP4`): [`deepseek-ai/DeepSeek-V4.1-Flash`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/tree/314113f8a7dadb388215396ba5fbee2c610a04d6) at `314113f8`
- **Source:** [kacper-daftcode/vLLM-Moet @ `213d5fa9`](https://github.com/kacper-daftcode/vLLM-Moet/tree/213d5fa913190521c54e0169068d967526d1fb54)
- **License:** Apache-2.0

vLLM-Moet serves frontier mixture-of-experts models on sm_120 cards where official vLLM images do not
run them. Its active line runs the official DeepSeek-V4.1-Flash checkpoint (MXFP8 dense, MXFP4
experts, 510 GB) on four RTX PRO 6000 with the sm_120 gaps closed and the decode step rebuilt around
small tensor-core kernels: 512K context, DSpark speculation, vision, thinking, tool calls and KV
offload to host RAM and disk. A second line serves 2-bit experts on vLLM v0.24.0.

---

## Design

| Part | How |
| --- | --- |
| Indexer | 128-row pages in DeepGEMM's SM120 paged MQA logits (bit-exact to 64-row pages) |
| KV formats | FP4 compressed record (288 B) and MXFP4 indexer (68 B) on sm_120: 2x the GPU KV tokens |
| Dense decode | Tensor-core MXFP8 GEMV, 1.8-3.4x faster than CUTLASS's 128-row tile |
| MoE decode | 64-row tiles in one fused launch, padding rows dropped from L2 |
| Offload | One shared host copy of KV (896 B per token) plus a disk tier |

---

## Measured

Served image, 4x RTX PRO 6000, 2026-10-16:

| | |
| --- | --- |
| Decode, 1 stream | prose **190-224 tok/s**, code **436-466 tok/s** (88-90 steps/s) |
| Decode, 8 streams | prose 596-615, code 1,494-1,597 tok/s aggregate |
| Fresh prefill | 163K tokens in 8.5 s, 391K in 21.4 s |
| KV cache | 3.69M tokens on the GPUs, 76.7M more in 64 GiB of host RAM |

Against the official image with only the fixes it needs to start: 60 to 88-90 steps/s and 1.49M to
3.69M GPU KV tokens. Qwen3.8-Flash-Next-FP8 on the same cards decodes 243 tok/s prose and 346 code.

---

## Checks

- **Quality:** GSM8K-200 193/200, needle 6/6 to 367K tokens, tool calls, JSON and vision pass.
- **Exactness:** the changes are bit-identical to the paths they replace except the dense GEMVs (within one bf16 rounding) and the two KV formats, which passed a quality gate.

---

## Limits

- Needs four RTX PRO 6000; an RTX 5090 does not fit the model at TP4.
- With the disk tier each request pays ~20 ms more before its first token (cause open).
- First start takes ~22 minutes (FlashInfer autotune).

---

## Build and chat

```sh
git clone https://github.com/kacper-daftcode/vLLM-Moet && cd vLLM-Moet
git checkout 213d5fa913190521c54e0169068d967526d1fb54
export IMAGE=vllm-moet-sm120:dsv41-nightly-49cb2bbf-r1 MODEL_DIR=/srv/models/DeepSeek-V4.1-Flash CACHE_DIR=/srv/cache/ds41
DOCKER_BUILDKIT=1 docker build -f Dockerfile.sm120-dsv41-nightly -t $IMAGE .
GPU_MEM_UTIL=0.92 docker/sm120/run-dsv41.sh
```

Download `deepseek-ai/DeepSeek-V4.1-Flash` into `MODEL_DIR` first and check it with
`python -B tools/verify_model.py <this engine folder> <model folder>`. The API listens on
`127.0.0.1:8001`; every knob is in
[docs/sm120-deploy.md](https://github.com/kacper-daftcode/vLLM-Moet/blob/213d5fa913190521c54e0169068d967526d1fb54/docs/sm120-deploy.md).

---

## Credits

Apache License 2.0,
[LICENSE](https://github.com/kacper-daftcode/vLLM-Moet/blob/213d5fa913190521c54e0169068d967526d1fb54/LICENSE)
at the pinned commit; third-party components in NOTICE. Built on the vLLM project, FlashInfer and
DeepGEMM; the checkpoint is DeepSeek-AI's.
