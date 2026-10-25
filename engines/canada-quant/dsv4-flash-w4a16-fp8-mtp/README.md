<!-- Rendered by tools/card.py from card.yaml: edit card.yaml, then run python -B tools/card.py on this folder. -->

# dsv4-flash-w4a16-fp8-mtp

### 2x-rtxpro6000 · deepseek-v4-flash@W4A16-FP8-MTP, 4x-rtxpro6000 · deepseek-v4-flash@W4A16-FP8-MTP

> [!WARNING]
> **Contributed by** canada-quant · [canada-quant/dsv4-flash-w4a16-fp8-mtp](https://github.com/canada-quant/dsv4-flash-w4a16-fp8-mtp)
>
> External engine: crit-hub has not reviewed its source. Read it before you build or run it.

| GPU · model | Decode (sampled) | Decode (greedy) | Prefill | TTFT | Perplexity | Mean KL divergence | Context |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 2x-rtxpro6000 · deepseek-v4-flash@W4A16-FP8-MTP | – | **95.2 tok/s** | – | – | – | – | **65,536 tok** |
| 4x-rtxpro6000 · deepseek-v4-flash@W4A16-FP8-MTP | – | **108.1 tok/s** | – | – | – | – | **65,536 tok** |

- **Measured with** (2x-rtxpro6000 · deepseek-v4-flash@W4A16-FP8-MTP): self-reported (vllm bench serve, random 256/256) · [self-reported.json](evidence/self-reported.json)
- **Hardware** (2x-rtxpro6000 · deepseek-v4-flash@W4A16-FP8-MTP): 2 of 4 RTX PRO 6000 Blackwell Server Edition 96 GB on a Brev g7e.24xlarge, TP=2
- **Measured with** (4x-rtxpro6000 · deepseek-v4-flash@W4A16-FP8-MTP): self-reported (vllm bench serve, random 256/256) · [self-reported.json](evidence/self-reported.json)
- **Hardware** (4x-rtxpro6000 · deepseek-v4-flash@W4A16-FP8-MTP): 4x RTX PRO 6000 Blackwell Server Edition 96 GB on a Brev g7e.24xlarge, TP=4
- **Requires:** Linux · sm_120a · 192 GB GPU memory · CUDA 12.9
- **Checkpoint** (`deepseek-v4-flash@W4A16-FP8-MTP`): [`canada-quant/DeepSeek-V4-Flash-W4A16-FP8-MTP`](https://huggingface.co/canada-quant/DeepSeek-V4-Flash-W4A16-FP8-MTP/tree/4a4f86303d4fd8c43bafbf02d5523b06275e577d) at `4a4f8630`
- **Source:** [canada-quant/dsv4-flash-w4a16-fp8-mtp @ `3e9cc06a`](https://github.com/canada-quant/dsv4-flash-w4a16-fp8-mtp/tree/3e9cc06a14f6255cdd858944825d7800a98bc95a)
- **License:** MIT

The reproduction and serving repository for canada-quant's DeepSeek-V4-Flash quantization with W4A16
INT4 routed experts, FP8 128x128 block attention and the BF16 Multi-Token Prediction head kept, so
vLLM can self-speculate. It covers the GPTQ calibration on 8x H200, the vLLM patch series and a
Docker image for RTX PRO 6000 Blackwell (SM 12.0) at TP=2 or TP=4.

---

## Design

| Part | How |
| --- | --- |
| Weights | W4A16 GPTQ routed experts (group 128), FP8 block attention, BF16 MTP head retained |
| Speculation | vLLM MTP, 1 draft token (DeepGEMM's paged MQA logits cap it at 1) |
| Engine | jasl/vllm `27fd665b` with the BF16-MTP cherry-pick, Marlin MoE `c_tmp`/workspace patches and native sm_120a Marlin cubins |
| KV cache | fp8, block size 256 |

---

## Measured

RTX PRO 6000 Blackwell Server Edition, fresh Docker image, 2026-05-29, random 256/256, MTP on, one replica:

| Batch size | TP=2 tok/s | TP=2 TPOT p50 | TP=4 tok/s | TP=4 TPOT p50 |
| --- | ---: | ---: | ---: | ---: |
| 1 | **95.2** | 8.05 ms | **108.1** | 7.32 ms |
| 4 | 40.6 | 83.18 ms | 104.3 | 11.31 ms |
| 8 | 45.7 | 79.02 ms | 433.2 | 16.44 ms |

TP=2 with two sequences also loads the model's full 1,048,576-token window (smoke tests only);
prefill at 1M tokens takes about 8-12 minutes.

---

## Checks

- **AIME-2024, thinking-high, c=4:** 27/30 (TP=2) and 29/30 (TP=4), with zero CUDA errors and 91-92% MTP acceptance.
- **GSM8K 8-shot, n=50:** 84-90% (TP=2), 80-86% (TP=4); the model card reports GSM8K 93.71%, MMLU 86.88%, HumanEval 84.76%.

---

## Limits

- The W4A16 Marlin MoE decode path has a kernel race under concurrent thinking-mode load on SM 12.0; the NVFP4 sibling is recommended for batched thinking.
- The artifact's scale tensors use `.weight_scale` naming, which current upstream vLLM rejects; the pinned jasl build loads it.
- Verified on Server Edition only; Workstation Edition is expected to be 5-15% slower.
- The README states MIT, but there is no LICENSE file at the pinned commit.

---

## Build and chat

```sh
git clone https://github.com/canada-quant/dsv4-flash-w4a16-fp8-mtp.git
git -C dsv4-flash-w4a16-fp8-mtp checkout 3e9cc06a14f6255cdd858944825d7800a98bc95a
```

Build the image from `docker/Dockerfile.rtx6000pro` (or pull `canada-quant/dsv4-rtx6000pro:v3`),
download `canada-quant/DeepSeek-V4-Flash-W4A16-FP8-MTP`, check it with
`python -B tools/verify_model.py <this engine folder> <model folder>`, and start
`scripts/serve_rtx6000pro_w4a16.sh` with `TP=2` or `TP=4`; the full recipe is in
[RECIPE_RTX6000PRO.md](https://github.com/canada-quant/dsv4-flash-w4a16-fp8-mtp/blob/3e9cc06a14f6255cdd858944825d7800a98bc95a/RECIPE_RTX6000PRO.md).

---

## Credits

MIT as stated in the README (no LICENSE file at the pinned commit), inherited from
deepseek-ai/DeepSeek-V4-Flash. Builds on jasl's vLLM fork and benchmark harness, kylesayrs's
compressed-tensors V4 attention path and the RedHatAI NVFP4-FP8 recipe topology.
