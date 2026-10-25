<!-- Rendered by tools/card.py from card.yaml: edit card.yaml, then run python -B tools/card.py on this folder. -->

# dsv4-flash-acti-mtp

### 2x-rtxpro6000 · deepseek-v4-flash@W4A16-FP8-Acti-MTP

> [!WARNING]
> **Contributed by** LordNeel · [LordNeel/DeepSeek-V4-Flash-Acti-MTP-W4A16-FP8](https://huggingface.co/LordNeel/DeepSeek-V4-Flash-Acti-MTP-W4A16-FP8)
>
> External engine: crit-hub has not reviewed its source. Read it before you build or run it.

| Decode (sampled) | Decode (greedy) | Prefill | TTFT | Perplexity | Mean KL divergence | Context |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| – | **85.52 tok/s** | – | **155 ms** | – | – | **524,288 tok** |

- **Measured with:** self-reported (Acti single-stream decode bench) · [self-reported.json](evidence/self-reported.json)
- **Hardware:** 2x RTX PRO 6000 Blackwell Max-Q Workstation Edition 96 GB, PCIe only (no NVLink); driver 580.126.09; NCCL 2.28.9
- **Requires:** Linux · sm_120a · 192 GB GPU memory · CUDA 12.9
- **Checkpoint** (`deepseek-v4-flash@W4A16-FP8-Acti-MTP`): [`LordNeel/DeepSeek-V4-Flash-Acti-MTP-W4A16-FP8`](https://huggingface.co/LordNeel/DeepSeek-V4-Flash-Acti-MTP-W4A16-FP8/tree/f5619820ba471ce89f34033ffb177d8c037ad3d9) at `f5619820`
- **Source:** [LordNeel/DeepSeek-V4-Flash-Acti-MTP-W4A16-FP8 @ `f5619820`](https://huggingface.co/LordNeel/DeepSeek-V4-Flash-Acti-MTP-W4A16-FP8/tree/f5619820ba471ce89f34033ffb177d8c037ad3d9)
- **License:** Apache-2.0

Acti's MTP-enabled variant of pasta-paul's DeepSeek-V4-Flash W4A16-FP8 quantization. The original
quantization dropped the model's Multi-Token Prediction head; Acti added it back with GPTQ-quantized
routed experts and patched vLLM's DeepSeek-V4 MTP loader, so vLLM can self-speculate one token per
step at TP=2. The Hugging Face repository holds the weights and a Docker build of the patched vLLM.

---

## Design

| Part | How |
| --- | --- |
| Main model | pasta-paul's W4A16 GPTQ experts and FP8_BLOCK attention, unchanged |
| MTP head | 256 routed experts GPTQ-quantized to W4A16 from activations captured on a live server (v2); attention FP8_BLOCK; new shard `model-mtp-w4a16.safetensors` |
| Engine | jasl/vllm `b158e5001` plus three MTP-loader patches (about 30 lines in `deepseek_v4_mtp.py`) |
| Speculation | vLLM MTP, 1 token (the model ships one MTP layer) |

---

## Measured

2x RTX PRO 6000 Max-Q, PCIe only, single stream, greedy:

| Profile | Decode | TTFT |
| --- | ---: | ---: |
| Base model without MTP, 524K | 52.85 tok/s | 91 ms |
| **This model, 524K** | **85.52 tok/s** | 155 ms |
| This model, 128K | ~111 tok/s | ~310 ms |

MTP acceptance measured from vLLM's counters: 77.7% (chat, 128K) to 84.9% (chat, 262K). A 15-prompt
real-world suite averaged 84.48 tok/s at 524K; the 128K profile serves 8 users at 467 tok/s aggregate.

---

## Checks

- **Output:** every accepted draft is verified by the full model, so greedy output matches decoding without MTP.
- **Benchmarks:** GSM8K 93/100, HumanEval 158/164 (full set, executed tests), MMLU 53/100 (mixed subjects, 128 max tokens).

---

## Limits

- Validated only on 2x RTX PRO 6000 Max-Q; without `--disable-custom-all-reduce` the pair deadlocks at warmup.
- TP=2 only: TP=1 runs out of memory, TP of 4 or more hits vllm#41511.
- The MTP calibration skipped attention (identity residual), which lowers acceptance, not output quality.
- No measurement date is given.

---

## Build and chat

```sh
git clone https://huggingface.co/LordNeel/DeepSeek-V4-Flash-Acti-MTP-W4A16-FP8 dsv4
git -C dsv4 checkout f5619820ba471ce89f34033ffb177d8c037ad3d9
docker build -t dsv4-flash-acti-mtp:0.1.0 dsv4/docker
```

The clone brings the weights too (use `GIT_LFS_SKIP_SMUDGE=1` and download them separately to keep
the build context small). Check the weights with
`python -B tools/verify_model.py <this engine folder> <model folder>`, mount them into the image and
run it; the 524K profile is the default.

---

## Credits

Apache License 2.0, as the repository's model card declares (no separate LICENSE file). DeepSeek-AI's
DeepSeek-V4-Flash, pasta-paul's W4A16 GPTQ quantization and jasl/vllm serving stack, and Acti's MTP
requantization and loader patches.
