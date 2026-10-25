<!-- Rendered by tools/card.py from card.yaml: edit card.yaml, then run python -B tools/card.py on this folder. -->

# dsv4-flash-w4a16-fp8

### 2x-dgx-spark · deepseek-v4-flash@W4A16-FP8, 2x-rtxpro6000 · deepseek-v4-flash@W4A16-FP8

> [!WARNING]
> **Contributed by** canada-quant · [canada-quant/dsv4-flash-w4a16-fp8](https://github.com/canada-quant/dsv4-flash-w4a16-fp8)
>
> External engine: crit-hub has not reviewed its source. Read it before you build or run it.

| GPU · model | Decode (sampled) | Decode (greedy) | Prefill | TTFT | Perplexity | Mean KL divergence | Context |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 2x-dgx-spark · deepseek-v4-flash@W4A16-FP8 | – | **14–17 tok/s** | – | – | – | – | **1,048,576 tok** |
| 2x-rtxpro6000 · deepseek-v4-flash@W4A16-FP8 | – | **47.5 tok/s** | – | – | – | – | – |

- **Measured with** (2x-dgx-spark · deepseek-v4-flash@W4A16-FP8): self-reported (canada-quant dual-Spark bench) · [self-reported.json](evidence/self-reported.json)
- **Hardware** (2x-dgx-spark · deepseek-v4-flash@W4A16-FP8): 2x DGX Spark (GB10, 128 GB unified memory each) over QSFP, TP=2
- **Measured with** (2x-rtxpro6000 · deepseek-v4-flash@W4A16-FP8): self-reported (canada-quant RTX PRO 6000 validation) · [self-reported.json](evidence/self-reported.json)
- **Hardware** (2x-rtxpro6000 · deepseek-v4-flash@W4A16-FP8): 2x RTX PRO 6000 Blackwell 96 GB, TP=2
- **Requires:** Linux · sm_121a, sm_120a, sm_90a · 192 GB GPU memory · CUDA 13.0
- **Checkpoint** (`deepseek-v4-flash@W4A16-FP8`): [`canada-quant/DeepSeek-V4-Flash-W4A16-FP8`](https://huggingface.co/canada-quant/DeepSeek-V4-Flash-W4A16-FP8/tree/087ae8abe03d7f8961e8adcb9d59206d3ce8377e) at `087ae8ab`
- **Source:** [canada-quant/dsv4-flash-w4a16-fp8 @ `6abcceeb`](https://github.com/canada-quant/dsv4-flash-w4a16-fp8/tree/6abcceeb055608b7f08ff593e2e38ffc753c97eb)
- **License:** MIT

The reproduction repository for canada-quant's DeepSeek-V4-Flash quantization with W4A16 GPTQ routed
experts and FP8 block attention, which serves in vLLM at TP=2 on H200 (SM 9.0a), DGX Spark GB10 (SM
12.1a) and RTX PRO 6000 (SM 12.0). The headline path is a single bootstrap script that brings up two
DGX Sparks as one TP=2 server with the model's full 1M-token context.

---

## Design

| Part | How |
| --- | --- |
| Weights | W4A16 GPTQ routed experts, FP8_BLOCK attention (RedHat's topology with NVFP4 swapped for W4A16) |
| Engine | jasl/vllm SM12x DeepSeek-V4 branch with the vendored kylesayrs deepseek-ct patch and a `packed_modules_mapping` patch |
| Dual Spark | `scripts/bootstrap_dsv4_spark.sh`: QSFP /30 setup, image build, head and worker containers, health wait |
| KV cache | fp8, block size 256; 1M context with one sequence |

---

## Measured

| Test | 2x DGX Spark TP=2 | 2x RTX PRO 6000 TP=2 |
| --- | --- | --- |
| Decode, batch size 1 (1,024 in / 1,024 out) | 14-17 tok/s | 47.5 tok/s (TPOT 20.8 ms) |
| GSM8K 8-shot strict-match | 95.45% | 95.07% |
| HumanEval pass@1 | 80.49% | 78.05% |
| NIAH 75K to 500K, single | 4/4 | 5/5 |

On two Sparks aggregate decode scales from 12 tok/s at one sequence to 125 tok/s at sixteen (16-128K
contexts); 1M tokens with one sequence fits, 1.5M does not.

---

## Checks

- **Tools:** toolcall15 41/45 (dual Spark) and 27/30 (RTX PRO 6000); chat-smoke passes on every SKU.
- **Reasoning:** AIME-2024 think-max 4/5 with no truncations at 65,536 max tokens (dual Spark).

---

## Limits

- The published artifact is load-blocked on current upstream vLLM (FP8 compressor loading, `.weight_scale` naming, a missing gate bias); pin `jasl/vllm@428e08e` or `abad5dc71`.
- TP=1 runs out of memory on one 141 GB H200; TP of 4 or more hits the Marlin MoE scale-sharding bug (vllm#41511).
- The RTX PRO 6000 figures are from a 2026-05-05 build; 1M context is dual-Spark only.

---

## Build and chat

```sh
git clone https://github.com/canada-quant/dsv4-flash-w4a16-fp8.git
git -C dsv4-flash-w4a16-fp8 checkout 6abcceeb055608b7f08ff593e2e38ffc753c97eb
./dsv4-flash-w4a16-fp8/scripts/bootstrap_dsv4_spark.sh --head-host spark-a --worker-host spark-b
```

The walk-through is
[findings/QUICKSTART_DUAL_SPARK.md](https://github.com/canada-quant/dsv4-flash-w4a16-fp8/blob/6abcceeb055608b7f08ff593e2e38ffc753c97eb/findings/QUICKSTART_DUAL_SPARK.md);
other TP=2 hardware builds jasl's branch with the two patches as in the README. Check the download
with `python -B tools/verify_model.py <this engine folder> <model folder>`.

---

## Credits

MIT, [LICENSE](https://github.com/canada-quant/dsv4-flash-w4a16-fp8/blob/6abcceeb055608b7f08ff593e2e38ffc753c97eb/LICENSE)
at the pinned commit, inherited from deepseek-ai/DeepSeek-V4-Flash. Credits jasl (SM12x DeepSeek-V4
support), kylesayrs (compressed-tensors V4 attention), aabbccddwasd, bbbearxyz, wuwenthink and
RedHatAI's NVFP4-FP8 reference.
