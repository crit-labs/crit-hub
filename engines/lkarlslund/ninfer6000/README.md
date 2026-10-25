<!-- Rendered by tools/card.py from card.yaml: edit card.yaml, then run python -B tools/card.py on this folder. -->

# ninfer6000

### rtxpro6000 · swift-1.5-qwen3.8-flash-next@NVFP4

> [!WARNING]
> **Contributed by** lkarlslund · [lkarlslund/ninfer6000](https://github.com/lkarlslund/ninfer6000)
>
> External engine: crit-hub has not reviewed its source. Read it before you build or run it.

| Decode (sampled) | Decode (greedy) | Prefill | TTFT | Perplexity | Mean KL divergence | Context |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| – | **401.3 tok/s** | **13,908 tok/s** | – | – | – | **262,144 tok** |

- **Measured with:** self-reported (ninfer_bench) · [self-reported.json](evidence/self-reported.json)
- **Hardware:** Linux; RTX PRO 6000 Blackwell Workstation Edition 96 GB, 600 W limit
- **Requires:** Linux · sm_120a · 96 GB GPU memory · CUDA 13.0
- **Checkpoint** (`swift-1.5-qwen3.8-flash-next@NVFP4`): [`ukisai/Swift-1.5-Qwen3.8-Flash-Next-NVFP4`](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-Flash-Next-NVFP4/tree/3ff0520224f264a2d0ac4ab56ece8f2f13aadb38) at `3ff05202`
- **Checkpoint** (`qwen3.8-flash-next@NVFP4`): [`RadixArk/Qwen3.8-Flash-Next-NVFP4`](https://huggingface.co/RadixArk/Qwen3.8-Flash-Next-NVFP4/tree/7b719225242aacd3dbd3f9407468c2ee9a9d2594) at `7b719225`
- **Source:** [lkarlslund/ninfer6000 @ `613d5c4b`](https://github.com/lkarlslund/ninfer6000/tree/613d5c4b73080823949ec61b9c5f2b6ec050cd01)
- **License:** Apache-2.0

NInfer 6000 runs Qwen3.8-Flash-Next (125B-A6B) on one RTX PRO 6000 Blackwell with every expert in
VRAM. It is a fork of [NInfer](https://github.com/Neroued/ninfer) that adds an 8-bit weight option,
an 8-bit PLE table option and decode and MTP kernel work for Flash-Next. The measured model is
Swift 1.5, a Qwen3.8-Flash-Next derivative; NVIDIA-format checkpoints from RadixArk convert with the
same tool.

---

## Design

| Part | How |
| --- | --- |
| Weights | NVFP4 routed experts; the 8-bit variant stores 510 non-expert matrices as FP8 E4M3 with a BF16 scale per row |
| PLE table | Read directly from the artifact file; 8-bit FP8 (51 GB) in the measured variant, kept in the page cache when RAM allows |
| Speculation | MTP3, optionally with a smaller quantized proposal head (`--lm-head-draft`, +4-15%) |
| Execution | One request in the benchmarks, BF16 KV, CUDA Graphs, 8,192-token prefill chunks |

---

## Measured

One request, BF16 KV, greedy, 600 W, Swift 1.5 (tok/s):

| Context | No MTP, 16-bit / 8-bit | MTP3, 16-bit / 8-bit | MTP3 + head, 16-bit / 8-bit |
| ---: | ---: | ---: | ---: |
| 512 | 118.0 / 172.0 | 171.1 / 256.2 | 197.1 / 274.8 |
| 8K | 117.8 / 170.6 | 268.2 / 381.5 | 303.1 / **401.3** |
| 128K | 107.2 / 149.8 | 253.5 / 348.1 | 282.6 / 368.0 |
| 256K (maximum) | 106.6 / 149.0 | 251.7 / 346.8 | 277.4 / 360.6 |

Prefill of an 8,192-token prompt: 13,903 / 13,908 tok/s (16-bit / 8-bit, no MTP). The long prompts
repeat the same text, so they accept 3.9-4.0 tokens per round; typical text gives lower MTP3 throughput.
At 450 W decode is unchanged and prefill drops about 13%.

---

## Checks

- **8-bit quality** against the BF16 Swift artifact on six oracle prompts: mean log-probability change
  of generated tokens 0.003-0.086, perplexity -2.4% to +1.7%, greedy generation changes only at
  near-tie tokens.

---

## Limits

- Only the Swift 8-bit combination is measured; the other converter combinations are not.
- A 64 GB host keeps only part of the PLE table in the page cache; the rest is read from storage.
- The benchmark text repeats, which flatters MTP3 acceptance at long context.

---

## Build and chat

```sh
git clone https://github.com/lkarlslund/ninfer6000.git
git -C ninfer6000 checkout 613d5c4b73080823949ec61b9c5f2b6ec050cd01
cd ninfer6000
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
python3 -m tools.convert.qwen3_8_flash_next_125b_a6b.convert --model /path/to/Swift-1.5-Qwen3.8-Flash-Next-NVFP4 \
  --source-profile swift --ple-format fp8_e4m3fn --projection-format fp8_e4m3fn_row_bf16 \
  --out out/v3/swift_1_5_qwen3_8_flash_next_nvfp4_fp8ple_fp8proj.ninfer --device cuda
./build/apps/ninfer-serve out/v3/swift_1_5_qwen3_8_flash_next_nvfp4_fp8ple_fp8proj.ninfer --port 8003 \
  --max-context 196608 --kv-capacity auto --kv-dtype bf16 --max-concurrency 2 \
  --spec mtp --draft-tokens 3 --lm-head-draft --vision --preserve-thinking --prefill-chunk 8192
```

Check the source checkpoint first with `python -B tools/verify_model.py <this engine folder> /path/to/Swift-1.5-Qwen3.8-Flash-Next-NVFP4`.

---

## Credits

Apache License 2.0,
[LICENSE](https://github.com/lkarlslund/ninfer6000/blob/613d5c4b73080823949ec61b9c5f2b6ec050cd01/LICENSE)
at the pinned commit. A fork of [Neroued/ninfer](https://github.com/Neroued/ninfer); vendored
dependencies keep their licenses under `third_party/`, and the weights keep their own.
