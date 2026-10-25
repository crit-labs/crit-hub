<!-- Rendered by tools/card.py from card.yaml: edit card.yaml, then run python -B tools/card.py on this folder. -->

# voidinfer

### rtx5090 · qwen3.8-27b@EXL3-6.0bpw

> [!WARNING]
> **Contributed by** The8Darkness · [The8Darkness/voidinfer](https://github.com/The8Darkness/voidinfer)
>
> External engine: crit-hub has not reviewed its source. Read it before you build or run it.

| Decode (sampled) | Decode (greedy) | Prefill | TTFT | Perplexity | Mean KL divergence | Context |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| – | **147 tok/s** | **3,900 tok/s** | **8.1 s** | – | **0.014** | **262,144 tok** |

- **Measured with:** self-reported (VoidInfer engine comparison) · [self-reported.json](evidence/self-reported.json)
- **Hardware:** Windows; RTX 5090 32 GB
- **Requires:** Windows · sm_120a · 32 GB GPU memory · CUDA 13.3
- **Checkpoint** (`qwen3.8-27b@EXL3-6.0bpw`): [`Mia-AiLab/Qwen3.8-27B-EXL3`](https://huggingface.co/Mia-AiLab/Qwen3.8-27B-EXL3/tree/7d9681e6b9464f3b9dbb2f45d04f50bc4d59d9aa) at `7d9681e6`
- **Download** (drafter): [`Mia-AiLab/Qwen3.8-27B-DFlash2-EXL3-5.0bpw`](https://huggingface.co/Mia-AiLab/Qwen3.8-27B-DFlash2-EXL3-5.0bpw/tree/4f0436269bca761b071f05319e8e04a87cc633f9): the DFlash2 draft in EXL3 5 bpw, from incoai/Qwen3.8-27B-DFlash2
- **Source:** [The8Darkness/voidinfer @ `a7dc1a74`](https://github.com/The8Darkness/voidinfer/tree/a7dc1a74392d5368a513b11acabd09c65e95d90a)
- **License:** Apache-2.0

VoidInfer is a native Windows C++20/CUDA engine for Qwen3.8-27B on one RTX 5090, with 6-bit EXL3
weights, DFlash2 speculative decoding, up to eight concurrent lanes and image/video input. It began
as a Windows adaptation of the NInfer codebase. Its attention history lives on the GPU as compressed
INT4 OSCAR codes for drafting, and every drafted token is re-checked against attention over the exact
FP16 history in host RAM before it is published, so the compression changes speed, not output.

---

## Design

| Part | How |
| --- | --- |
| Weights | EXL3 6 bpw (branch `SC_6.00bpw_H6_V6`); the vision tower from Qwen's BF16 checkpoint |
| Speculation | DFlash2 with an EXL3 5 bpw draft |
| KV cache | L0 OSCAR INT4 codes on the GPU; exact FP16 history in host RAM (VeriCache) checks each drafted token |
| Agents | Each agent's exact history in its own host-RAM plane set; resuming is a remap |
| Serving | Up to eight lanes; image and video input |

---

## Measured

Same RTX 5090, token-identical prompts, greedy, against NInfer `a0cf1c17` (NVFP4 and groupwise
artifacts, FP8 KV):

| | VoidInfer | NInfer |
| --- | ---: | ---: |
| Decode, 1 request (4K prompt, 1,024 tokens) | 147 tok/s | **190 tok/s** |
| Prefill, 3K / 32K / 128K | 3.5K / 3.9K / 3.2K tok/s | **9.3K / 10.3K / 6.2K tok/s** |
| TTFT, 32K / 128K | 8.1 / 40.8 s | **3.1 / 20.9 s** |
| KL to BF16, served path, 4K | **0.014** | 0.080 |
| Max context, 1 / 4 requests | **262,144 / 69,632 each** | 208,896 / 26,624 each |

NInfer is faster on raw speed; the fidelity costs decode and prefill throughput. On the full
SuperGPQA and MMLU-Pro sets VoidInfer scores 40.21% / 60.14% against 38.17% / 58.93%.

---

## Checks

- **Fidelity:** weights-only KL to the FP32 reference 0.0025 over 138,880 tokens in 7 domains; BF16
  argmax agrees with 99.3% of served greedy answers on 382 task items.
- **CI:** a Windows integrity workflow runs on the repository.

---

## Limits

- Windows only; the build accepts only `CMAKE_CUDA_ARCHITECTURES=120a`.
- Host RAM grows with context: 2 GiB per 32K tokens of exact history.
- Slower than NInfer on decode, prefill and agent resume.

---

## Build and chat

```bat
git clone https://github.com/The8Darkness/voidinfer.git
git -C voidinfer checkout a7dc1a74392d5368a513b11acabd09c65e95d90a
call C:\BuildTools2026\Common7\Tools\VsDevCmd.bat -arch=amd64
cmake -S voidinfer -B voidinfer\build -G Ninja -DCMAKE_BUILD_TYPE=Release
```

The full steps, vcpkg pins and the prebuilt package (which ships the OSCAR rotations) are in
[docs/build-and-run.md](https://github.com/The8Darkness/voidinfer/blob/a7dc1a74392d5368a513b11acabd09c65e95d90a/docs/build-and-run.md).
Download the target (`Mia-AiLab/Qwen3.8-27B-EXL3`, branch `SC_6.00bpw_H6_V6`) and the draft, check the
target with `python -B tools/verify_model.py <this engine folder> <model folder>`, then start
`voidinfer-serve.exe`.

---

## Credits

Apache License 2.0,
[LICENSE](https://github.com/The8Darkness/voidinfer/blob/a7dc1a74392d5368a513b11acabd09c65e95d90a/LICENSE)
at the pinned commit. Began from the NInfer codebase ([Neroued/ninfer](https://github.com/Neroued/ninfer));
the EXL3 weights are Mia-AiLab's and the DFlash2 draft is Inco AI's.
