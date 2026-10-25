<!-- Rendered by tools/card.py from card.yaml: edit card.yaml, then run python -B tools/card.py on this folder. -->

# ninfer-v100x2

### 2x-v100 · qwen3.8-27b@NVFP4

> [!WARNING]
> **Contributed by** tuxKOH · [tuxKOH/ninfer-V100X2](https://github.com/tuxKOH/ninfer-V100X2)
>
> External engine: crit-hub has not reviewed its source. Read it before you build or run it.

| Decode (sampled) | Decode (greedy) | Prefill | TTFT | Perplexity | Mean KL divergence | Context |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| – | **103.43–230.4 tok/s** | – | – | – | – | **262,144 tok** |

- **Measured with:** self-reported (NInfer V100X2 speed matrix) · [self-reported.json](evidence/self-reported.json)
- **Hardware:** Linux; 2x Tesla V100-SXM2 16 GB on SXM2-to-PCIe adapters, PCIe P2P, no NVLink
- **Requires:** Linux · sm_70 · 32 GB GPU memory · CUDA 12.8
- **Checkpoint** (`qwen3.8-27b@NVFP4`): [`MirkoCovizzi/Qwen3.8-27B-QUASAR-NVFP4-NInfer`](https://huggingface.co/MirkoCovizzi/Qwen3.8-27B-QUASAR-NVFP4-NInfer/tree/d03d4f4f7bdc4efa6df63a2fa3d9becceeb9ae2c) at `d03d4f4f`
- **Source:** [tuxKOH/ninfer-V100X2 @ `5c515130`](https://github.com/tuxKOH/ninfer-V100X2/tree/5c51513096a9fceb9bb1609e43b15efecac30f93)
- **License:** Apache-2.0

NInfer V100X2 runs Qwen3.8-27B on two Tesla V100-SXM2 16 GB cards (CUDA 12.8, SM70) in tensor
parallel, over PCIe P2P through SXM2-to-PCIe adapters and without NVLink. It is a Volta port of
[NInfer](https://github.com/Neroued/ninfer); the recommended model is the QUASAR NVFP4 v3 artifact,
with TP2, INT8 group-64 KV and CUDA Graphs. GUI and TUI front ends ship with it.

---

## Design

| Part | How |
| --- | --- |
| Parallelism | TP2 across the two V100s over PCIe P2P |
| Weights | QUASAR NVFP4 v3 artifact, decoded on Volta (no FP4 tensor cores) |
| Speculation | Optimized-head MTP4 |
| KV cache | INT8 group-64; historical prefix branches in host RAM (`--prefix-cache-tokens`, default 384,000) |
| Experiments | `--volta-long-attention` and `--volta-fast-decode`, off by default; both can change output |
| Serving | HTTP API with Responses and reasoning-effort mapping; optional TP2 image and video input |

---

## Measured

At 85,000 input tokens, greedy, committed wall decode (2026-10-19):

| Task | Vision loaded, 248,000 capacity | Text only, 262,144 capacity + long attention |
| --- | ---: | ---: |
| Chinese story | 86.44 tok/s | 103.43 tok/s |
| English → Chinese translation | 143.57 tok/s | 179.43 tok/s |
| 32-record JSONL | 185.04 tok/s | **230.40 tok/s** |
| Five-slot logic puzzle | 165.22 tok/s | **209.63 tok/s** |

The headline decode is the Text-only range. Prefill at 85K averages about 1,314-1,321 tok/s, with the
first output after 64-65 s. A separate 85K / 180K-capacity coding fixture measured 198.1 tok/s with
both experiments on.

---

## Checks

- **Cached first token:** an exact 300-token fully cached prompt publishes its first token in
  about 2.0-2.6 ms; paired outputs, logits and MTP/state checks pass.
- **Quality:** no new capability evaluation was run for these speed figures; the GPQA and AIME
  figures quoted upstream are the source checkpoint authors'.

---

## Limits

- `--volta-long-attention` changes floating-point reductions and `--volta-fast-decode` NVFP4/MTP
  arithmetic; both can change output and are off by default.
- In the concurrency matrix a GPU fell off the PCIe bus (Xid 79) during 85K/c4.
- No matched current comparison against llama.cpp.

---

## Build and chat

```sh
git clone https://github.com/tuxKOH/ninfer-V100X2.git
git -C ninfer-V100X2 checkout 5c51513096a9fceb9bb1609e43b15efecac30f93
cd ninfer-V100X2
bash tools/v100/build_dependencies.sh
```

Then configure `build-v100` with CMake and Ninja as the README describes, and launch through the GUI,
the TUI or `ninfer-serve` ([docs/serving.md](https://github.com/tuxKOH/ninfer-V100X2/blob/5c51513096a9fceb9bb1609e43b15efecac30f93/docs/serving.md)).
Check the artifact with `python -B tools/verify_model.py <this engine folder> <model folder>`.

---

## Credits

Apache License 2.0,
[LICENSE](https://github.com/tuxKOH/ninfer-V100X2/blob/5c51513096a9fceb9bb1609e43b15efecac30f93/LICENSE)
at the pinned commit. A fork of [Neroued/ninfer](https://github.com/Neroued/ninfer). The QUASAR NVFP4
weights are QUASAR-QAT's, converted to NInfer v3 by MirkoCovizzi.
