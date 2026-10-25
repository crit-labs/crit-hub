<!-- Rendered by tools/card.py from card.yaml: edit card.yaml, then run python -B tools/card.py on this folder. -->

# dwarfstar

### dgx-spark · deepseek-v4-flash@ds4-Q2

> [!WARNING]
> **Contributed by** antirez · [antirez/ds4](https://github.com/antirez/ds4)
>
> External engine: crit-hub has not reviewed its source. Read it before you build or run it.

| Decode (sampled) | Decode (greedy) | Prefill | TTFT | Perplexity | Mean KL divergence | Context |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| – | **18.05 tok/s** | **825.76 tok/s** | – | – | – | **65,536 tok** |

- **Measured with:** self-reported (ds4-bench context sweep) · [self-reported.json](evidence/self-reported.json)
- **Hardware:** DGX Spark, 128 GB unified memory
- **Further figures:** long-context decode 13.84 tok/s
- **Requires:** macOS, Linux · sm_121a, sm_120a, sm_89 · 128 GB GPU memory · CUDA 13.0
- **Checkpoint** (`deepseek-v4-flash@ds4-Q2`): [`antirez/deepseek-v4-gguf`](https://huggingface.co/antirez/deepseek-v4-gguf/tree/f71f23d552d664e523b422157b2befbf74040380) at `f71f23d5`
- **Source:** [antirez/ds4 @ `5379c1c9`](https://github.com/antirez/ds4/tree/5379c1c9ac315f1540371b06a3db88491d4059ff)
- **License:** MIT

DwarfStar is Salvatore Sanfilippo's small native inference engine for a few excellent large models on
hardware people can own: DeepSeek V4 Flash first, plus DeepSeek V4.1 Flash, GLM 5.2/5.3, GLM 5.3
Flash, DeepSeek V4 PRO and Qwen3.8 Flash Next. It is deliberately narrow, runs only the GGUF files
the project produces, and tests loading, prompt rendering, tool calls, KV state, the HTTP server and
its native coding agent together. Metal is the primary target; on CUDA the DGX Spark is the main goal.

---

## Design

| Part | How |
| --- | --- |
| Engine | `ds4.c`, no GGML linkage; GGUF quant layouts and some kernels adapted from llama.cpp under MIT |
| Weights | Project GGUFs: Flash Q2 has 2-bit routed experts with Q8 attention projections, shared experts and output |
| Memory | SSD streaming for models larger than RAM; compressed KV caches; disk KV snapshots for sessions |
| Scale-out | Two Macs over RDMA with tensor parallelism; layer pipelines across machines; multi-GPU CUDA including Ada |
| Speculation | DSpark for V4 Flash, MTP for GLM and Qwen (opt-in) |

---

## Measured

Recorded DeepSeek V4 Flash Q2 baseline, 2,048-token continued-prefill intervals, 128 greedy tokens per frontier:

| Machine | Context | Prefill | Generation |
| --- | ---: | ---: | ---: |
| DGX Spark, 128 GB | 2,048 | 825.76 tok/s | **18.05 tok/s** |
| DGX Spark, 128 GB | 16,384 | 872.44 tok/s | 15.10 tok/s |
| DGX Spark, 128 GB | 32,768 | 855.94 tok/s | 14.43 tok/s |
| DGX Spark, 128 GB | 65,536 | 822.98 tok/s | 13.84 tok/s |
| M5 Max, 128 GB | 2,048 | 790.18 tok/s | 39.35 tok/s |

Eight L40S cards reach about 126 tok/s aggregate with 16 sessions.

---

## Checks

- **Capability:** `ds4-eval` runs embedded regression suites (`core`, `hard`) against a real GGUF; these are integration checks, not leaderboard scores.
- **Releases:** a QA run precedes each release (QA_BEFORE_RELEASES.md).

---

## Limits

- Beta quality and fast-changing; the speed table is a recorded baseline, not a fresh benchmark of the pinned commit.
- The pinned file is the current Flash Q2 GGUF (checkpoint 0731), which may be newer than the one recorded.
- The project states it is developed with strong assistance from AI coding agents.

---

## Build and chat

```sh
git clone https://github.com/antirez/ds4.git
git -C ds4 checkout 5379c1c9ac315f1540371b06a3db88491d4059ff
make -C ds4 cuda-spark
```

Download the model with `./download_model.sh ds4f-q2` (into `gguf/`), check it with
`python -B tools/verify_model.py <this engine folder> <model folder>`, then `./ds4 --cuda` or
`./ds4-server --ctx 32768`. Platform guides:
[docs/DGX_SPARK.md](https://github.com/antirez/ds4/blob/5379c1c9ac315f1540371b06a3db88491d4059ff/docs/DGX_SPARK.md).

---

## Credits

MIT, [LICENSE](https://github.com/antirez/ds4/blob/5379c1c9ac315f1540371b06a3db88491d4059ff/LICENSE)
at the pinned commit, keeping the ggml authors' and DeepSeek's copyright notices. By Salvatore
Sanfilippo and the ds4.c authors, with thanks to llama.cpp and GGML.
