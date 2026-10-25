<!-- Rendered by tools/card.py from card.yaml: edit card.yaml, then run python -B tools/card.py on this folder. -->

# promethevs

### rtx3060 · deepseek-v4.1-flash-1017@CtC-2bit

> [!WARNING]
> **Contributed by** Magnvs7 · [Magnvs7/Promethevs](https://github.com/Magnvs7/Promethevs)
>
> External engine: crit-hub has not reviewed its source. Read it before you build or run it.

| Decode (sampled) | Decode (greedy) | Prefill | TTFT | Perplexity | Mean KL divergence | Context |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| – | **5.8 tok/s** | – | **126.0 s** | – | – | – |

- **Measured with:** self-reported (PROMETHEVS report 0.1 runs) · [self-reported.json](evidence/self-reported.json)
- **Hardware:** Windows 11; Ryzen 9 3950X; 64 GB DDR4; RTX 3060 12 GB; two NVMe drives (2 TB + 1 TB)
- **Requires:** Windows · sm_86, sm_89 · 12 GB GPU memory · CUDA 13.4
- **Checkpoint** (`deepseek-v4.1-flash-1017@CtC-2bit`): [`deepseek-ai/DeepSeek-V4.1-Flash-1017`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/tree/1bae0e1088993b9b82de08f59de5be5a34e42170) at `1bae0e10`
- **Source:** [Magnvs7/Promethevs/engine-0.2 @ `9e062215`](https://github.com/Magnvs7/Promethevs/tree/9e06221518531b353c2d45e71a60951ac452eaef/engine-0.2)
- **License:** Apache-2.0

PROMETHEVS runs DeepSeek-V4.1-Flash-1017 from all 510 GB of its original weights on a PC that cannot
load it: a Ryzen 9 3950X, 64 GB of DDR4, an RTX 3060 12 GB and two NVMe drives. Its method,
compress-then-correct by Ryan Flavell, computes every token on a 2-bit copy of the experts and, when
the model's top two choices are close, recomputes that token with the exact weights rebuilt from a
stored correction plane. Release 0.2 ships the engine as a Windows bundle.

---

## Design

| Part | How |
| --- | --- |
| Experts | 2-bit copies answer every token; a correction plane rebuilds the exact FP4 expert |
| Gate | A token is redone exactly when the gap between the top two candidates is below 0.45 |
| Speculation | DeepSeek's draft head proposes several tokens, checked in parallel lanes |
| Memory | Most-used experts in VRAM, the rest in RAM and on two NVMe drives; ~1,009 GiB of stores built locally from the original weights |
| Serving | OpenAI-compatible endpoint on localhost with streaming and tool calls; one request at a time |

---

## Measured

Build machine, 160 new tokens per prompt, two passes per run (cold: empty expert cache; warm: second
pass of the same prompt):

| | Engine (full method) | Baseline build |
| --- | ---: | ---: |
| Code prompt, cold | **5.8 tok/s** | 5.9 |
| Code prompt, warm (best case) | 7.4 tok/s | 6.6 |
| Reasoning prompt, cold / warm | 4.7 / 4.4 | 4.5 / 4.1 |
| Agent prompt, cold / warm | 5.2 / 4.8 | 4.6 / 4.2 |
| Plain exact decoding, cold | 2.14 tok/s | |

The first token of a 1,163-token prompt arrives after about 126 s; follow-ups start in about 2 s.

---

## Checks

- **Agreement with the exact path:** 99.17% on 960 audited words by the unchecked count, 98.33% counting every difference (the 2-bit copy alone: 97.40%).
- **GSM8K:** 9 of 10.

---

## Limits

- All numbers come from one machine and one prompt family; no measurement dates are given.
- The engine ships prebuilt in the release zip; a byte-for-byte source rebuild is not provided.
- Windows only, RTX 30/40 with 12 GB, one request at a time; reasoning and agent prompts run below 5.3 tok/s.

---

## Build and chat

```powershell
# from the extracted PROMETHEVS_0.2.zip of the v0.2 release
powershell -ExecutionPolicy Bypass -File scripts\check_hardware.ps1
powershell -ExecutionPolicy Bypass -File scripts\build_stores.ps1 --run
powershell -ExecutionPolicy Bypass -File scripts\start_engine.ps1
```

The store builder downloads DeepSeek's weights and checks them against published hashes; compare
the download with `python -B tools/verify_model.py <this engine folder> <model folder>` too. The
endpoint is `http://127.0.0.1:8100/v1`. Steps and known gaps:
[engine-0.2/INSTALL.md](https://github.com/Magnvs7/Promethevs/blob/9e06221518531b353c2d45e71a60951ac452eaef/engine-0.2/INSTALL.md).

---

## Credits

Engine: Apache License 2.0,
[engine-0.2/LICENSE](https://github.com/Magnvs7/Promethevs/blob/9e06221518531b353c2d45e71a60951ac452eaef/engine-0.2/LICENSE);
report, figures and logs: CC BY 4.0. By Ryan Flavell, MAGNVS Ltd
([doi:10.5281/zenodo.23577839](https://doi.org/10.5281/zenodo.23577839)); builds on DeepSeek's model
and draft head and on ggml and llama.cpp.
