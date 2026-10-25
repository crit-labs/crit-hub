<!-- Rendered by tools/card.py from card.yaml: edit card.yaml, then run python -B tools/card.py on this folder. -->

# condor

### rtx4080super · qwen3.8-flash-next@hummingbird-precise

> [!WARNING]
> **Contributed by** surya-koritala · [surya-koritala/glyd-condor](https://github.com/surya-koritala/glyd-condor)
>
> External engine: crit-hub has not reviewed its source. Read it before you build or run it.

| Decode (sampled) | Decode (greedy) | Prefill | TTFT | Perplexity | Mean KL divergence | Context |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| – | **118 tok/s** | – | **7.5 s** | – | **0.22** | **262,144 tok** |

- **Measured with:** self-reported (Condor bench/compare.py) · [self-reported.json](evidence/self-reported.json)
- **Hardware:** RTX 4080 SUPER 16 GB; Ryzen 9 7950X3D; 64 GB RAM; Ubuntu 26.04
- **Requires:** Linux, Windows · sm_86, sm_89, sm_120 · 12 GB GPU memory · CUDA 13.0
- **Checkpoint** (`qwen3.8-flash-next@hummingbird-precise`): [`glyd/Qwen3.8-Flash-Next-hummingbird`](https://huggingface.co/glyd/Qwen3.8-Flash-Next-hummingbird/tree/c6dd8e9a9404239b0b920ae4a42a2e2a5702dc1d) at `c6dd8e9a`
- **Source:** [surya-koritala/glyd-condor @ `2f6267b1`](https://github.com/surya-koritala/glyd-condor/tree/2f6267b1f41392a3980e06b12d3e7b9192b1887c)
- **License:** BUSL-1.1

Condor runs Qwen3.8-Flash-Next (180B MoE) on one 12-16 GB NVIDIA card with 32-64 GB of RAM, on Linux
or Windows, with its own engine and its own compressed weights. The busy experts sit on the GPU and
the rest in RAM, where the CPU computes the misses while the GPU works on the layer; experts are
compressed by use, and the model's draft head proposes tokens checked in the same pass. It serves
OpenAI- and Anthropic-compatible APIs, a chat page and MCP. The engine is a closed binary.

---

## Design

| Part | How |
| --- | --- |
| Experts | Hot experts in VRAM, the rest in RAM computed on the CPU, overlapped with the GPU |
| Weights | glyd's hummingbird compression: precise ~3.0 bpw experts, best ~3.4, fast ~2.0; attention 6 bit (4-5 in fast) |
| Speculation | The model's MTP draft head, verified in the same pass |
| Prefill | Long prompts go through a layer at a time, so a 32K prompt fits |
| Serving | OpenAI API at `/v1`, Anthropic at `/v1/messages`, MCP, a chat page; one request at a time |

---

## Measured

RTX 4080 SUPER 16 GB, Ryzen 9 7950X3D, 64 GB RAM, greedy, thinking off:

| | precise | best | fast (v0.2.2) |
| --- | ---: | ---: | ---: |
| Download | 82 GB | 89 GB | 66 GB |
| KL to BF16 (wikitext-2) | 0.22 | 0.165 | 0.39 |
| Writing, short / 4K / 30K prompt | **118** / 110 / 96 | 104 / 92 / 97 | 124 / 116 / 125 tok/s |
| Read a 30K-token prompt | 7.5 s | 7.5 s | 7.4 s |
| GSM8K / MMLU-Pro (150 q each) | 148 / 126 | 149 / 128 | 149 / 121 |

Fast was measured on 2026-10-19, precise and best in earlier October runs.

---

## Checks

- **Quality:** GSM8K and MMLU-Pro through Inspect AI, logs in `results/`.
- **Tools:** 19 tool-call tests in `tests/`; two fail in fast mode.

---

## Limits

- The engine is closed source under BUSL-1.1 (non-commercial free, commercial use needs a license); only scripts and docs are Apache-2.0.
- Text only; NVIDIA only; one request at a time.
- No exact measurement date for precise mode; CUDA version not stated (driver 580+ required).

---

## Build and chat

```sh
git clone https://github.com/surya-koritala/glyd-condor.git condor
git -C condor checkout 2f6267b1f41392a3980e06b12d3e7b9192b1887c
cd condor && ./setup.sh
```

The setup checks GPU, RAM and disk, installs the prebuilt engine, downloads the model (precise by
default) and opens `http://127.0.0.1:8080`. Check the download with
`python -B tools/verify_model.py <this engine folder> <model folder>`. Windows: `setup.ps1`.

---

## Credits

Engine: Business Source License 1.1,
[ENGINE-LICENSE](https://github.com/surya-koritala/glyd-condor/blob/2f6267b1f41392a3980e06b12d3e7b9192b1887c/ENGINE-LICENSE);
repository: Apache License 2.0,
[LICENSE](https://github.com/surya-koritala/glyd-condor/blob/2f6267b1f41392a3980e06b12d3e7b9192b1887c/LICENSE).
By Surya Koritala (glyd). The model is Qwen's, and the compressed weights carry its license.
