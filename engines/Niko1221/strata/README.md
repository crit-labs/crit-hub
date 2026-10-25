<!-- Rendered by tools/card.py from card.yaml: edit card.yaml, then run python -B tools/card.py on this folder. -->

# strata

### rtx5070 · qwen3.8-flash-next@GSQ-Q2-0

> [!WARNING]
> **Contributed by** Niko1221 · [Niko1221/Strata](https://github.com/Niko1221/Strata)
>
> External engine: crit-hub has not reviewed its source. Read it before you build or run it.

| Decode (sampled) | Decode (greedy) | Prefill | TTFT | Perplexity | Mean KL divergence | Context |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| – | **94 tok/s** | **2,650 tok/s** | – | – | – | – |

- **Measured with:** self-reported (Strata README speed table) · [self-reported.json](evidence/self-reported.json)
- **Hardware:** RTX 5070 12 GB; Ryzen 5 7600; 64 GB RAM
- **Requires:** Windows, Linux, macOS · sm_75, sm_86, sm_89, sm_120 · 12 GB GPU memory · CUDA 13.0
- **Checkpoint** (`qwen3.8-flash-next@GSQ-Q2-0`): [`ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF`](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/tree/ed59f92082b1e93c0e96d60a8b11aab089b52f09) at `ed59f920`
- **Source:** [Niko1221/Strata @ `9ae93f7b`](https://github.com/Niko1221/Strata/tree/9ae93f7b1fe989d04a8e404d32cddaeeff10505f)
- **License:** MIT

Strata runs Qwen3.8-Flash-Next, a 125-billion-parameter MoE, on a normal PC with at least 32 GB of RAM
and graphics memory together. It chats, writes code, reads pictures and serves OpenAI and Anthropic
APIs for coding agents. It ships as a desktop app, a `strata` terminal command and an MCP server, for
NVIDIA, AMD, Intel and Apple Silicon. Basalt, Xolotl and strata-nvfp4 are forks of it.

---

## Design

| Part | How |
| --- | --- |
| Weights | One model in several GGUF sizes, from ISTA-DASLab's GSQ-RCO quants (Q2_0 to IQ3_S), a Coder and Gyro |
| Experts | Split between VRAM and RAM; a learned expert profile ships with the engine |
| Speculation | The model's MTP drafter (`--spec`, `--spec-min-p`) |
| Serving | OpenAI and Anthropic APIs, vision on by default, multi-GPU of the same make |

---

## Measured

On a 12 GB card (RTX 5070, Ryzen 5 7600, 64 GB RAM), from the README at the pinned commit:

| Model | Decode | Prompt reading |
| --- | ---: | ---: |
| Q2_0 | **94 tok/s** | 2,650 tok/s |
| IQ2_XS | 79 tok/s | 2,090 tok/s |
| IQ3_XXS | 62 tok/s | 1,750 tok/s |
| IQ3_S | 53 tok/s | 1,620 tok/s |
| Coder | 55 tok/s | 2,180 tok/s |

On a 16 GB AMD RX 9070 XT the Q2_0 model decodes 60 tok/s. Settings per size and community results
are in `docs/MODELS.md` and `docs/COMMUNITY_BENCHMARKS.md` upstream.

---

## Checks

- **Downloads:** the installer and setup verify SHA-256 and test-start the engine before use (`docs/SAFETY.md`).

---

## Limits

- Needs at least 32 GB of RAM and VRAM together; smaller sizes trade quality for speed.
- Never splits a model across NVIDIA and AMD cards.
- The measurement conditions behind the table are summarized; see `docs/MODELS.md` for each size.

---

## Build and chat

```sh
git clone https://github.com/Niko1221/Strata.git
git -C Strata checkout 9ae93f7b1fe989d04a8e404d32cddaeeff10505f
```

Build from source as the README's *Build from source* section describes, or install the
[app or terminal](https://github.com/Niko1221/Strata/releases) and run `strata launch --model q2_0 --yes --no-tui`.
Check the GGUF with `python -B tools/verify_model.py <this engine folder> <model folder>`.

---

## Credits

MIT, [LICENSE](https://github.com/Niko1221/Strata/blob/9ae93f7b1fe989d04a8e404d32cddaeeff10505f/LICENSE)
at the pinned commit. The GSQ-RCO quants are IST Austria DAS Lab's, and the model is Qwen's.
