<!-- Rendered by tools/card.py from card.yaml: edit card.yaml, then run python -B tools/card.py on this folder. -->

# strata-nvfp4

### rtx5090 · qwen3.8-flash-next@NVFP4-GPTQ

> [!WARNING]
> **Contributed by** sergqwer · [sergqwer/strata-nvfp4](https://github.com/sergqwer/strata-nvfp4)
>
> External engine: crit-hub has not reviewed its source. Read it before you build or run it.

| Decode (sampled) | Decode (greedy) | Prefill | TTFT | Perplexity | Mean KL divergence | Context |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| – | **177 tok/s** | – | – | – | – | – |

- **Measured with:** self-reported (strata-nvfp4 model table) · [self-reported.json](evidence/self-reported.json)
- **Hardware:** Windows 11; Ryzen 9 9950X3D; 128 GB DDR5-5600; RTX 5090 32 GB
- **Requires:** Windows, Linux · sm_120 · 32 GB GPU memory · CUDA 13.0
- **Checkpoint** (`qwen3.8-flash-next@NVFP4-GPTQ`): [`Maximilian228/Qwen3.8-Flash-Next-NVFP4-GPTQ-Strata`](https://huggingface.co/Maximilian228/Qwen3.8-Flash-Next-NVFP4-GPTQ-Strata/tree/9b70ad3f640b2546449bd05387407a64113acf8b) at `9b70ad3f`
- **Source:** [sergqwer/strata-nvfp4 @ `57293b8e`](https://github.com/sergqwer/strata-nvfp4/tree/57293b8ec57203a741dc7b2abd69f24f077037ab)
- **License:** MIT

Strata NVFP4 is a fork of [Strata](https://github.com/Niko1221/Strata) (upstream 0.2.0-beta engine,
server and setup) that runs Qwen3.8-Flash-Next with NVFP4 routed experts on one 12 GB+ card and 64 GB+
of RAM. Setup installs the original Qwen in NVFP4, either the fork's own GPTQ quant or NVIDIA's
checkpoint converted, and still offers upstream's GGUF models.

---

## Design

| Part | How |
| --- | --- |
| Weights | NVFP4 routed experts (GPTQ, the default, or NVIDIA's converted); Q8_0 and NVFP4+Q8 mixes; upstream's GGUFs |
| Experts | In RAM where they fit, read from the SSD in the RAM-budget mode; route tail skip on by default for GGUFs |
| Speculation | The model's MTP drafter (`--spec 6 --spec-min-p 0.7` in Infernix's comparison) |
| Serving | Upstream's OpenAI and Anthropic APIs, vision |

---

## Measured

RTX 5090 (32 GB), Ryzen 9 9950X3D, 128 GB DDR5-5600, Windows 11; median of 3 runs of 4 greedy chats
and one 32,007-token prompt:

| Model | Same top token as Q8_0 | Decode | 32K prompt | RAM in use |
| --- | ---: | ---: | ---: | ---: |
| NVFP4 GPTQ (default) | 93.6% | **177 tok/s** | 3.9 s | 69 GiB |
| NVIDIA NVFP4, converted | 92.9% | 167 tok/s | 3.9 s | 69 GiB |
| NVFP4 + Q8_0 mix | 95.0% | 91 tok/s | 4.2 s | 105 GiB |
| Unsloth UD-Q4_K_XL | 94.1% | 169 tok/s | 4.9 s | 77 GiB |
| ISTA-DASLab IQ3_S | 90.0% | 228 tok/s | 3.9 s | 52 GiB |

On a 64 GB PC the NVFP4 models decode about 115 tok/s in the low-RAM mode.

---

## Checks

- **Accuracy:** top-1 agreement with every routed expert in Q8_0 over 5,153 tokens of code, docs,
  chats and agent transcripts, all models with experts on the CPU and route tail skip off.

---

## Limits

- The speed figures are a 128 GB machine's; smaller RAM falls back to slower modes.
- An opt-in switch on the fork's own quant turns censorship off; the default ships as Qwen does.

---

## Build and chat

```sh
git clone https://github.com/sergqwer/strata-nvfp4.git
git -C strata-nvfp4 checkout 57293b8ec57203a741dc7b2abd69f24f077037ab
```

Build and run setup as the README describes; setup downloads
`Maximilian228/Qwen3.8-Flash-Next-NVFP4-GPTQ-Strata` by default. Check it with
`python -B tools/verify_model.py <this engine folder> <model folder>`.

---

## Credits

MIT, [LICENSE](https://github.com/sergqwer/strata-nvfp4/blob/57293b8ec57203a741dc7b2abd69f24f077037ab/LICENSE)
at the pinned commit (Niko1221 and the Strata contributors). The GPTQ and converted checkpoints are
Maximilian228's; NVIDIA's NVFP4 checkpoint and Qwen's model keep their own licenses.
