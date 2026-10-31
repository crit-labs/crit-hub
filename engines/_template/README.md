<!-- Rendered by tools/card.py from card.yaml: edit card.yaml, then run python -B tools/card.py on your engine's folder. -->

# <engine name>

### rtx5090 · <model>@<encoding>

> [!WARNING]
> **Contributed by** [<your account>](https://github.com/<your account>)
>
> External engine: crit-hub has not reviewed its source. Read it before you build or run it.

| Decode (sampled) | Decode (greedy) | Prefill | TTFT | Perplexity | Mean KL divergence | Context |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **100.0 tok/s** | **240.0 tok/s** | **1,000 tok/s** | **150 ms** | **+0.1% vs reference** | **0.003** | **262,144 tok** |

- **Measured with:** crit Eval (crit-eval-v7, decode sampled at the model's defaults) · [crit-eval.json](evidence/crit-eval.json)
- **Hardware:** <OS>; <CPU> (<physical>/<logical> cores); <RAM> GiB RAM; <GPU> <VRAM> GiB, PCIe <generation>.0 x<width>
- **Further figures:** long-context decode 300.0 tok/s
- **Requires:** Linux · sm_120a · 32 GB GPU memory · CUDA 13.1
- **Checkpoint** (`<model>@<encoding>`): [`<owner>/<name>`](https://huggingface.co/<owner>/<name>/tree/<40-character commit>) at `<40-char`
- **Source:** [<your account>/<your repository> @ `<40-char`](https://github.com/<your account>/<your repository>/tree/<40-character commit>)
- **License:** Apache-2.0

- What the engine is: written from scratch or adapted; crit Core version, or the toolkit or
  runtime of an external engine
- What makes it different, a bullet each. The platform, GPU, CUDA and context are already in the header

---

## Design

| Part | How |
| --- | --- |
| Weights | ... |
| Decode | ... |
| Prefill | ... |

---

## Measured

- The conditions behind the numbers: crit Eval version, the candidate, the context
- What sets the speed, and anything a figure needs to be read right

---

## Checks

- **Arithmetic:** ...
- **Functional:** ...
- **Resources:** ...

---

## Limits

- Known issues first, then platform, memory and feature limits.

---

## Build and chat

```sh
git clone https://github.com/<your account>/<your repository> && git -C <your repository> checkout <40-character commit>
```

- The build commands and what they need
- Check the checkpoint: `python -B tools/verify_model.py <engine folder> <model folder>`
- How to serve it and, when the engine ships one, its standalone chat command

---

## Credits

- The source's license, linked at the pinned commit
- The credits for what the engine uses
