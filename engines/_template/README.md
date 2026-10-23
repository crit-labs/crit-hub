<!-- Rendered by tools/card.py from card.yaml: edit card.yaml, then run python -B tools/card.py on your engine's folder. -->

# <Model> @ <encoding> · <GPU> · <engine name>

> [!WARNING]
> **External engine.** Not built with crit, and its source has not been reviewed by crit-hub. Read
> it before you build or run it.
>
> **Contributed by** <your account> · [<your account>/<your repository>](https://github.com/<your account>/<your repository>) · source at
> [`<40-char`](https://github.com/<your account>/<your repository>/tree/<40-character commit>)

| Decode | Prefill | TTFT | Perplexity vs reference | Mean KL divergence | Context |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **100.0 tok/s** | **1,000 tok/s** | **150 ms** | **+0.1%** | **0.003** | **262,144 tok** |

<sub>rtx5090 · `<model>@<encoding>` · crit Eval (crit-eval-v6) · measured 2026-10-23 · [crit-eval.json](evidence/crit-eval.json) · long-context decode 300.0 tok/s</sub>

- **Requires:** Linux · sm_120a · 32 GB GPU memory · CUDA 13.1
- **Checkpoint** (`<model>@<encoding>`): [`<owner>/<name>`](https://huggingface.co/<owner>/<name>/tree/<40-character commit>) at `<40-char`
- **Source:** [<your account>/<your repository> @ `<40-char`](https://github.com/<your account>/<your repository>/tree/<40-character commit>)
- **License:** Apache-2.0

One short paragraph: what the engine is (written from scratch or adapted; crit Core version, or
the toolkit or runtime of an external engine), the tested platform (OS, CUDA, GPU and memory, host
RAM when it matters) and the context length.

## Model

The checkpoint repository and revision, the architecture in one or two sentences, how the weights
are used (as stored, or what is re-encoded and why), any drafter or extra download, and a pointer to
card.lock.json and `tools/verify_model.py`.

## Design

| Part | How |
| --- | --- |
| ... | ... |

At most one short paragraph on how decoding and prompts run.

## Measured

| Metric | Result |
| --- | ---: |
| Decode | ... tok/s |
| Prefill | ... tok/s |
| Time to first token (geomean) | ... |

One short paragraph: measurement conditions (crit Eval version, repetitions, greedy or sampled,
free memory) and what sets the speed.

## Checks

- **Arithmetic:** ...
- **Functional:** ...
- **Resources:** ...

## Limits

- Known issues first, then platform, memory and feature limits.

## Build and chat

A clone of the pinned source, then the build commands, the checkpoint check
(`python -B tools/verify_model.py <engine folder> <model folder>`), how to serve it and, when the
engine ships one, its standalone chat command:

```sh
git clone https://github.com/<your account>/<your repository> && git -C <your repository> checkout <40-character commit>
```

## Credits

The source's license, linked at the pinned commit, and the credits for what the engine uses.
