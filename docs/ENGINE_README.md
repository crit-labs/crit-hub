# Engine README template

Every engine README in crit-hub uses these sections, with these names, in this order. Keep it
to about 100 lines; put longer history, validation detail or design notes in a file under
`engine/` and link it from the section it belongs to.

````markdown
# <Model> @ <encoding> · <GPU> · <engine name>

> [!NOTE]
> **Contributed by** <name> · [<owner>/<repo>](https://github.com/<owner>/<repo>)
>
> Built with [crit](https://github.com/crit-labs/crit): a sealed crit candidate, exported with `tools/export_engine.py`.

One short paragraph: what the engine is (written from scratch or adapted, crit Core version), the
tested platform (OS, CUDA, GPU and memory, host RAM when it matters) and the context length.

## Model

The checkpoint repository and revision, the architecture in one or two sentences, how the weights
are used (as stored, or what is re-encoded and why), any drafter or extra download, and a pointer to
model.json and `tools/verify_model.py`.

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
| Long-context decode | ... |
| Quality, worst path | ... |

| Run benchmark, 256 output tokens | Decode | Time to first token | Prefill |
| --- | ---: | ---: | ---: |
| 512-token prompt | ... | ... | ... |
| 14,336-token prompt | ... | ... | ... |

One short paragraph: measurement conditions (crit Eval version, repetitions, greedy or sampled,
free memory) and what sets the speed.

## Checks

- **Arithmetic:** ...
- **Functional:** ...
- **Resources:** ...

## Limits

- Known issues first, then platform, memory and feature limits.

## Build and chat

Build commands, the checkpoint check, `crit serve` and, when the engine ships one, its standalone
chat command.

## Benchmark your machine

Only when the engine ships a benchmark tool: how to run it and how to share results.

## Credits

Credits and licenses are in [THIRD_PARTY.md](THIRD_PARTY.md) and [LICENSES.txt](LICENSES.txt).
````

## External engines

An external engine uses the same sections in the same order, with three differences.

The notice under the title says the engine was not built with crit and links the pinned commit:

````markdown
> [!WARNING]
> **External engine.** Not built with crit, and its source has not been reviewed by crit-hub. Read
> it before you build or run it.
>
> **Contributed by** <name> · [<owner>/<repo>](https://github.com/<owner>/<repo>) · source at
> [`<short SHA>`](https://github.com/<owner>/<repo>/tree/<40-character commit SHA>)
````

- The opening paragraph names the toolkit or runtime instead of a crit Core version.
- **Build and chat** starts with a clone of the pinned commit:
  `git clone <repository> && git -C <folder> checkout <40-character commit SHA>`.
- **Credits** names the source's license and links its license file at the pinned commit.
