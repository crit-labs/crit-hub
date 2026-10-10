# Gemma 4 26B-A4B @ IQ4_XS · RTX 5060 Ti 16 GB · mtp-n3

A resident chat engine for Gemma 4 26B-A4B IQ4_XS that reaches decode rates no
plain placement of this model can reach on one 16 GB card, by speculating with
the model's own multi-token-prediction (MTP) head instead of sampling one token
at a time.

Model: `unsloth/gemma-4-26B-A4B-it-GGUF`, file `gemma-4-26B-A4B-it-UD-IQ4_XS.gguf`
(13,597,177,568 B, sha256 `babd1e38…ffe34aecf`). Weights stay external and the
engine refuses to start against bytes that do not match.

| Frozen 512-in / 128-out case | Prose | Code | List |
| --- | ---: | ---: | ---: |
| Control, no speculation (t/s) | 87.1 | 84.0 | 87.9 |
| This engine, MTP `n-max 3` (t/s) | **130.0** | **150.7** | **159.1** |
| Speedup | +49% | +79% | +81% |

Each figure is the median of three measured repetitions after one warmup on the
same host in the same session, so the comparison is paired. Decode counts the
tokens generated over the time from end of prefill to the last token, sampling
temperature 0 with top-k 1, concurrency 1. Draft acceptance measured 0.53-0.93
with a mean accepted length of 2.6-3.8 tokens per round.

Longer output keeps most of the gain: 700-token prose measured 116.5 t/s
(+34%) and a 512-token prose case 118.7 t/s (+32%).

## Checks

- Three fixed prompts returned coherent, on-topic answers: an explanation of
  Rayleigh scattering, the exact answer `51` to 17 × 3, and a working
  `def sum_list(numbers): return sum(numbers)`.
- Two identical greedy prompts both returned `51`, so decode is self-consistent.
- A multi-turn conversation retained a name given in the previous turn.
- The seeded-answer prompts are answered from the model, not from any cache
  shortcut: the same prompts measured without speculation give different
  timings and the same text.

## Design

The engine drives a prebuilt llama.cpp `b11463` (commit `d2a79e604`) CUDA
backend over the local `llama-server` HTTP API with the model's own Jinja chat
template, and adds the one change this workload rewards:

- **Speculation with the checkpoint's own MTP head.** The plain GGUF has no
  `mtp`/`nextn` tensors, so `--spec-type draft-mtp` cannot run on it. Unsloth
  publishes the head separately as
  `MTP/mtp-gemma-4-26B-A4B-it-Q8_0.gguf` (461,766,816 B, sha256 `6326fb9f…`),
  architecture `gemma4-assistant`. It is loaded as `--model-draft` with
  `--spec-type draft-mtp --spec-draft-n-max 3`. The drafter shares the target's
  KV cache, so it costs about 0.44 GB of VRAM and needs no second cache.
- **All layers and experts on the GPU** (`-ngl 99`, `--n-cpu-moe 0`).
- **f16 K and V cache** at 8192 tokens with `-fa on`, 12 host threads.

Drafter width was swept, not assumed. On this workload `n-max 3` is the best
general-purpose value: 122-131 t/s on prose and 144-156 on code. `n-max 6`
reaches 176 t/s on very predictable code but collapses to 97 on prose as
acceptance falls to 0.31-0.37, and `n-max 8` is worse on both. Higher values buy
tokens per verification only where the continuation is nearly certain.

Limits. One sequence, greedy decoding, text only, 8192-token context, and a
single desktop GPU. Speculation costs about 8 s of extra load time (18.3 s
control against 27.7 s here) and is worth it only on the second turn onward.
Draft acceptance below roughly 0.5 will erase the gain, so a different model or
a much colder style of prompt is not covered.

## Build and run

No compiler is needed, but the prebuilt runtime is fetched rather than tracked: binaries stay outside this repository, and several of them exceed the 100 MB per-file limit GitHub enforces. See [VENDORED-RUNTIME.md](engine/VENDORED-RUNTIME.md) for the upstream assets and the SHA-256 of every required file, and restore and verify them before running.

```sh
python -B launch.py --crit-config configs/server-gemma4.package.json
```

The retained config resolves the backend and runtime paths relative to the
package root, and the model locator is supplied by the caller at run time
through the frozen request, so the package is portable and holds no absolute
path. The sealed candidate also opens in crit's terminal chat.

Credits and licenses are in [THIRD_PARTY.md](THIRD_PARTY.md) and
[LICENSES.txt](LICENSES.txt).

