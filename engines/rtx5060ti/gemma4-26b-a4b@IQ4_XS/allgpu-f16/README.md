# Gemma 4 26B-A4B @ IQ4_XS · RTX 5060 Ti 16 GB · allgpu-f16

The sealed resident chat engine for Gemma 4 26B-A4B IQ4_XS on one RTX 5060 Ti
16 GB with a Ryzen 5 3600. It is the fastest configuration the run sealed while
still clearing the fixed 1 GiB VRAM headroom gate, and it is the honest
before-picture for the speculation engine published alongside it.

Model: `unsloth/gemma-4-26B-A4B-it-GGUF`, file `gemma-4-26B-A4B-it-UD-IQ4_XS.gguf`
(13,597,177,568 B, sha256 `babd1e38…ffe34aecf`). Weights stay external; the run
records their exact identity and the engine refuses to start against other bytes.

| Frozen 512-in / 128-out case | Measured |
| --- | ---: |
| Decode, median of three | 88.4 t/s (87.2-88.9) |
| Prefill | 1895 t/s |
| First token | 169 ms |
| Load | 19.7 s |
| Minimum free VRAM | 1409 MiB |
| Peak VRAM | 15,694,036,992 B |

Three configurations were swept and measured on this host rather than assumed:

| configuration | decode | min free VRAM |
| --- | ---: | ---: |
| all experts on GPU, f16 KV, `-fa auto` | **88.4 t/s** | 1409 MiB |
| all experts on GPU, f16 KV (`llama-bench`) | 94.6 t/s | 1348 MiB |
| `--n-cpu-moe 1`, f16 KV | 75.1 t/s | 1594 MiB |
| `--n-cpu-moe 1`, q8_0 KV | 70.6 t/s | 1180 MiB |
| `--n-cpu-moe 2`, f16 KV | 54-75 t/s | 1938 MiB |

The engine therefore keeps every layer and expert in VRAM, uses f16 K and V,
and also disables the model's extended-thinking channel by default: with thinking
on a 256-token budget was consumed entirely inside the thought channel with zero
answer characters and `finish_reason=length`; with it off the same prompt returned
a complete answer in about 31% less wall time. A per-turn request override
re-enables the channel.

## Checks

- Three fixed prompts returned coherent, on-topic text, including the exact
  answer `51` for 17 × 3.
- Two identical prompts at temperature 0 produced byte-identical output within
  one package process, and greedy decode was self-consistent across runs.
- A two-turn conversation retained a number given in the first turn.
- Greedy decode through the wrapped backend is self-consistent; this is not a
  byte comparison against a different engine.

## Design

Prebuilt llama.cpp `b11463` (commit `d2a79e604`) CUDA 13.4 x64 driven as a
resident `llama-server` over the local HTTP API, with the model's own Jinja chat
template applied by the server so the adapter never hand-renders turn markers.
`-ngl 99`, `--n-cpu-moe 0`, 8192-token f16 K/V cache, `-fa auto`, 12 threads.
Decode graphs, IQ4_XS dequant and matmul kernels, and CUDA attention are all the
backend's own.

Limits. One sequence, greedy decoding, text only, 8192-token context, one desktop
GPU. This candidate fails the run's 2 GiB free-host-RAM gate. That gate was a
preflight assumption, not a user requirement, and it is structurally
unsatisfiable here: llama.cpp memory-maps the 12.66 GiB GGUF and Windows counts
mapped page cache as used physical memory, so free RAM falls to roughly 230-250 MB
during inference for every placement. No OOM, abort or truncated workload
occurred. The VRAM half of the gate passes at 1409 MiB free.

Speculative decoding is **not** implemented in this candidate. It was measured on
this workload through `llama-cli` (ngram variants and a cross-family draft) and
lost to the plain control; the MTP route was not yet discovered. See the sibling
`mtp-n3` engine for the speculation result.

## Build and run

No compiler is needed, but the prebuilt runtime is fetched rather than tracked: binaries stay outside this repository, and several of them exceed the 100 MB per-file limit GitHub enforces. See [VENDORED-RUNTIME.md](engine/VENDORED-RUNTIME.md) for the upstream assets and the SHA-256 of every required file, and restore and verify them before running.

```sh
python -B launch.py --crit-config configs/server-gemma4.package.json
```

The model locator is supplied by the caller at run time; the package is portable.

Credits and licenses are in [THIRD_PARTY.md](THIRD_PARTY.md) and
[LICENSES.txt](LICENSES.txt).

