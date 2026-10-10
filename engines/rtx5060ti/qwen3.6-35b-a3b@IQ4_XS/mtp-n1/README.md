# Qwen 3.6 35B-A3B @ IQ4_XS · RTX 5060 Ti 16 GB · mtp-n1

A decode-favouring resident engine for Qwen 3.6 35B-A3B IQ4_XS on a single
16 GB card. The model is larger than VRAM, so the routed-expert tensors of
several MoE layers are held in host RAM while the attention, SSM, dense and
shared tensors stay resident on the device. On top of that placement the engine
speculates with the checkpoint's own multi-token-prediction head, which lifts
steady-state decode by a measured 12-15% over the same placement without it.

Model: `unsloth/Qwen3.6-35B-A3B-MTP-GGUF`, file
`Qwen3.6-35B-A3B-UD-IQ4_XS.gguf` (18,209,036,576 B) — the same IQ4_XS
quantisation as the plain release but with the `nextn` head embedded
(`qwen35moe.nextn_predict_layers = 1`). The plain `unsloth/Qwen3.6-35B-A3B-GGUF`
release has no `mtp`/`nextn` tensors at all, so `--spec-type draft-mtp` fails on
it; that file is the fallback, not this engine.

| Frozen 512-in / 128-out case, paired | Prose | Code |
| --- | ---: | ---: |
| Plain IQ4_XS, same placement (t/s) | 52.9 | 48.5 |
| This engine, MTP `n-max 1` (t/s) | **60.6** | **52.7** |
| Speedup | +14.5% | +11.6% |

`n-max 2` is equivalent on this host (60.0 prose, 54.1 code). Draft acceptance
measured 0.83-1.00 with a mean accepted length around 2.0 tokens per round.
Placement for both columns is `--n-cpu-moe 8`, f16 K and V, `-fa on`, 12 threads,
`-np 1`.

## Checks

- Placement was swept on the real server with continuous VRAM sampling rather
  than with `llama-bench -p 0`, which never allocates the KV cache and so reports
  the wrong boundary. `--n-cpu-moe 8` was the smallest split that met the run's
  headroom requirement; in the timed benchmark it measured 55.7-58.2 t/s with a
  minimum of 965,738,496 B free VRAM.
- The model loads and answers through the retained adapter, greedy decode is
  self-consistent across repeated runs, and a multi-turn conversation retains
  context.
- The MTP gain is measured paired: same prompts, same placement, same session,
  with and without `--spec-type draft-mtp`.

## Design

- **Heterogeneous expert execution.** `--n-cpu-moe 8` keeps the routed-expert
  tensors of 8 of 40 MoE layers in host RAM; the other 32 layers plus every
  dense, attention, SSM and shared tensor stay in VRAM. The split is the
  dominant decode lever on this host, so it is set explicitly and never left to
  an automatic planner that could change between builds.
- **Speculation.** The embedded `nextn` head is used directly with
  `--spec-type draft-mtp` and `-np 1`; no separate `-md` draft is loaded.
- **Host thread count.** All 12 hardware threads. Raising `-t` from 6 to 12 at
  identical placement lifted the real server rate from 69 to about 74 t/s, the
  largest single lever found after placement.
- **Quantized KV was rejected after measurement.** q8_0 measured 72.0 t/s and
  q4_0 54.1 t/s against f16 at the same placement, and neither freed enough VRAM
  to matter, so the cache stays f16.

Limits. One sequence, greedy decoding, text only, 8192-token context. MTP raises
cold load from about 24 s to about 167 s and the very first request after a cold
load runs at roughly 3 t/s before the steady state above; that tax is real and
was not tuned. The model is memory-mapped, so free host RAM reads low on a 16 GiB
machine for every placement.

## Build and run

No compiler is needed, but the prebuilt runtime is fetched rather than tracked: binaries stay outside this repository, and several of them exceed the 100 MB per-file limit GitHub enforces. See [VENDORED-RUNTIME.md](engine/VENDORED-RUNTIME.md) for the upstream assets and the SHA-256 of every required file, and restore and verify them before running.

```sh
python -B launch.py --crit-config configs/server-moe8-t12.package.json
```

The model locator is supplied by the caller at run time; the package holds no
absolute path and resolves the backend and CUDA runtime relative to its own root.

Credits and licenses are in [THIRD_PARTY.md](THIRD_PARTY.md) and
[LICENSES.txt](LICENSES.txt).

