# Qwen 3.6 35B-A3B @ IQ4_XS · RTX 5060 Ti 16 GB · moe4-t12

A decode-favouring resident chat engine for Qwen 3.6 35B-A3B IQ4_XS on a single
RTX 5060 Ti 16 GB. This is the fastest sustained decode rate the run measured for
this model, at the cost of nearly all VRAM headroom; the `mtp-n1` engine
published alongside it is the one to prefer, because the model's own MTP head
recovers decode that the conservative placement would otherwise spend.

Model: `unsloth/Qwen3.6-35B-A3B-GGUF`, file `Qwen3.6-35B-A3B-UD-IQ4_XS.gguf`
(17,730,509,792 B, sha256 `649d7508…c14a67ebbb3`). Weights stay external. Note
that the first transfer of this file reached the right byte count with a
different hash and produced degenerate output; it is retained as
`corrupt-first-transfer.gguf` and this engine is checked against the verified
second transfer.

| Frozen 512-in / 128-out case | Measured |
| --- | ---: |
| Decode, median of three | 74.2 t/s (73.9-74.4) |
| Prefill | 1683 t/s |
| First token | 91 ms |
| Load | 24.7 s |
| Minimum free VRAM | 86,983,232 B |
| Peak VRAM | 16,745,758,720 B |

## Checks

- Functional and arithmetic gates passed. Greedy decode is self-consistent
  across repeated runs and a multi-turn conversation retains context.
- The resources gate **fails by design of this placement**: with only 4 of 40 MoE
  layers host-resident, free VRAM bottoms out around 83-86 MB against the run's
  fixed 1 GiB requirement. The engine is published as a measured data point, not
  as a qualified configuration. Raising `--n-cpu-moe` to 8 restores headroom and
  was also measured.

## Design

- **Heterogeneous expert execution.** `--n-cpu-moe 4` keeps the routed-expert
  tensors of 4 of 40 MoE layers in host RAM while the remaining 36 layers and all
  dense, attention, SSM and shared tensors stay in VRAM. The boundary was
  re-derived on the real server with its KV allocation: the `llama-bench -p 0`
  boundary at moe 3 does not survive the KV allocation and collapses to about
  13 t/s.
- **Host thread count.** All 12 hardware threads; raising `-t` from 6 to 12 lifted
  the real server rate from 69 to about 74 t/s at identical placement.
- **Attention state.** 8192-token KV cache, f16 K and V, `-fa auto` via an
  explicit server argument block, prompt cache reuse enabled.
- **Portable package.** The config resolves package-relative paths, and the
  launch entry takes the verified model locator at run time instead of baking an
  absolute path in. A deficiency found by launching from a copied location: if
  the bundled CUDA runtime directory is not on `PATH`, `ggml-cuda.dll` silently
  falls back to CPU kernels and decode drops to about 6 t/s.

Placement sweep from the real server (decode median / minimum free VRAM):
`--n-cpu-moe 4` 74.2 t/s / 83 MB, `n-cpu-moe 5` about 68 t/s / 174 MB,
`n-cpu-moe 8` 58.1 t/s / 1.28 GB, `n-cpu-moe 3` 13.2 t/s (KV pushes it over the
card), `n-cpu-moe 5` with 6 threads 59 t/s. Quantized KV at the same placement
measured q8_0 72.0 t/s and q4_0 54.1 t/s and did not restore headroom. ngram
speculation measured 47.9 t/s against 74 t/s unspeculated.

Limits. One sequence, greedy decoding, text only, 8192-token context. The 2 GiB
free-host-RAM gate is unsatisfiable on this host because llama.cpp memory-maps the
16.5 GiB GGUF and Windows counts mapped page cache as used physical memory. MTP
speculation is not wired up in this candidate.

## Build and run

No compiler is needed, but the prebuilt runtime is fetched rather than tracked: binaries stay outside this repository, and several of them exceed the 100 MB per-file limit GitHub enforces. See [VENDORED-RUNTIME.md](engine/VENDORED-RUNTIME.md) for the upstream assets and the SHA-256 of every required file, and restore and verify them before running.

```sh
python -B launch.py --crit-config configs/server-moe4-t12.package.json
```

Credits and licenses are in [THIRD_PARTY.md](THIRD_PARTY.md) and
[LICENSES.txt](LICENSES.txt).

