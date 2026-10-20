# nmh: engine source

Runnable source of the `mtp3-cpu-experts` engine (internal name nmh) for Nemotron-3.5-Super-120B-A12B
@ NVFP4 on the RTX 5090: a native CUDA library on crit Core 0.2.1 and the Python adapters that load
it. The [engine README](../README.md) has the design, measurements, limits and the full build and
serve commands. Contributed by crit-labs · [crit-labs/crit](https://github.com/crit-labs/crit).

| Path | What it is |
| --- | --- |
| `src/` | CUDA and C++ sources of `libnmh.so`: Mamba-2, attention and latent-MoE layers, the GPU expert cache, the AVX-512 CPU expert path, NVFP4 grouped prompt GEMMs, MTP speculative rounds, the crit Core engine interface |
| `python/` | `eval_adapter.py` for crit Eval and `serve_adapter.py` for `crit serve`, both on crit Core's stock native adapters |
| `third_party/` | NVIDIA CUTLASS headers for the grouped NVFP4 prompt GEMMs, see [third_party/](third_party/README.md) |
| `CMakeLists.txt`, `package.sh` | Build against an installed crit Core SDK and assemble a runnable package: `bash engine/package.sh engine build <package dir> <crit Core install>` |
| `configuration.json` | The settings the sealed candidate applied |
| `NOTICE.md` | Origin of the sources and linked components |
