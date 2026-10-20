# kestrel: engine source

Runnable source of the `mtp7-fp8attn` engine (internal name kestrel) for Qwen3.8-27B @ NVFP4 on the
RTX 5090: a native CUDA library on crit Core 0.1.0 and the Python adapters that load it. The
[engine README](../README.md) has the design, measurements, limits and the full build and chat
commands. Contributed by crit-labs · [crit-labs/crit](https://github.com/crit-labs/crit).

| Path | What it is |
| --- | --- |
| `src/` | CUDA and C++ sources of `libkestrel.so`: decode GEMVs, DeltaNet, FP8 prompt attention, MTP speculative rounds, the crit Core engine interface |
| `python/` | `eval_adapter.py` for crit Eval, `chat_adapter.py` for `crit serve`, `kestrel_native.py` (ctypes binding), `chat.py` (terminal chat, added at export) and `gemm_table.txt` (cuBLASLt algorithms recorded on the RTX 5090) |
| `CMakeLists.txt`, `build.sh` | Build against an installed crit Core SDK and assemble a runnable package: `bash engine/build.sh engine build/kestrel package core-install` |
| `CORE.md` | The crit revision and toolchain the engine was built with |
| `configuration.json` | The settings the sealed candidate applied |
| `NOTICE.md` | Origin of the sources and linked components |
