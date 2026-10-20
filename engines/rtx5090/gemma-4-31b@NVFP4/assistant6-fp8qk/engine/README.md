# g4: engine source

Runnable source of the `assistant6-fp8qk` engine (internal name g4) for Gemma-4-31B @ NVFP4 on the
RTX 5090: a native CUDA library on crit Core 0.2.1 and the Python adapters that load it. The
[engine README](../README.md) has the design, measurements, limits and the full build and chat
commands. Contributed by crit-labs · [crit-labs/crit](https://github.com/crit-labs/crit).

| Path | What it is |
| --- | --- |
| `src/` | CUDA and C++ sources of `libg4.so`: BF16-activation decode GEMVs, sliding-window and full attention over the FP8 KV cache, prompt GEMMs on cuBLASLt, the drafter and speculative rounds, the crit Core engine interface |
| `python/` | `eval_adapter.py` for crit Eval, `chat_adapter.py` for `crit serve`, `g4_native.py` (ctypes binding) and `gemma_tools.py` (the package's Gemma tool-call syntax, used as `ResidentChat.tool_format`) |
| `CMakeLists.txt`, `build.sh` | Build against an installed crit Core SDK and assemble a runnable package: `bash engine/build.sh engine build/g4 package core-install` |
| `CORE.md` | The crit Core version and toolchain the engine was built with |
| `configuration.json` | The settings the sealed candidate applied; set `draft_model` to your copy of the drafter |
| `provenance.json` | Sealed run and candidate, source and model manifest hashes, toolchain, contributor |
| `NOTICE.md` | Origin of the sources and linked components |
