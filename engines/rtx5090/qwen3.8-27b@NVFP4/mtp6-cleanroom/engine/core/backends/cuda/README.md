# CUDA services

[cuda.cpp](cuda.cpp) supplies explicit CUDA allocations, pinned host ownership,
streams/events, ordering and release services through
[cuda.h](../../include/crit/cuda.h). Model kernels live outside the core.
The backend is opt-in through `CRIT_CORE_CUDA` and requires CUDA Toolkit 13.1.
See the [Core build guide](../../README.md) and
[ownership contracts](../../contracts/README.md).

Use an event recording that belongs to the current operation before calling
`CudaEvent::order`. A wait observes already recorded work; it does not wait for
a future host worker to record that event. Pooling event storage requires job
ownership through release, not a modulo index that can expose an older recording.
See [CUDA event semantics](https://docs.nvidia.com/cuda/cuda-runtime-api/cuda_runtime_api/group__CUDART__EVENT.html).

Keep staging and mapping owners through compute/transfer settlement. Backend
registration must report actual device aliases: adjacent host ranges can require
separate device source segments. The Forge's
[mapped-source validator](../../../forge/cuda-memory-bridge/description.md)
rejects pointer-only resolution when aliases are discontinuous. It performs no
CUDA registration. See [expert execution](../../../docs/EXPERT_EXECUTION.md)
for graph bindings and complete bundle publication responsibilities.

`upload()` and `download()` synchronize their stream and are for loading only,
never per-step work.

## Speculative rounds

Run draft, verify and accept for a round on the device, then issue exactly one
`readback_async(host_dst, offset, device_src, offset, bytes, completion)` for the
accepted tokens and return `&completion` as `Progress::readiness`. The copy uses
the compute queue, so it follows the kernels that produced the tokens, and needs
an owned pinned host destination and a device source. It adds no host fence.
`commit(accepted)` publishes only what the device already chose.

`backend.stage_timer(capacity)` preallocates timing events. Call `begin(stage,
index)`/`end()` around graph launches on the compute stream; recording inside a
stream capture and exceeding capacity throw. After readiness, `collect()` returns
the step's `DeviceTiming` values without waiting (it throws if the work is not
complete), which suits `device_timings()`. `reset()` before the next step.
