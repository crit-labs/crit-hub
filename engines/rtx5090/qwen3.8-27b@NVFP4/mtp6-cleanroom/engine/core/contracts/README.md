# Native core contracts

The authoritative native interface is [crit/core.h](../include/crit/core.h);
the current source version is 0.2.0. [CONTRACTS.json](CONTRACTS.json) describes
its ownership responsibilities; it is a documentation record, not a wire ABI.

Model/Session supply computation and logical state. The core owns the worker,
bounded request admission, resource accounting, readiness, commit/publication,
cancellation and retirement. Concurrency is opt-in and constrained by both
Profile and ModelInfo; unsupported consumers retain the serial path.
Declared scoring uses an [exclusive FIFO barrier](../runtime/README.md#scoring)
on concurrent models, after all earlier generation sessions retire. Cancellation
drains active scoring work before discarding it; failed settlement retains the
returned allocation along with prompt, options and model owners.

Borrowed progress storage stays valid until commit. The core waits readiness,
copies/charges accepted tokens and publishes only after successful model commit.
Grouped failure may have changed state: poison/quiesce the request infrastructure
and publish no rows from that group. Earlier committed output remains valid.
When quiescence and drain succeed, failed-request completion follows session
closure and release of its settled reservations, just as ordinary retirement
does. A failed handle does not race that cleanup. Unproved settlement instead
retains the owners and poisons the runtime for process recovery.

Every asynchronous producer belongs in cleanup. The core quiesces producers,
drains backend work and retains complete owners when settlement is unproved.
Timeouts do not prove physical completion. Public CUDA services use explicit
compute/transfer ordering and last-owner physical release.

Optional [lifecycle tracing](../runtime/README.md#lifecycle-tracing) records bounded
per-engine host diagnostics. Writers never wait for snapshot readers; loss is
explicit. Traces do not change readiness, deadline or cleanup obligations.

An optional independent [text frontend](../runtime/README.md#text-services)
permits concurrent host tokenization and private incremental decoders while the
worker generates. It must retain immutable tables independently of the engine;
legacy model tokenizer callbacks remain serialized and idle-only.

## Speculative rounds

A speculative adapter runs a whole round (draft, verify and accept on the device)
in one `advance()` or `decode()` row and returns all accepted tokens. Readiness is
exactly one device-to-host readback of those tokens. `commit(accepted)` publishes
only what the device already chose; host code does not re-verify candidates.
When only the readback reveals the round's length, `advance()` returns readiness
with `deferred_tokens` and empty tokens; Core waits once, then reads
`ready_tokens()` before validation and commit. Cancellation after readiness closes
the session without a commit, so `close()` must settle a launched, uncommitted round.
Stage timings are recorded around graph launches and reported through
`device_timings()` as trace `device` events with `stage`, `index` and
`duration_ns` under the step's operation ID.

## Construction and borrowed backend handles

A failing model factory still causes backend drain during Engine cleanup.
Construct and validate owned state before publishing a borrowed handle to the
backend. A derived C++ destructor does not run when its constructor throws;
clearing the handle there cannot repair an early publication. If early exposure
is essential, use an owning/rollback guard that survives the relevant unwind.
Drain must also tolerate construction that never published a handle.

After successful construction, keep the borrowed target alive through drain,
including later declaration/startup failures. Unregister it after settlement and
before its owner is destroyed. These responsibilities belong to the consumer's
adapter; the core cannot extend the lifetime of an arbitrary external pointer.
The host regression `borrowed_startup` covers pre-publication validation failure,
post-construction declaration failure and ordinary request/final cleanup. It
qualifies this lifecycle pattern only, not external GPU allocation or arithmetic.

Consumers supply encodings, tensor/state layouts, kernels, tokenizer and sampling.
They use installed public SDK targets and keep generated outputs outside immutable
packages. See [building engines](../../docs/BUILDING_ENGINES.md) and
[Forge contracts](../../forge/README.md).

## Optional cached execution and source ownership

These rules apply when a consumer chooses the corresponding Forge techniques.
They add no common core requirement: a complete engine can integrate without
expert caching, mapped host sources or CUDA graphs. The core supplies resource
reservations, physical ownership, readiness and retirement. A consumer that uses
expert caching owns component descriptors, cache admission,
replacement, source-aware projection and exact miss fallback. A loading slot
is not executable residency: publish a complete bundle only after all required
component copies settle successfully under the same upload identity.

Use operation-specific worker/copy completion. `CudaEvent::order` requires an
already recorded marker, and an earlier recording cannot stand for new worker
production. Retain staging, source mappings and destination owners through the
actual consumer fence. Missing completion or a failed enqueue requires retained
owners and explicit backend quiescence, including during cancellation.

Captured graphs retain stable buffers and immutable source generations. Safely ordered
route/map content changes can reuse topology; moved addresses or incompatible
bindings cannot. A mapped-source resolver that returns one device pointer must
validate full source coverage and device alias continuity. Segmented resolvers
can use separate spans under their own backend contract. Drain readers before
unregistering an active mapping.
Optional cache admission must account for host/device envelopes and preserve
ordinary placement/execution when rejected.

The [expert execution guide](../../docs/EXPERT_EXECUTION.md) maps these rules to
actual Forge host code and lists the missing native hooks. These are consumer
integration contracts; no model-specific cache policy or new native ABI is added.
