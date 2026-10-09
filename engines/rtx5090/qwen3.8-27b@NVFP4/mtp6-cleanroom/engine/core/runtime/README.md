# Request runtime

[runtime.cpp](runtime.cpp) implements Engine/Session lifecycle, single and opt-in
bounded concurrent requests, rotating prefill, grouped commit/publication,
readiness, cancellation and independent retirement polling.
The public interface is [core.h](../include/crit/core.h).
Model computation is supplied through the public Model/Session interfaces.
See [current contracts](../contracts/README.md). Admission is FIFO: a deferred
head blocks later requests even if a smaller request could fit. Prefill rotates
in bounded chunks; decode groups commit before publishing accepted output.

Submission checks the outstanding allowance before retaining payload memory.
Concurrent requests reserve their maximum accepted-token staging buffer at
submission, alongside prompt and result storage, and reuse it across steps.
Insufficient Core buffer capacity rejects the new submission without failing
accepted work. Staging is released at completion; prompt/results remain charged
until their handle is consumed. Model/session memory entitlement remains the
engine adaptation's admission responsibility.

## Request options

`submit(prompt, output_tokens, options)` accepts a value-owned `RequestOptions`.
Unset sampling fields use the adapter's defaults. Temperature must be finite and
nonnegative (zero requests greedy output); top-p is in `(0, 1]`; top-k is at most
the declared vocabulary (zero disables that filter). Seeds are unsigned 64-bit
values. `ModelInfo.sampling` declares support for each field. Unsupported or
invalid options reject submission before acquiring memory or calling the model.
Core passes the values through; the adapter owns filter order, rounding, RNG
algorithm and private per-request RNG state. A seed does not promise identical
output across different numerical implementations.

`options.stop_tokens` replaces the model's stop-token list for this request.
An unset value inherits the model list; an explicitly empty vector disables it.
Core trims at the first stop, includes that token in the committed result, and
reports `Finish::stop` in both serial and batched generation. String/sequence
stopping that needs tokenizer semantics belongs to the adapter.

For additional parameters, derive an immutable host-only `RequestContext` and
set `options.context`; the adapter must declare `ModelInfo.request_context`.
`host_bytes()` reports retained host payload and must be fast, nonblocking and
constant. Core charges that declared extent conservatively per request, even
when requests share a context. It also charges the stop vector's retained
capacity. This is a reservation ledger, not process-RSS measurement. Execution
state, device allocations and asynchronous work belong to model/session owners,
not this parameter object. Do not mutate the context through another alias.

Serial adapters implement `open_request(prompt, options)`; concurrent adapters
receive the same options through the non-null `RequestView::options` pointer.
Borrowed prompts/options remain valid through session close; failed settlement
retains them with the quarantined runtime. Original caller vectors may be edited
after submission. The shared context remains immutable. Context-bearing scoring
uses `score_request(prompt, first_target, options)`; scoring rejects generation
options. Legacy hooks remain usable with default options and Core-owned stop
overrides, but cannot silently ignore sampling/context fields.

The [native fixture](../../tests/core/request_options_tests.cpp) covers option
forwarding, stop overrides, serial/batch RNG isolation, caller mutation, resource
ceilings, invalid capabilities and retained ownership after failed drain. Its
sampler is synthetic and does not qualify model arithmetic or sampling quality.

## Scoring

`submit_score(prompt, first_target, options)` requires `ModelInfo.scoring` and
returns one finite score per token from `first_target` to the end. The first
target must have a preceding token. The adapter defines score semantics (for
example, log probabilities); Core checks count and finiteness, not calibration.

For a `ConcurrentModel`, scoring joins the same FIFO and outstanding allowance
as generation. A score is an exclusive barrier: earlier generation must finish
retirement before the scoring hook runs, and later generation cannot pass it.
Scores use `Model::score_request()` (or the legacy `score()` with default options),
without creating a generation session or consuming a lane generation. Multiple
scores run sequentially. This supports evaluation on a concurrent model; it does
not batch scoring or overlap it with generation. Long scores delay later work.

Queued cancellation and pending deadlines remove the score without invoking the
model. Active cancellation waits for the hook to return and backend work to drain,
then discards the values. Hooks must return; Core cannot preempt a stuck callback.
A scoring failure on the concurrent path poisons the engine, fails its remaining
queue, and quiesces/drains before releasing owners, as with generation failures.
The serial path retains its existing recovery after successfully drained errors.

Core reserves the public result extent at submission. The adapter must reserve
its temporary/returned score allocation separately through `Resources`, retaining
that reservation until the next ordinary request callback or model destruction.
Join host producers before returning; queued backend writes may finish during
Core's drain. Core retains the returned allocation, prompt and options through
settlement, including failed-drain quarantine. Once ready, it validates and copies
the scores into its reserved result, so an oversized temporary allocation does
not inflate the public result's retained capacity.

The [concurrency fixture](../../tests/core/concurrency_tests.cpp) exercises FIFO
barriers, delayed retirement, queued backend writes, context ownership, queued
expiry/cancellation, active cancellation and failure quarantine. These are host
lifecycle checks, not model-quality or throughput measurements.

## Text services

Return a `shared_ptr<const TextFrontend>` from `Model::text_frontend()` to opt
into independent text processing. Core obtains it once during construction.
`Engine::tokenize()` and `detokenize()` then use that service without the model
lock, including during active generation and inside a `Handle::wait()` token
sink. Every service method must support concurrent host callers and own its
tables independently of mutable model/backend state. Core validates token IDs
on both boundaries. Tokenization failures do not poison generation.

`Engine::create_decoder()` creates private state for one reply. Feed accepted
tokens to `push(span)` and append its returned text; call `finish()` at the end.
The adapter owns codec-specific buffering, including incomplete UTF-8 fragments,
special tokens and context-sensitive spacing. A generic cumulative detokenization
fallback cannot promise that previously emitted text remains unchanged, so no
such fallback is supplied. A frontend without incremental support rejects decoder
creation explicitly.

Each decoder is used serially; different decoders can run concurrently. Core keeps
the frontend alive until its decoder is destroyed, even after Engine destruction.
Invalid token IDs are rejected before entering the decoder. A decoder exception
or `finish()` closes that stream; subsequent calls reject reuse. The caller should
cancel its request if a text-sink failure means it no longer needs generation.
The [host fixture](../../tests/core/frontend_tests.cpp) covers concurrent text
calls, decoding inside token sinks, split text fragments and independent lifetimes.

Returning no service preserves existing behavior: Core serializes legacy
`Model::tokenize()` and `detokenize()` calls and rejects them while generation
is busy or the runtime is poisoned. Decoding through those methods is not a
streaming interface. Moving or destroying an Engine still requires exclusive
access to that Engine object.

## Lifecycle tracing

Set tracing before constructing the engine; resource limits must cover its ring:

```cpp
crit::core::Profile profile;
profile.host_limit = 64 * 1024 * 1024;
profile.device_limit = 1024 * 1024 * 1024;
profile.trace = {.capacity = 4096, .slow_step_us = 50000};
// Construct Engine with this profile, backend and factory.
// auto snapshot = engine.trace();
```

`capacity` is the number of fixed-size events, at most 65536. Zero disables tracing.
The ring charges `capacity * sizeof(TraceEvent)` host bytes. Snapshot copies allocate
on the calling thread; collection performs no per-event allocation or device fence.

| Field | Meaning |
| --- | --- |
| `sequence`, `timestamp_ns` | Recording order and monotonic host time since engine construction |
| `key` | Request, lane and generation; request zero is engine-wide, generation zero has no generation lane (queued or scoring) |
| `phase`, `mark` | Lifecycle phase with point, begin, end or exception exit |
| `operation` | Correlates a begin/end pair; grouped rows share an operation ID |
| `duration_ns` | Host elapsed time for a completed phase |
| `gap_ns` | On step events, time since the previous step ended |
| `slow`, `slow_steps` | Threshold classification and count of whole steps, counted once per group |
| `finish` | Request result on completion events |
| `stage`, `index` | On device events, the adapter's `DeviceStage` and its index within the step |
| `overwritten`, `dropped` | Ring replacement and events lost to contention/counter exhaustion |

Phases cover submission, admission, open, prefill/decode or score, readiness,
commit, publication, retirement, quiesce and drain. A `step` wraps advance,
readiness and commit/publication. Its host duration can include waits; it is not
GPU kernel time. Threshold zero disables slow classification.

`device` events carry adapter-measured GPU time. While tracing is enabled, Core
calls `Session::device_timings()` (or `ConcurrentModel::device_timings()` for the
last decode/prefill group) after readiness and before commit. Each timing becomes
one point event with the step's `operation`, `duration_ns`, `stage` and `index`;
grouped timings are recorded once per row key. The returned storage is borrowed
until commit. A throwing hook counts as `dropped` and does not fail the request.
Without tracing the hook is not called.

Snapshots remain available while the worker is blocked in a callback and may run
concurrently with requests. Serialize moving/destroying the Engine against readers.
Writers never wait for a snapshot lock: inspect loss counters and tolerate missing
pairs. The bounded ring does not freeze on a slow step or failure; capture snapshots
before later traffic overwrites them. Tracing provides neither callback preemption
nor backend recovery.

## Speculative rounds

One `advance()` (or one `decode()` row) covers a whole speculative round: draft,
verify and accept run on the device. Its `readiness` is exactly one readback of
the accepted tokens, such as a CUDA `readback_async` completion, so the host waits
once per round. When the device decides how many tokens the round produced, return
`deferred_tokens = true` with empty `tokens`: after readiness Core calls
`Session::ready_tokens()` (or `ConcurrentModel::ready_tokens(key)` for a batched row)
and validates that span as usual, so `advance()` itself never waits. `commit(accepted)`
then publishes only what the device already chose; Core may trim the count to the
output budget or a stop token, but never adds tokens. Record stage timings around
graph launches and report them through `device_timings()`.
