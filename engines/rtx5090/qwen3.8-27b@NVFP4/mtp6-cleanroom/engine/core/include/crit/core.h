#pragma once
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace crit::core {
using Token = std::int32_t;
enum class Memory { host, device };
struct Usage { std::size_t live = 0, peak = 0, limit = 0; };

// A reservation is shared by aliases and releases exactly once at the last owner.
class Reservation {
public:
    Reservation() = default;
    std::size_t bytes() const noexcept;
private:
    struct Impl;
    std::shared_ptr<Impl> impl_;
    explicit Reservation(std::shared_ptr<Impl>);
    friend class Resources;
};
class Resources {
public:
    Resources(std::size_t host_limit, std::size_t device_limit);
    Reservation reserve(Memory, std::size_t bytes);
    Usage usage(Memory) const;
private:
    struct Impl;
    std::shared_ptr<Impl> impl_;
};

// Borrowed storage. Marks/rewind are arithmetic operations, never device fences.
class Arena {
public:
    Arena(void* storage, std::size_t capacity);
    std::span<std::byte> allocate(std::size_t bytes, std::size_t alignment = 256);
    std::size_t mark() const noexcept;
    void rewind(std::size_t mark);
    std::size_t peak() const noexcept;
private:
    std::byte* storage_;
    std::size_t capacity_, offset_ = 0, peak_ = 0;
};

struct TraceConfig {
    // Zero disables tracing. Capacity is bounded to 65536 fixed-size events.
    std::size_t capacity = 0;
    // Zero disables slow-step classification; timestamps remain host timings.
    std::uint64_t slow_step_us = 0;
};
struct Profile {
    std::uint32_t context = 32768, prefill_chunk = 1024;
    std::size_t host_limit = 0, device_limit = 0;
    // Generation concurrency is opt-in. Scores join its FIFO as exclusive
    // barriers after generation retirement. Completed, unconsumed handles also
    // occupy the outstanding allowance (max_concurrency + max_pending).
    std::uint32_t max_concurrency = 1, max_pending = 0;
    std::uint32_t pending_timeout_ms = 30000, retirement_timeout_ms = 30000;
    TraceConfig trace = {};
};
struct SamplingOptions {
    // Unset uses the adapter default. Temperature zero requests greedy output;
    // top_k zero disables that filter. RNG state belongs to the private session.
    std::optional<double> temperature, top_p;
    std::optional<std::uint32_t> top_k;
    std::optional<std::uint64_t> seed;
};
struct SamplingSupport { bool temperature = false, top_p = false, top_k = false, seed = false; };
class RequestContext {
public:
    virtual ~RequestContext() = default;
    // Immutable host-only extension parameters. Report their retained payload
    // extent; Core charges it conservatively for each request, even when shared.
    // This metadata call must not block, enqueue work, or change the extent.
    virtual std::size_t host_bytes() const noexcept = 0;
};
struct RequestOptions {
    SamplingOptions sampling;
    // Unset inherits ModelInfo.stop_tokens; an empty vector disables token stops.
    std::optional<std::vector<Token>> stop_tokens;
    std::shared_ptr<const RequestContext> context;
};
struct ModelInfo {
    std::string implementation;
    std::uint32_t vocabulary = 0, context = 0, prefill_chunk = 0;
    std::vector<Token> stop_tokens;
    bool generation = true, scoring = false;
    std::uint32_t max_concurrency = 1;
    SamplingSupport sampling = {};
    bool request_context = false;
};
class Backend {
public:
    virtual ~Backend() = default;
    // Called once before model construction. Owners may retain a copy of the ledger.
    virtual void initialize(Resources&) {}
    virtual void bind_thread() = 0;
    // Settles all backend work submitted by model/session calls, including failures.
    virtual void drain() = 0;
};
struct Services { Resources& resources; Backend& backend; };
// Failed backend drains retain the complete owner rather than freeing live storage.
// A permanently failed backend requires process recovery; this counter is diagnostic.
std::size_t quarantined_engines() noexcept;
class Readiness {
public:
    virtual ~Readiness() = default;
    virtual void wait() = 0;
};
struct WorkBudget {
    std::uint32_t prefill_tokens, output_tokens;
    bool prefill_complete;
};
struct Progress {
    std::uint32_t prompt_tokens = 0;
    bool prefill_complete = false;
    std::span<const Token> tokens;
    Readiness* readiness = nullptr;
    // A device round may learn how many tokens it produced only after its
    // readback. With deferred_tokens set, tokens must be empty, prefill complete
    // and readiness non-null; after readiness Core reads the final span from
    // Session::ready_tokens() (ConcurrentModel::ready_tokens for batched rows).
    bool deferred_tokens = false;
};
// Device stage durations measured by the adapter, e.g. around graph launches.
enum class DeviceStage : std::uint8_t { forward, draft, verify, accept, state, head, other };
struct DeviceTiming {
    DeviceStage stage;
    std::uint32_t index = 0;
    std::uint64_t duration_ns = 0;
};
enum class Finish { length, stop, cancelled, failed };
class Session {
public:
    virtual ~Session() = default;
    virtual Progress advance(WorkBudget) = 0;
    // Called only while tracing, after readiness and before commit. Storage is
    // borrowed until commit; a throw drops the timings, not the request.
    virtual std::span<const DeviceTiming> device_timings() { return {}; }
    // Called once after readiness when advance() returned deferred_tokens.
    // Storage follows the same borrowing rule as Progress::tokens.
    virtual std::span<const Token> ready_tokens();
    // Token storage remains live until commit/close. Commit may throw; no output
    // is published on failure. Async state updates must obey backend ordering.
    virtual void commit(std::uint32_t accepted, bool terminal) = 0;
    // Called after backend drain. Must not throw, enqueue, or retain prompt views.
    virtual void close(Finish) noexcept = 0;
};
// Private state for one text stream. Emit only complete text; retain incomplete
// byte/token fragments until a later push or finish. Access each decoder serially.
class TextDecoder {
public:
    virtual ~TextDecoder() = default;
    virtual std::string push(std::span<const Token>) = 0;
    virtual std::string finish() = 0;
};
// Optional host-only service, independent of mutable Model/Backend state. All
// methods must be safe for concurrent calls, including while generation runs.
// Retain everything needed independently of the engine's lifetime.
class TextFrontend {
public:
    virtual ~TextFrontend() = default;
    virtual std::vector<Token> tokenize(std::string_view) const = 0;
    virtual std::string detokenize(std::span<const Token>) const = 0;
    virtual std::unique_ptr<TextDecoder> create_decoder() const;
};
class Model {
public:
    virtual ~Model() = default;
    virtual ModelInfo info() const = 0;
    virtual std::unique_ptr<Session> open(std::span<const Token> prompt) = 0;
    // Options remain live through close and any failed-drain quarantine.
    // The default preserves legacy open only when no adapter options are set.
    virtual std::unique_ptr<Session> open_request(std::span<const Token>, const RequestOptions&);
    virtual std::vector<Token> tokenize(std::string_view) = 0;
    virtual std::string detokenize(std::span<const Token>) = 0;
    // Called once at construction. Empty preserves serialized idle-only access
    // to the legacy tokenize/detokenize callbacks above.
    virtual std::shared_ptr<const TextFrontend> text_frontend() const { return {}; }
    // Return one finite score per target. Join host producers before returning;
    // Core drains backend queue work before validating/publishing these values.
    // ConcurrentModel scoring runs only with all generation sessions retired.
    // Reserve returned scratch in model-owned resources until the next ordinary
    // request callback or destruction. Core separately reserves public results
    // and retains returned storage through drain, including failed-drain quarantine.
    virtual std::vector<float> score(std::span<const Token>, std::uint32_t first_target);
    virtual std::vector<float> score_request(std::span<const Token>, std::uint32_t first_target, const RequestOptions&);
};
// These identities are logical, never physical allocation or graph-cache keys.
// All callback methods run on the one engine worker. They must return; deadlines
// cannot preempt a stuck extension callback. No callback may call this Engine.
struct RequestKey {
    std::uint64_t request = 0, generation = 0;
    std::uint32_t lane = 0;
    bool operator==(const RequestKey&) const = default;
};
enum class TracePhase { submitted, admission, open, step, prefill, decode, score,
                        readiness, commit, publication, retirement, quiesce, drain, completion, device };
enum class TraceMark { point, begin, end, failed };
struct TraceEvent {
    std::uint64_t sequence = 0, operation = 0, timestamp_ns = 0;
    RequestKey key; // Zero request denotes engine-wide work; generation zero has no generation lane (queued or scoring).
    TracePhase phase = TracePhase::submitted;
    TraceMark mark = TraceMark::point;
    std::uint64_t duration_ns = 0, gap_ns = 0;
    bool slow = false;
    Finish finish = Finish::length; // Meaningful on completion events.
    DeviceStage stage = DeviceStage::other; // Device events: adapter stage and index;
    std::uint32_t index = 0;                // operation is the enclosing step's.
};
struct TraceSnapshot {
    bool enabled = false;
    std::uint64_t overwritten = 0, dropped = 0, slow_steps = 0;
    std::vector<TraceEvent> events; // Recording order; loss is counted separately.
};
class ConcurrentSession {
public:
    virtual ~ConcurrentSession() = default;
    // Only after a successful retirement receipt or quiesce + backend drain.
    // Must not enqueue, throw, or retain borrowed prompt views.
    virtual void close(Finish) noexcept = 0;
};
struct RequestView {
    RequestKey key;
    std::span<const Token> prompt;
    std::uint32_t output_tokens;
    // Always non-null when supplied by Core; same lifetime as the prompt view.
    const RequestOptions* options = nullptr;
};
enum class Admission { ready, deferred, infeasible };
struct WorkItem {
    RequestKey key;
    ConcurrentSession* session;
    WorkBudget budget;
};
struct RowProgress { RequestKey key; Progress progress; };
struct CommitDecision {
    RequestKey key;
    ConcurrentSession* session;
    std::uint32_t accepted;
    bool terminal, cancelled;
};
class Retirement {
public:
    virtual ~Retirement() = default;
    // Nonblocking certificate: every session reader AND host producer has settled.
    // False preserves the occupied lane. An exception poisons the whole engine.
    virtual bool ready() = 0;
};
class ConcurrentModel : public Model {
public:
    std::unique_ptr<Session> open(std::span<const Token>) final;
    // Inspection only; no submissions or provisional allocations. Full request
    // memory entitlement is extension policy, reserved by open before execution.
    virtual Admission admission(const RequestView&) { return Admission::ready; }
    virtual std::unique_ptr<ConcurrentSession> open(const RequestView&) = 0;
    virtual RowProgress prefill(const WorkItem&) = 0;
    // Exactly one row per compact WorkItem, in any order; variable candidates per
    // row allow ordinary, batched speculative, or solo-specific strategies.
    virtual std::vector<RowProgress> decode(std::span<const WorkItem>) = 0;
    // Timings for the last decode()/prefill() group, recorded for every row key.
    // Same calling and lifetime rules as Session::device_timings().
    virtual std::span<const DeviceTiming> device_timings() { return {}; }
    // Final tokens of a row whose RowProgress set deferred_tokens; called once per
    // such row after every row's readiness, before commit.
    virtual std::span<const Token> ready_tokens(const RequestKey&);
    // All borrowed rows are copied before this call. No output is published until
    // it returns successfully. A throw need not roll back model state: it causes
    // whole-engine quiescence/failure, with no publication from this group.
    virtual void commit(std::span<const CommitDecision>) = 0;
    virtual std::unique_ptr<Retirement> retire(RequestKey, ConcurrentSession&, Finish) = 0;
    // Idempotently stop/join producers, preventing all subsequent submissions.
    // Called before final/failure backend drain; throwing retains complete owners.
    virtual void quiesce() = 0;
};
using Factory = std::function<std::unique_ptr<Model>(const Profile&, Services)>;
struct Result {
    std::vector<Token> tokens;
    std::vector<float> scores;
    Finish finish = Finish::length;
    std::uint64_t request_id = 0;
};
using TokenSink = std::function<void(Token)>;
class Handle {
public:
    Handle() noexcept;
    ~Handle();
    Handle(Handle&&) noexcept;
    Handle& operator=(Handle&&) noexcept;
    Handle(const Handle&) = delete;
    Handle& operator=(const Handle&) = delete;
    // May run concurrently with wait(); move/destruction require exclusive ownership.
    void cancel() noexcept;
    std::uint64_t id() const noexcept;
    Result wait(TokenSink sink = {});
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    std::shared_ptr<std::atomic<bool>> cancellation_;
    std::uint64_t id_ = 0;
    explicit Handle(std::unique_ptr<Impl>);
    friend class Engine;
};
class Engine {
public:
    Engine(Profile, std::shared_ptr<Backend>, Factory);
    ~Engine();
    Engine(Engine&&) noexcept;
    Engine& operator=(Engine&&) noexcept;
    Engine(const Engine&) = delete;
    Engine& operator=(const Engine&) = delete;
    Handle submit(std::vector<Token> prompt, std::uint32_t output_tokens, RequestOptions = {});
    Handle submit_score(std::vector<Token> prompt, std::uint32_t first_target, RequestOptions = {});
    std::vector<Token> tokenize(std::string_view);
    std::string detokenize(std::span<const Token>);
    // Requires an independent frontend with an incremental decoder. The returned
    // decoder retains its frontend, validates token IDs, and may outlive Engine.
    std::unique_ptr<TextDecoder> create_decoder();
    ModelInfo info() const;
    Profile profile() const; // Actual core admission/budget/resource settings.
    Usage usage(Memory) const;
    // Safe while the worker is running, including a stuck callback. Allocates
    // only in the reading thread. Does not fence device work or invoke the model.
    TraceSnapshot trace() const;
private:
    struct Impl;
    std::shared_ptr<Impl> impl_;
    friend class Handle;
};
}
