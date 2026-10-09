#pragma once
#include "crit/core.h"
#include <cuda_runtime_api.h>

namespace crit::core {
enum class Queue { compute, transfer };
enum class SyncSchedule { spin, blocking, yield, automatic };
struct CudaState;

// Shared allocation aliases charge once. The last owner settles both backend
// streams before freeing, including during failed partial model construction.
// Raw views may be used only on those streams and must not outlive their owner.
class CudaBuffer {
public:
    CudaBuffer() = default;
    void* data() const noexcept;
    std::size_t size() const noexcept;
    Memory memory() const;
    // Loading only, never per-step work: both synchronize the stream. Pageable
    // transfers use owned pinned staging, so a failed fence never leaves GPU work
    // using the borrowed host span. In work loops use copy_async/readback_async
    // with owned pinned buffers. The caller orders earlier uses on the other stream.
    void upload(std::span<const std::byte>, std::size_t offset = 0, Queue = Queue::transfer);
    void download(std::span<std::byte>, std::size_t offset = 0, Queue = Queue::transfer) const;
private:
    struct Impl;
    std::shared_ptr<Impl> impl_;
    explicit CudaBuffer(std::shared_ptr<Impl>);
    static CudaBuffer create(std::shared_ptr<CudaState>, Memory, std::size_t);
    friend class CudaBackend;
};

// Events order work without a host fence. wait() is the explicit result fence.
// Recording/querying before record and ordering foreign streams are rejected.
// Event operations require exclusive object access; backend owners may be shared.
class CudaEvent final : public Readiness {
public:
    CudaEvent() noexcept;
    ~CudaEvent();
    CudaEvent(CudaEvent&&) noexcept;
    CudaEvent& operator=(CudaEvent&&) noexcept;
    CudaEvent(const CudaEvent&) = delete;
    CudaEvent& operator=(const CudaEvent&) = delete;
    void record(cudaStream_t);
    void order(cudaStream_t) const;
    bool ready() const;
    void wait() override;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    explicit CudaEvent(std::unique_ptr<Impl>);
    friend class CudaBackend;
    friend class CudaTimer;
};
// Fixed-capacity stage timings on the compute stream, with preallocated events
// and no per-step allocation. Time around graph launches: recording inside a
// stream capture is rejected. collect() requires completed work and never waits.
class CudaStageTimer {
public:
    CudaStageTimer() noexcept;
    ~CudaStageTimer();
    CudaStageTimer(CudaStageTimer&&) noexcept;
    CudaStageTimer& operator=(CudaStageTimer&&) noexcept;
    void begin(DeviceStage, std::uint32_t index = 0);
    void end();
    void reset();
    std::span<const DeviceTiming> collect();
    std::size_t capacity() const noexcept;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    explicit CudaStageTimer(std::unique_ptr<Impl>);
    friend class CudaBackend;
};
class CudaTimer {
public:
    void start();
    void record_stop();
    float elapsed_ms() const; // Explicitly wait on stop before asking for elapsed.
    float stop_ms();
private:
    CudaEvent start_, stop_;
    cudaStream_t stream_;
    CudaTimer(CudaEvent, CudaEvent, cudaStream_t);
    friend class CudaBackend;
};

// One backend belongs to one engine/ledger. Buffers and events retain the CUDA
// state even after this facade dies. Extensions must submit all work to its two
// streams; no device-global synchronization is imposed in the work loop.
class CudaBackend final : public Backend {
public:
    explicit CudaBackend(int device = 0, SyncSchedule = SyncSchedule::spin);
    ~CudaBackend() override;
    void initialize(Resources&) override;
    void bind_thread() override;
    void drain() override;
    int device() const noexcept;
    cudaDeviceProp properties() const;
    cudaStream_t stream(Queue = Queue::compute) const;
    SyncSchedule sync_schedule() const;
    CudaBuffer allocate(Memory, std::size_t bytes);
    CudaEvent event();
    CudaTimer timer(Queue = Queue::compute);
    // Owners must belong to this backend. Async calls do not add host fences;
    // last-owner release protects pending accesses. Zero-byte calls are valid.
    void copy_async(const CudaBuffer& dst, std::size_t dst_offset,
                    const CudaBuffer& src, std::size_t src_offset, std::size_t bytes,
                    Queue = Queue::transfer);
    void fill_async(const CudaBuffer&, int value = 0, Queue = Queue::compute);
    // Pinned-host destination, device source. Copies on the compute queue after
    // the kernels that produced the data, then records completion; no host fence.
    // Return &completion as Progress::readiness.
    void readback_async(const CudaBuffer& host_dst, std::size_t dst_offset,
                        const CudaBuffer& device_src, std::size_t src_offset, std::size_t bytes,
                        CudaEvent& completion);
    CudaStageTimer stage_timer(std::size_t capacity);
private:
    std::shared_ptr<CudaState> state_;
};
// Failed CUDA drains/free calls retain storage plus its reservation and backend.
// Diagnostic exceptional retention requires process recovery, not normal reuse.
std::size_t quarantined_cuda_allocations() noexcept;
std::size_t quarantined_cuda_backends() noexcept;
}
