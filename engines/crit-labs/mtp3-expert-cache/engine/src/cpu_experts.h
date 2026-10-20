// Routed experts computed on the CPU (AVX-512) straight from their host records, for 1..4 tokens, and the
// server thread that answers the decode pass's mailbox.
#pragma once
#include "cpu_mailbox.h"
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

namespace fnl {

// One expert of a job: its host record (weights + CSF fixed planes), the CSF exception words of its three
// matrices, the three global scales, and the tokens routed to it with their routing weights.
struct CpuExpert {
    const uint8_t* record;
    const uint32_t* exc[3];
    uint32_t exc_count[3];
    float s2[3];
    int mask;
    float weight[kCpuMaxTokens];
};

// One expert applied to up to 4 tokens, each output written separately (no routing weight).
struct CpuRowJob {
    CpuExpert expert;
    int tokens;
    const float* x[kCpuMaxTokens];   // [kCpuHidden] each
    float* y[kCpuMaxTokens];         // [kCpuHidden] each
};

// A fixed pool of worker threads. run() computes y[t] = sum_j weight_jt * down_j(silu(s_g gate_j x_t) *
// (s_u up_j x_t)) * s_d over the experts routed to token t (deterministic: each thread owns fixed output rows
// and sums experts in list order). Blocks the caller until done; the caller's thread takes part.
class CpuExperts {
public:
    explicit CpuExperts(int threads);
    ~CpuExperts();
    void run(const float* x, int tokens, const CpuExpert* experts, int count, float* y);
    // Each job computed whole by one thread (jobs taken in turn); blocks until all are done.
    void run_each(const CpuRowJob* jobs, int count);
    int threads() const { return int(workers_.size()) + 1; }

private:
    void worker(int id);
    void work(int id);
    void barrier();

    std::vector<std::thread> workers_;
    std::mutex mutex_;
    std::condition_variable wake_;
    std::uint64_t generation_ = 0;
    bool stop_ = false;
    std::atomic<int> arrived_{0};
    std::atomic<std::uint64_t> phase_{0};
    std::atomic<int> done_{0};
    // job
    const CpuRowJob* row_jobs_ = nullptr;
    int row_count_ = 0;
    std::atomic<int> row_next_{0};
    void work_rows();
    const CpuExpert* experts_ = nullptr;
    int count_ = 0, tokens_ = 0;
    float* y_ = nullptr;
    alignas(64) float xe_[kCpuMaxTokens][1280], xo_[kCpuMaxTokens][1280];   // even / odd elements of x
    std::vector<float> act_;                  // [count][tokens][640] as even / odd halves
};

// Polls a mailbox in pinned memory and answers each job with the CPU pool. expert(layer, e) describes expert e.
class CpuMissServer {
public:
    using Describe = std::function<void(int layer, int expert, CpuExpert& out)>;
    // After each reply: fill the job's slots in the background (enqueue copies); then issued() reaches the seq.
    using Insert = std::function<void(const CpuJob& job)>;
    CpuMissServer(CpuJob* job, CpuReply* reply, Describe describe, Insert insert, int threads);
    uint32_t issued() const { return issued_.load(std::memory_order_acquire); }
    ~CpuMissServer();
    unsigned long long jobs() const { return jobs_.load(); }
    unsigned long long experts() const { return experts_.load(); }
    unsigned long long busy_ns() const { return busy_ns_.load(); }
    unsigned long long insert_ns() const { return insert_ns_.load(); }

private:
    void serve();
    CpuJob* job_;
    CpuReply* reply_;
    Describe describe_;
    Insert insert_;
    CpuExperts pool_;
    std::atomic<uint32_t> issued_{0};
    std::unique_ptr<CpuJob> snapshot_;
    std::atomic<bool> stop_{false};
    std::atomic<unsigned long long> jobs_{0}, experts_{0}, busy_ns_{0}, insert_ns_{0};
    std::thread thread_;
};

}  // namespace fnl
