// Routed experts computed on the CPU (AVX-512) straight from their pinned host records, for 1..4 tokens, with
// exactly the FP32 operations of the device's expert kernels (experts.cu): per row, 16 element accumulators
// p[j] = fma(w[g][j] * s[g], x[g*16+j], p[j]) over the groups in order, then the fixed tree
// ((p[i] + p[i+8]) -> +4 -> +2 -> +1), times the global scale. An expert's contribution is therefore bitwise the same
// whichever processor computes it, and the decode output does not depend on where a miss was served.
#pragma once
#include "dims.h"
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <thread>
#include <vector>

namespace nmh {

constexpr int kCpuMaxEntries = 48;   // experts of one layer a job may hand to the CPU
constexpr int kCpuMaxTokens = 4;

// Mailbox in mapped pinned memory: the device writes a job and bumps seq; the CPU answers with the same seq.
struct CpuJob {
    volatile uint32_t seq;
    int group, tokens, count;
    int expert[kCpuMaxEntries];
    int mask[kCpuMaxEntries];
    float s2[kCpuMaxEntries][2];
    alignas(64) float x[kCpuMaxTokens][kLatent];
};
struct CpuReply {
    volatile uint32_t seq;
    // part[k][t][r]: s2_down * (down row r . act[k][t]) of entry k (tokens outside its mask left unwritten)
    alignas(64) float part[kCpuMaxEntries][kCpuMaxTokens][kLatent];
};

// One expert for 1..4 tokens: act = relu2(s2_up * (up . x)), part = s2_down * (down . act).
void cpu_expert_up_rows(const uint8_t* record, float s2_up, const float* x, int tokens, int mask, int row0, int row1,
                        float* act);   // act [kCpuMaxTokens][kInter]
void cpu_expert_down_rows(const uint8_t* record, float s2_down, const float* act, int tokens, int mask, int row0,
                          int row1, float* part);   // part [kCpuMaxTokens][kLatent]

// Routed experts of a short prompt chunk on the CPU: expert list[i] takes rows first[i] .. first[i] + count[i] of
// `out` ([rows][kLatent], FP32), row j for latent row x[row_token[j]] (FP16 bits, kLatent wide). Tokens go through
// the expert in groups of up to 4 while its record stays in the CPU caches; experts are spread over `threads`
// threads. part rows are s2_down * (down . relu2(s2_up * (up . x))), the device's per-row operations in FP32.
struct CpuPromptExpert {
    int expert = 0, first = 0, count = 0;
};
void cpu_prompt_experts(const uint8_t* records, int group, const float* s2, const std::vector<CpuPromptExpert>& list,
                        const int* row_token, const uint16_t* x, float* out, int threads);

// A thread pool polling the mailbox and answering each job: the up rows of every entry split over the threads,
// a barrier, then the down rows.
class CpuMissServer {
public:
    CpuMissServer(CpuJob* job, CpuReply* reply, const uint8_t* records, int threads);
    ~CpuMissServer();
    unsigned long long jobs() const { return jobs_.load(); }
    unsigned long long experts() const { return experts_.load(); }
    unsigned long long busy_ns() const { return busy_ns_.load(); }

private:
    void run(int id);
    void barrier(int gen, int round);
    CpuJob* job_;
    CpuReply* reply_;
    const uint8_t* records_;
    int threads_;
    std::vector<std::thread> pool_;
    std::atomic<bool> stop_{false};
    // current job (snapshot of the mailbox)
    std::unique_ptr<CpuJob> snap_;
    std::atomic<uint32_t> posted_{0};     // seq of the job being worked
    std::atomic<long long> arrived_{0};
    std::atomic<int> generation_{0};
    std::atomic<int> next_item_{0};
    std::vector<float> act_;              // [kCpuMaxEntries][kCpuMaxTokens][kInter]
    std::atomic<unsigned long long> jobs_{0}, experts_{0}, busy_ns_{0};
};

}  // namespace nmh
