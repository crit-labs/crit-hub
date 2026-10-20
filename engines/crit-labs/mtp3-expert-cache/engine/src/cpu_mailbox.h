// Mailbox in mapped pinned memory through which the decode pass hands part of a layer's missed experts to the
// CPU: the device writes a job (experts, routing, activations) and bumps seq; the CPU computes the experts'
// weighted sum per token and answers with the same seq.
#pragma once
#include <cstdint>

namespace fnl {

constexpr int kCpuMaxEntries = 40;   // kMaxUnion
constexpr int kCpuMaxTokens = 4;
constexpr int kCpuHidden = 2560;

struct CpuJob {
    volatile uint32_t seq;
    int layer, tokens, count;
    int expert[kCpuMaxEntries];
    int mask[kCpuMaxEntries];
    float weight[kCpuMaxEntries][kCpuMaxTokens];
    unsigned long long slot[kCpuMaxEntries];   // device slot to fill in the background (0: none)
    alignas(64) float x[kCpuMaxTokens][kCpuHidden];
};

struct CpuReply {
    volatile uint32_t seq;
    alignas(64) float y[kCpuMaxTokens][kCpuHidden];
};

}  // namespace fnl
