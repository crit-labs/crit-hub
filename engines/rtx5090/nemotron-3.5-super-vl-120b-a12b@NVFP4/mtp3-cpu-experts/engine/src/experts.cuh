// Routed experts: pinned host records, a device slot cache with a device-side directory (CLOCK), and the
// decode-time expert GEMVs that read missed experts straight from mapped host memory and write them through into
// their new slot. Part of a pass's misses can go to the CPU (cpu_experts.h) through a mailbox in mapped memory.
#pragma once
#include "common.cuh"
#include "cpu_experts.h"

namespace nmh {

constexpr int kMaxPassTokens = 4;
constexpr int kMaxUnion = kMaxPassTokens * kTopK;   // 88
constexpr int kMtpGroup = kMoeLayers;              // expert-table group of the MTP block
constexpr int kGroups = kMoeLayers + 1;            // MoE layers plus the MTP block

struct ExpertTables {
    int* slot_of = nullptr;            // [kGroups * kExperts], -1 when not resident
    int* owner = nullptr;              // [nslots], group * kExperts + expert or -1
    uint8_t* ref = nullptr;            // [nslots] CLOCK reference bits
    uint32_t* stamp = nullptr;         // [nslots] step that last used the slot
    int* hand = nullptr;               // [1]
    uint32_t* step = nullptr;          // [1]
    uint32_t* pass_first = nullptr;    // [1] first step of the current decode pass
    const float* s2 = nullptr;         // [kGroups * kExperts * 2] global scales (up, down)
    const uint8_t* host = nullptr;     // mapped pinned records [kGroups][kExperts][kSlotBytes]
    uint8_t* const* slot_addr = nullptr;  // [nslots] device address of each slot (own memory or lent)
    const uint8_t* valid = nullptr;    // [nslots] slot usable now (lent slots come and go)
    int nslots = 0;
    unsigned long long* stats = nullptr;  // [0] lookups, [1] misses, [2] misses served by the CPU
    // CPU share of misses: with at least cpu_min misses in a pass, the first round(cpu_share * misses) go to the
    // CPU (not cached). cpu_state: [0] last posted seq, [1] this layer posted a job.
    CpuJob* job = nullptr;             // mapped pinned
    const CpuReply* reply = nullptr;   // mapped pinned
    float cpu_share = 0.f;
    int cpu_min = 1;
    int* cpu_state = nullptr;
};

struct Route {
    int expert[kTopK];
    float weight[kTopK];
};
struct ExpertRef {
    const uint8_t* src;   // device slot or mapped host record
    uint8_t* dst;         // new slot when the miss is written through
    int miss;             // 0 hit, 1 read from host memory by the device, 2 computed by the CPU
    int write_through;
    int cpu;              // entry of the CPU job (miss == 2)
    int pad;
};
// The experts of one decode pass (1..4 tokens) on one layer: the union of the tokens' routes.
struct ExpertPlan {
    int n;
    int expert[kMaxUnion];
    int mask[kMaxUnion];                       // bit t: token t routes to the entry
    float weight[kMaxUnion][kMaxPassTokens];   // routing weight of token t
    float s2[kMaxUnion][2];
    ExpertRef ref[kMaxUnion];
};

// router logits [T][512] -> sigmoid, top-22 by score + correction bias, weights normalized and scaled by 5.
void route(const float* logits, const float* bias, Route* routes, int tokens, cudaStream_t stream);
void begin_pass(const ExpertTables& t, cudaStream_t stream);
// Union, directory lookup and CLOCK replacement for one decode pass on group g (layer's MoE index or kMtpGroup).
// x [T][kLatent]: the experts' input, copied into the CPU job when the CPU takes part of the misses.
void plan_experts(const ExpertTables& t, int group, const Route* routes, int tokens, ExpertPlan* plan,
                  const float* x, cudaStream_t stream);
// act[u][t][i] = relu2(s2_up * (up_u . x_t)) for every device entry u and token t routed to it; x [T][kLatent].
// Per row: 16 element accumulators over the groups in order, then the fixed tree (cpu_experts.h).
void expert_up(const ExpertPlan* plan, const float* x, float* act, int tokens, cudaStream_t stream);
// part[u][t][r] = s2_down * (down_u row r . act[u][t]); after the CPU's answer (when a job was posted),
// y[t][r] = sum_u w_ut part[u][t][r] in plan order, CPU entries read from the reply.
void expert_down(const ExpertTables& t, const ExpertPlan* plan, const float* act, float* part, float* y, int tokens,
                 cudaStream_t stream);

}  // namespace nmh
