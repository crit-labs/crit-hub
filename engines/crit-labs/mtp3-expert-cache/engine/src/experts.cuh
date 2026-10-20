// Routed experts: host-resident records, a device slot cache with a device-side directory, and the
// decode-time expert GEMVs that read missed experts straight from mapped host memory.
#pragma once
#include "common.cuh"
#include "cpu_mailbox.h"

namespace fnl {

// Host record of one expert (kHostExpertBytes): gate W, up W, down W (as in a device slot), then the
// three CSF fixed planes (gate, up, down). Exception words live in a separate pool.
// Device slot (kSlotBytes): gate W, up W, down W, then decoded E4M3 scale bytes (gate, up, down).
constexpr std::size_t kSlotScales = kExpertWeightBytes;
constexpr std::size_t kHostCsf = kExpertWeightBytes;

struct ExpertTables {
    int* slot_of = nullptr;          // [kLayers * kExperts], -1 when not resident
    int* owner = nullptr;            // [nslots], layer * kExperts + expert or -1
    uint8_t* ref = nullptr;          // [nslots] CLOCK reference bits
    uint32_t* stamp = nullptr;       // [nslots] step that last pinned the slot
    int* hand = nullptr;             // [1] CLOCK hand
    uint32_t* step = nullptr;        // [1] step counter
    const float* s2 = nullptr;       // [kLayers * kExperts * 3] global scales (gate, up, down)
    const uint32_t* exc_off = nullptr;   // [kLayers * kExperts * 3] word offset into exc_pool
    const uint32_t* exc_cnt = nullptr;   // [kLayers * kExperts * 3] words
    const uint32_t* exc_pool = nullptr;  // mapped host
    const uint8_t* const* host_layer = nullptr;  // [kLayers] mapped host base of the layer's records
    uint8_t* const* slot_addr = nullptr;  // [nslots] device address of each slot (own memory or lent)
    const uint8_t* valid = nullptr;  // [nslots] slot usable now (lent slots come and go)
    uint8_t* scratch = nullptr;      // kMaxUnion device slots for experts that are not cached
    int nslots = 0;
    // The MTP layer (layer index kLayers): every expert resident at fixed_base + e * kSlotBytes.
    uint8_t* fixed_base = nullptr;
    const float* fixed_s2 = nullptr; // [kExperts * 3]
    // CPU share of missed experts: the first round(share * misses) misses of a pass go to the CPU through the
    // mailbox (mapped pinned memory); cpu_state [0] last posted seq, [1] a job is pending for this layer.
    CpuJob* job = nullptr;
    float cpu_share = 0.f;
    int* cpu_state = nullptr;
    uint32_t* pass_first = nullptr;  // first step of the current decode pass
    // Statistics: [0] lookups, [1] misses.
    unsigned long long* stats = nullptr;
};

// Routing result of one token on one layer.
struct Route {
    int expert[kTopK];
    float weight[kTopK];
};
struct ExpertRef {
    const uint8_t* src;   // weights to read: device slot or mapped host record
    uint8_t* dst;         // device slot (or scratch) holding decoded scales; weights written through on a miss
    int miss;             // 0 hit, 1 read from host by the device, 2 computed by the CPU
    int write_through;
};

constexpr int kMaxPassTokens = 4;
constexpr int kMaxUnion = kMaxPassTokens * kTopK;
constexpr int kMtpLayer = kLayers;   // layer index of the MTP block in the expert tables

// The experts of one decode pass (1..4 tokens) on one layer: the union of the tokens' routes, each entry read
// once for every token routed to it.
struct ExpertPlan {
    int n;                               // union entries
    int expert[kMaxUnion];
    int mask[kMaxUnion];                 // bit t: token t routes to the entry
    float weight[kMaxUnion][kMaxPassTokens];  // routing weight of token t
    float s2[kMaxUnion][3];
    ExpertRef ref[kMaxUnion];
};

// router logits (512) -> softmax, top-10, renormalised weights.
void route_topk(const float* logits, Route* route, cudaStream_t stream);
// Union, directory lookup and CLOCK replacement for one decode pass of T tokens on one layer.
// With t.job, also posts the CPU's share of the misses (activations x [T][kHidden]).
void plan_experts(const ExpertTables& t, int layer, const Route* routes, int tokens, ExpertPlan* plan,
                  const float* x, cudaStream_t stream);
// Waits for the CPU's answer to this layer's job (if one was posted).
void cpu_wait(const ExpertTables& t, const CpuReply* reply, cudaStream_t stream);
// H[t][s][i] += (out[t][i] + cpu[t][i] when a job was posted) * inj[t][s]
void hc_combine_cpu(float* H, const float* out, const float* inj, int tokens, const CpuReply* reply,
                    const int* cpu_state, cudaStream_t stream);
// Start of a decode pass for the CLOCK protection of slots touched in it.
void begin_pass(const ExpertTables& t, cudaStream_t stream);
// Decode the CSF scales of missed experts into their destination slot.
void plan_miss_scales(const ExpertTables& t, int layer, const ExpertPlan* plan, int tokens, cudaStream_t stream);
// act[u][t][i] = silu(gate_u . x_t) * (up_u . x_t) for every entry u and token t routed to it.
void expert_gate_up(const ExpertPlan* plan, const float* x, float* act, int tokens, cudaStream_t stream);
// y[t] = sum_u w_ut down_u(act[u][t]) + shared_gate[t] * shared[t]. With part ([kMaxUnion][kMaxPassTokens]
// [kHidden] floats of workspace): per-entry blocks, then the sum over the entries in plan order.
void expert_down(const ExpertPlan* plan, const float* act, const float* shared, const float* shared_gate, float* y,
                 int tokens, cudaStream_t stream, float* part = nullptr);

}  // namespace fnl
