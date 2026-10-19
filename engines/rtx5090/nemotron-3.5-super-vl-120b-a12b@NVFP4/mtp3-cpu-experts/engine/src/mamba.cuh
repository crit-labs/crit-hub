// Mamba-2 mixer kernels (decode steps of 1..4 tokens and prompt chunks): causal depthwise conv with SiLU, the
// selective state-space recurrence in FP32 and the gated group RMS norm.
#pragma once
#include "common.cuh"

namespace nmh {

constexpr int kStateRing = 5;   // per-layer SSM states: the committed one plus one after each token of a 4-token pass
constexpr int kMaxTokensRec = 4;   // conv inputs recorded per decode pass
constexpr std::size_t kSsmFloats = std::size_t(kMHeads) * kMHeadDim * kMState;   // 1,048,576 per state

struct MambaWeights {
    const float* conv_w = nullptr;   // [kConvDim][4]
    const float* conv_b = nullptr;   // [kConvDim]
    const float* dt_bias = nullptr;  // [kMHeads]
    const float* neg_a = nullptr;    // [kMHeads] -exp(A_log)
    const float* d = nullptr;        // [kMHeads]
    const float* norm_w = nullptr;   // [kMInner]
    float dt_min = kDtMin;           // lower limit of dt after softplus
};

// xc[t][c] = silu(b[c] + sum_k w[c][k] * in[t - 3 + k][c]) over the window (state, then the T inputs), for the
// conv channels of proj (row t at proj + t * ld + kMInner). With update_state, state <- last 3 inputs.
// The decode form records the inputs in rec[T][kConvDim] for a later commit (rec may be null).
void conv_tokens(const MambaWeights& w, const float* proj, int ld, float* state, float* xc, float* rec, int tokens,
                 bool update_state, cudaStream_t stream);
// Sequential SSM over T tokens. The layer's states are ring[kStateRing] and the current one is ring[*cur].
// snapshots: the state after token t goes to ring[(*cur + 1 + t) % kStateRing] (the commit advances *cur);
// otherwise the final state replaces ring[*cur]. y[t][h*64+p] = C . state + D x. dt from proj[t][kMInner+kConvDim+h].
void ssm_tokens(const MambaWeights& w, const float* xc, const float* proj, int ld, float* ring, const int* cur,
                bool snapshots, float* y, int tokens, cudaStream_t stream);
// out[t] = rmsnorm_g(y[t] * silu(z[t])) * w per group of 1024, z at proj + t * ld. FP32 or FP16 output.
void gated_norm(const MambaWeights& w, const float* y, const float* proj, int ld, float* out, __half* out16,
                int tokens, cudaStream_t stream);
// After a pass whose first *n tokens are kept: each layer's conv state <- last 3 of (state, its recorded inputs)
// and every ring's current index advances by *n.
void commit_mamba(float* conv_states, const float* rec, int layers, int* cur, const int* n, cudaStream_t stream);

}  // namespace nmh
