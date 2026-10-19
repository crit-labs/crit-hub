// Kernels used only by the prompt path: routed experts grouped by expert.
#pragma once
#include "common.cuh"
#include "experts.cuh"

namespace nmh {

// counts[s][e]: routes of tokens in sub-chunk s (sub tokens each) to expert e. counts zeroed by the caller.
void count_routes(const Route* routes, int tokens, int sub, int* counts, cudaStream_t stream);
// For tokens [t0, t1): route_token[off[e] + i] = t - t0, pos_of[t][k] = that row. cursor zeroed by the caller.
// fixed (optional, [t1 - t0][kTopK]): routes with fixed[] >= 0 take that row (rows computed elsewhere) and are
// left out of the expert groups.
void scatter_routes(const Route* routes, int t0, int t1, const int* offsets, int* cursor, int* route_token,
                    int* pos_of, cudaStream_t stream, const int* fixed = nullptr);
// out[t][i] = f16(sum_k w_k route_out[pos_of[t][k]][i] * in_scale), t in [t0, t1); out rows relative to t0.
void moe_combine(const Route* routes, const int* pos_of, const float* route_out, __half* out, int t0, int t1,
                 cudaStream_t stream);

}  // namespace nmh
