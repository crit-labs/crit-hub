#include "prefill.cuh"

namespace nmh {
namespace {

__global__ void count_routes_kernel(const Route* routes, int tokens, int sub, int* counts) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if(t >= tokens) return;
    const int s = t / sub;
#pragma unroll
    for(int k = 0; k < kTopK; ++k) atomicAdd(&counts[s * kExperts + routes[t].expert[k]], 1);
}

// Row order within an expert depends on atomics, but every GEMM row is computed independently and the combine
// sums in route order, so outputs do not depend on it.
__global__ void scatter_routes_kernel(const Route* routes, int t0, int t1, const int* offsets, int* cursor,
                                      int* route_token, int* pos_of, const int* fixed) {
    const int t = t0 + blockIdx.x * blockDim.x + threadIdx.x;
    if(t >= t1) return;
#pragma unroll
    for(int k = 0; k < kTopK; ++k) {
        if(fixed && fixed[(t - t0) * kTopK + k] >= 0) {
            pos_of[(t - t0) * kTopK + k] = fixed[(t - t0) * kTopK + k];
            continue;
        }
        const int e = routes[t].expert[k];
        const int p = offsets[e] + atomicAdd(&cursor[e], 1);
        route_token[p] = t - t0;
        pos_of[(t - t0) * kTopK + k] = p;
    }
}

// grid (t1 - t0), block 256
__global__ void moe_combine_kernel(const Route* routes, const int* pos_of, const float* route_out, __half* out,
                                   int t0) {
    const int r = blockIdx.x, t = t0 + r;
    __shared__ int pos[kTopK];
    __shared__ float w[kTopK];
    if(threadIdx.x < kTopK) {
        pos[threadIdx.x] = pos_of[r * kTopK + threadIdx.x];
        w[threadIdx.x] = routes[t].weight[threadIdx.x];
    }
    __syncthreads();
    for(int i = threadIdx.x; i < kLatent; i += blockDim.x) {
        float acc = 0.f;
#pragma unroll
        for(int k = 0; k < kTopK; ++k) acc = fmaf(w[k], route_out[std::size_t(pos[k]) * kLatent + i], acc);
        out[std::size_t(r) * kLatent + i] = __float2half_rn(acc);
    }
}

}  // namespace

void count_routes(const Route* routes, int tokens, int sub, int* counts, cudaStream_t stream) {
    count_routes_kernel<<<(tokens + 255) / 256, 256, 0, stream>>>(routes, tokens, sub, counts);
    NMH_CUDA(cudaGetLastError());
}
void scatter_routes(const Route* routes, int t0, int t1, const int* offsets, int* cursor, int* route_token,
                    int* pos_of, cudaStream_t stream, const int* fixed) {
    scatter_routes_kernel<<<(t1 - t0 + 255) / 256, 256, 0, stream>>>(routes, t0, t1, offsets, cursor, route_token,
                                                                      pos_of, fixed);
    NMH_CUDA(cudaGetLastError());
}
void moe_combine(const Route* routes, const int* pos_of, const float* route_out, __half* out, int t0, int t1,
                 cudaStream_t stream) {
    moe_combine_kernel<<<t1 - t0, 256, 0, stream>>>(routes, pos_of, route_out, out, t0);
    NMH_CUDA(cudaGetLastError());
}

}  // namespace nmh
