// Log-probabilities of chosen ids over logits rows (natural log, normalized over the valid vocabulary).
#include "score.cuh"

namespace fnl {
namespace kern {
// grid n rows, block 1024.
__global__ void row_scores_kernel(const float* logits, int ld, int valid, const int* targets, const int* queries,
                                  int per, float* tlp, float* qlp, int* amax) {
    const int r = blockIdx.x;
    const float* x = logits + std::size_t(r) * ld;
    __shared__ float sv[32];
    __shared__ int si[32];
    __shared__ double sd[32];
    __shared__ float mx;
    __shared__ int mi;
    __shared__ double lse;
    float v = -INFINITY;
    int idx = 0x7fffffff;
    for(int i = threadIdx.x; i < valid; i += blockDim.x) {
        const float y = x[i];
        if(y > v || (y == v && i < idx)) { v = y; idx = i; }
    }
#pragma unroll
    for(int o = 16; o > 0; o >>= 1) {
        const float ov = __shfl_xor_sync(0xffffffffu, v, o);
        const int oi = __shfl_xor_sync(0xffffffffu, idx, o);
        if(ov > v || (ov == v && oi < idx)) { v = ov; idx = oi; }
    }
    if((threadIdx.x & 31) == 0) { sv[threadIdx.x >> 5] = v; si[threadIdx.x >> 5] = idx; }
    __syncthreads();
    if(threadIdx.x == 0) {
        float bv = sv[0];
        int bi = si[0];
        for(int w = 1; w < int(blockDim.x >> 5); ++w)
            if(sv[w] > bv || (sv[w] == bv && si[w] < bi)) { bv = sv[w]; bi = si[w]; }
        mx = bv;
        mi = bi;
    }
    __syncthreads();
    const float m = mx;
    double s = 0.0;
    for(int i = threadIdx.x; i < valid; i += blockDim.x) s += double(expf(x[i] - m));
#pragma unroll
    for(int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
    if((threadIdx.x & 31) == 0) sd[threadIdx.x >> 5] = s;
    __syncthreads();
    if(threadIdx.x == 0) {
        double t = 0.0;
        for(int w = 0; w < int(blockDim.x >> 5); ++w) t += sd[w];
        lse = double(m) + log(t);
    }
    __syncthreads();
    if(threadIdx.x == 0) {
        if(amax) amax[r] = mi;
        if(tlp) {
            const int t = targets[r];
            tlp[r] = (t >= 0 && t < valid) ? float(double(x[t]) - lse) : -INFINITY;
        }
    }
    for(int j = threadIdx.x; j < per; j += blockDim.x) {
        const int q = queries[std::size_t(r) * per + j];
        qlp[std::size_t(r) * per + j] = (q >= 0 && q < valid) ? float(double(x[q]) - lse) : -INFINITY;
    }
}
}  // namespace kern
using namespace kern;

void row_scores(const float* logits, int ld, int valid, int rows, const int* targets, const int* queries, int per,
                float* tlp, float* qlp, int* amax, cudaStream_t stream) {
    if(rows <= 0) return;
    row_scores_kernel<<<rows, 1024, 0, stream>>>(logits, ld, valid, targets, queries, per, tlp, qlp, amax);
    FNL_CUDA(cudaGetLastError());
}

}  // namespace fnl
