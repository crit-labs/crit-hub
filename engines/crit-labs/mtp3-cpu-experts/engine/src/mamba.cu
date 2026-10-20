#include "mamba.cuh"

namespace nmh {
namespace {

// grid (ceil(kConvDim / 256), T), block 256. The window of token t is (state[0..2], in[0..T-1])[t .. t + 3].
__global__ void conv_kernel(MambaWeights w, const float* proj, int ld, const float* state, float* xc, float* rec,
                            int tokens) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    const int t = blockIdx.y;
    if(c >= kConvDim) return;
    const float* in = proj + kMInner + c;
    float acc = w.conv_b[c];
#pragma unroll
    for(int k = 0; k < kConv; ++k) {
        const int i = t - (kConv - 1) + k;   // input index, negative: state
        const float v = i >= 0 ? in[std::size_t(i) * ld] : state[c * 3 + (3 + i)];
        acc = fmaf(w.conv_w[c * kConv + k], v, acc);
    }
    xc[std::size_t(t) * kConvDim + c] = siluf(acc);
    if(rec) rec[std::size_t(t) * kConvDim + c] = in[std::size_t(t) * ld];
}

// state <- last 3 of (state, in[0..T-1]).
__global__ void conv_state_kernel(const float* proj, int ld, float* state, int tokens) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if(c >= kConvDim) return;
    const float* in = proj + kMInner + c;
    float win[3];
#pragma unroll
    for(int j = 0; j < 3; ++j) {
        const int i = tokens + j - 3;   // index into (state, inputs) shifted by 3
        win[j] = i >= 0 ? in[std::size_t(i) * ld] : state[c * 3 + (3 + i)];
    }
#pragma unroll
    for(int j = 0; j < 3; ++j) state[c * 3 + j] = win[j];
}

constexpr int kScanThreads = 256;
constexpr int kTile = 32;
constexpr int kBPad = kMState + kMState / 32 * 4;   // 144: n + (n / 32) * 4

// grid kMHeads, block 256: thread (p = tid / 4, q = tid % 4) holds state[p][q*32 .. q*32+31] in registers.
__global__ void __launch_bounds__(kScanThreads) ssm_kernel(MambaWeights w, const float* xc, const float* proj, int ld,
                                                           float* ring, const int* cur_ptr, int snapshots, float* y,
                                                           int tokens) {
    const int h = blockIdx.x, g = h / (kMHeads / kMGroups);
    const int tid = threadIdx.x, p = tid >> 2, q = tid & 3;
    __shared__ __align__(16) float xs[kTile][kMHeadDim];
    __shared__ __align__(16) float bs[kTile][kBPad];
    __shared__ __align__(16) float cs[kTile][kBPad];
    __shared__ float dts[kTile], das[kTile];
    const int cur = *cur_ptr;
    float* state_in = ring + std::size_t(cur) * kSsmFloats + (std::size_t(h) * kMHeadDim + p) * kMState + q * 32;
    float s[32];
#pragma unroll
    for(int j = 0; j < 8; ++j) {
        const float4 v = *reinterpret_cast<const float4*>(state_in + 4 * j);
        s[4 * j] = v.x; s[4 * j + 1] = v.y; s[4 * j + 2] = v.z; s[4 * j + 3] = v.w;
    }
    const float a = w.neg_a[h], dbias = w.dt_bias[h], dd = w.d[h];
    for(int t0 = 0; t0 < tokens; t0 += kTile) {
        const int nt = min(kTile, tokens - t0);
        __syncthreads();
        for(int i = tid; i < nt * kMHeadDim; i += kScanThreads) {
            const int tt = i / kMHeadDim, pp = i % kMHeadDim;
            xs[tt][pp] = xc[std::size_t(t0 + tt) * kConvDim + h * kMHeadDim + pp];
        }
        for(int i = tid; i < nt * kMState; i += kScanThreads) {
            const int tt = i / kMState, n = i % kMState;
            const float* row = xc + std::size_t(t0 + tt) * kConvDim + kMInner;
            bs[tt][n + (n >> 5) * 4] = row[g * kMState + n];
            cs[tt][n + (n >> 5) * 4] = row[kMGroups * kMState + g * kMState + n];
        }
        if(tid < nt) {
            float dt = softplusf(proj[std::size_t(t0 + tid) * ld + kMInner + kConvDim + h] + dbias);
            dt = fmaxf(dt, w.dt_min);
            dts[tid] = dt;
            das[tid] = expf(dt * a);
        }
        __syncthreads();
        for(int tt = 0; tt < nt; ++tt) {
            const float x = xs[tt][p];
            const float bx = dts[tt] * x, da = das[tt];
            const float* b = &bs[tt][q * 36];
            const float* c = &cs[tt][q * 36];
            float acc = 0.f;
#pragma unroll
            for(int j = 0; j < 8; ++j) {
                const float4 bv = *reinterpret_cast<const float4*>(b + 4 * j);
                const float4 cv = *reinterpret_cast<const float4*>(c + 4 * j);
                s[4 * j] = fmaf(s[4 * j], da, bx * bv.x);
                s[4 * j + 1] = fmaf(s[4 * j + 1], da, bx * bv.y);
                s[4 * j + 2] = fmaf(s[4 * j + 2], da, bx * bv.z);
                s[4 * j + 3] = fmaf(s[4 * j + 3], da, bx * bv.w);
                acc = fmaf(s[4 * j], cv.x, acc);
                acc = fmaf(s[4 * j + 1], cv.y, acc);
                acc = fmaf(s[4 * j + 2], cv.z, acc);
                acc = fmaf(s[4 * j + 3], cv.w, acc);
            }
            acc += __shfl_xor_sync(0xffffffffu, acc, 1);
            acc += __shfl_xor_sync(0xffffffffu, acc, 2);
            if(q == 0) y[std::size_t(t0 + tt) * kMInner + h * kMHeadDim + p] = fmaf(dd, x, acc);
            if(snapshots) {
                float* dst = ring + std::size_t((cur + 1 + t0 + tt) % kStateRing) * kSsmFloats +
                             (std::size_t(h) * kMHeadDim + p) * kMState + q * 32;
#pragma unroll
                for(int j = 0; j < 8; ++j)
                    *reinterpret_cast<float4*>(dst + 4 * j) = make_float4(s[4 * j], s[4 * j + 1], s[4 * j + 2], s[4 * j + 3]);
            }
        }
    }
    if(!snapshots) {
#pragma unroll
        for(int j = 0; j < 8; ++j)
            *reinterpret_cast<float4*>(state_in + 4 * j) = make_float4(s[4 * j], s[4 * j + 1], s[4 * j + 2], s[4 * j + 3]);
    }
}

// grid (kMGroups, T), block 256: 4 elements per thread.
__global__ void __launch_bounds__(256) gated_norm_kernel(MambaWeights w, const float* y, const float* proj, int ld,
                                                         float* out, __half* out16) {
    const int g = blockIdx.x, t = blockIdx.y, tid = threadIdx.x;
    __shared__ float red[8];
    const int base = g * kNormGroup + tid * 4;
    const float4 yv = *reinterpret_cast<const float4*>(y + std::size_t(t) * kMInner + base);
    const float* z = proj + std::size_t(t) * ld + base;
    float v[4] = {yv.x * siluf(z[0]), yv.y * siluf(z[1]), yv.z * siluf(z[2]), yv.w * siluf(z[3])};
    float ss = v[0] * v[0] + v[1] * v[1] + v[2] * v[2] + v[3] * v[3];
    ss = warp_sum(ss);
    if((tid & 31) == 0) red[tid >> 5] = ss;
    __syncthreads();
    float tot = 0.f;
#pragma unroll
    for(int i = 0; i < 8; ++i) tot += red[i];
    const float r = rsqrtf(tot / float(kNormGroup) + kEps);
#pragma unroll
    for(int i = 0; i < 4; ++i) {
        const float o = v[i] * r * w.norm_w[base + i];
        if(out) out[std::size_t(t) * kMInner + base + i] = o;
        else out16[std::size_t(t) * kMInner + base + i] = __float2half_rn(o);
    }
}

// grid (ceil(kConvDim / 256), layers): conv state <- last 3 of (state, rec[0..n-1]).
__global__ void commit_conv_kernel(float* states, const float* rec, const int* n_ptr) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    const int l = blockIdx.y;
    if(c >= kConvDim) return;
    const int n = *n_ptr;
    float* st = states + (std::size_t(l) * kConvDim + c) * 3;
    const float* in = rec + std::size_t(l) * kMaxTokensRec * kConvDim + c;
    float win[3];
#pragma unroll
    for(int j = 0; j < 3; ++j) {
        const int i = n + j - 3;
        win[j] = i >= 0 ? in[std::size_t(i) * kConvDim] : st[3 + i];
    }
#pragma unroll
    for(int j = 0; j < 3; ++j) st[j] = win[j];
}
__global__ void commit_cur_kernel(int* cur, const int* n) {
    if(threadIdx.x == 0) *cur = (*cur + *n) % kStateRing;
}

}  // namespace

void conv_tokens(const MambaWeights& w, const float* proj, int ld, float* state, float* xc, float* rec, int tokens,
                 bool update_state, cudaStream_t stream) {
    conv_kernel<<<dim3((kConvDim + 255) / 256, tokens), 256, 0, stream>>>(w, proj, ld, state, xc, rec, tokens);
    NMH_CUDA(cudaGetLastError());
    if(update_state) {
        conv_state_kernel<<<(kConvDim + 255) / 256, 256, 0, stream>>>(proj, ld, state, tokens);
        NMH_CUDA(cudaGetLastError());
    }
}
void ssm_tokens(const MambaWeights& w, const float* xc, const float* proj, int ld, float* ring, const int* cur,
                bool snapshots, float* y, int tokens, cudaStream_t stream) {
    ssm_kernel<<<kMHeads, kScanThreads, 0, stream>>>(w, xc, proj, ld, ring, cur, snapshots ? 1 : 0, y, tokens);
    NMH_CUDA(cudaGetLastError());
}
void gated_norm(const MambaWeights& w, const float* y, const float* proj, int ld, float* out, __half* out16,
                int tokens, cudaStream_t stream) {
    gated_norm_kernel<<<dim3(kMGroups, tokens), 256, 0, stream>>>(w, y, proj, ld, out, out16);
    NMH_CUDA(cudaGetLastError());
}
void commit_mamba(float* conv_states, const float* rec, int layers, int* cur, const int* n, cudaStream_t stream) {
    commit_conv_kernel<<<dim3((kConvDim + 255) / 256, layers), 256, 0, stream>>>(conv_states, rec, n);
    NMH_CUDA(cudaGetLastError());
    commit_cur_kernel<<<1, 32, 0, stream>>>(cur, n);
    NMH_CUDA(cudaGetLastError());
}

}  // namespace nmh
