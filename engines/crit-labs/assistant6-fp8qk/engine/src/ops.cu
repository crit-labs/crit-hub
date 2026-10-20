#include "ops.cuh"

#include <cuda_fp8.h>

namespace g4 {
namespace kops {

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for(int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}
__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
    for(int o = 16; o; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}
// Sum over the block (blockDim.x a multiple of 32, at most 1024); every thread gets the total.
__device__ float block_sum(float v, float* red) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nw = blockDim.x >> 5;
    v = warp_sum(v);
    __syncthreads();
    if(lane == 0) red[warp] = v;
    __syncthreads();
    float t = 0.f;
    for(int i = 0; i < nw; ++i) t += red[i];
    return t;
}
__device__ float block_max(float v, float* red) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nw = blockDim.x >> 5;
    v = warp_max(v);
    __syncthreads();
    if(lane == 0) red[warp] = v;
    __syncthreads();
    float t = 0.f;
    for(int i = 0; i < nw; ++i) t = fmaxf(t, red[i]);
    return t;
}
__device__ __forceinline__ float gelu_tanh(float v) {
    return 0.5f * v * (1.f + tanhf(0.7978845608028654f * (v + 0.044715f * v * v * v)));
}
__device__ __forceinline__ std::uint8_t to_e4m3(float v) {
    return static_cast<std::uint8_t>(__nv_cvt_float_to_fp8(v, __NV_SATFINITE, __NV_E4M3));
}
__device__ __forceinline__ float from_e4m3(std::uint8_t c) {
    const __half_raw h = __nv_cvt_fp8_to_halfraw(static_cast<__nv_fp8_storage_t>(c), __NV_E4M3);
    return __half2float(*reinterpret_cast<const __half*>(&h));
}
__device__ __forceinline__ std::uint32_t e2m1_pair(float lo, float hi) {
    std::uint32_t r;
    asm("{ .reg .b8 t; cvt.rn.satfinite.e2m1x2.f32 t, %1, %2; cvt.u32.u8 %0, t; }" : "=r"(r) : "f"(hi), "f"(lo));
    return r;
}
__device__ __forceinline__ void fp4_block(const float (&v)[16], float bm, float gs, uint2& codes, std::uint8_t& scode) {
    scode = to_e4m3(bm / 6.f * gs);
    const float sdec = from_e4m3(scode);
    const float inv = sdec > 0.f ? gs / sdec : 0.f;
    std::uint32_t lo = 0, hi = 0;
#pragma unroll
    for(int i = 0; i < 4; ++i) lo |= e2m1_pair(v[2 * i] * inv, v[2 * i + 1] * inv) << (8 * i);
#pragma unroll
    for(int i = 0; i < 4; ++i) hi |= e2m1_pair(v[8 + 2 * i] * inv, v[8 + 2 * i + 1] * inv) << (8 * i);
    codes = make_uint2(lo, hi);
}

// ------------------------------------------------------------------------------------------------ embed / norms
__global__ void embed_kernel(const int* __restrict__ tok, const __nv_bfloat16* __restrict__ table, int dim, float scale,
                             float* __restrict__ out, int ldo) {
    pdl_wait();
    pdl_trigger();
    const int n = blockIdx.x;
    const std::size_t row = static_cast<std::size_t>(tok[n]) * dim;
    for(int i = threadIdx.x; i < dim / 8; i += blockDim.x) {
        const uint4 u = *reinterpret_cast<const uint4*>(table + row + i * 8);
        const __nv_bfloat162* b = reinterpret_cast<const __nv_bfloat162*>(&u);
        const float2 f0 = __bfloat1622float2(b[0]), f1 = __bfloat1622float2(b[1]), f2 = __bfloat1622float2(b[2]),
                     f3 = __bfloat1622float2(b[3]);
        float4* o = reinterpret_cast<float4*>(out + static_cast<std::size_t>(n) * ldo + i * 8);
        o[0] = make_float4(f0.x * scale, f0.y * scale, f1.x * scale, f1.y * scale);
        o[1] = make_float4(f2.x * scale, f2.y * scale, f3.x * scale, f3.y * scale);
    }
}

__global__ void rmsnorm_kernel(const float* __restrict__ x, int ldx, const float* __restrict__ w, int dim, __nv_bfloat16* __restrict__ out,
                               int ldo) {
    pdl_wait();
    pdl_trigger();
    __shared__ float red[32];
    const int n = blockIdx.x;
    const float* xr = x + static_cast<std::size_t>(n) * ldx;
    float ss = 0.f;
    for(int i = threadIdx.x; i < dim; i += blockDim.x) ss += xr[i] * xr[i];
    ss = block_sum(ss, red);
    const float r = rsqrtf(ss / dim + kEps);
    for(int i = threadIdx.x; i < dim; i += blockDim.x)
        out[static_cast<std::size_t>(n) * ldo + i] = __float2bfloat16_rn(xr[i] * r * w[i]);
}

constexpr int kSandPer = 8;
__global__ void __launch_bounds__(768) sandwich_kernel(const float* __restrict__ v, int ldv, float* __restrict__ y, int ldy,
                                                       const float* __restrict__ w_post, const float* __restrict__ scalar,
                                                       const float* __restrict__ w_next, __nv_bfloat16* __restrict__ h, int ldh,
                                                       int dim) {
    pdl_wait();
    pdl_trigger();
    __shared__ float red[32];
    const int n = blockIdx.x, bd = blockDim.x, tid = threadIdx.x;
    const float* vr = v + static_cast<std::size_t>(n) * ldv;
    float* yr = y + static_cast<std::size_t>(n) * ldy;
    float vv[kSandPer], yy[kSandPer];
    float ss = 0.f;
#pragma unroll
    for(int j = 0; j < kSandPer; ++j) {
        const int i = tid + j * bd;
        vv[j] = i < dim ? vr[i] : 0.f;
        ss += vv[j] * vv[j];
    }
    ss = block_sum(ss, red);
    const float r1 = rsqrtf(ss / dim + kEps);
    const float sc = scalar ? *scalar : 1.f;
    float s2 = 0.f;
#pragma unroll
    for(int j = 0; j < kSandPer; ++j) {
        const int i = tid + j * bd;
        if(i < dim) {
            yy[j] = (yr[i] + vv[j] * r1 * w_post[i]) * sc;
            yr[i] = yy[j];
            s2 += yy[j] * yy[j];
        }
    }
    s2 = block_sum(s2, red);
    const float r2 = rsqrtf(s2 / dim + kEps);
#pragma unroll
    for(int j = 0; j < kSandPer; ++j) {
        const int i = tid + j * bd;
        if(i < dim) h[static_cast<std::size_t>(n) * ldh + i] = __float2bfloat16_rn(yy[j] * r2 * w_next[i]);
    }
}

// ------------------------------------------------------------------------------------------------ attention prep
// Block (row n, slot); one thread per dimension. Slots: queries 0..31, keys 32..47, values 48..63.
__global__ void __launch_bounds__(kSHd) slide_prep_kernel(Proj qkv, Pos pos, const float* __restrict__ qn,
                                                         const float* __restrict__ kn, const float* __restrict__ inv,
                                                         __nv_bfloat16* __restrict__ q_out, int ldq, SlideCache c, int first) {
    pdl_wait();
    pdl_trigger();
    __shared__ float red[8];
    __shared__ float rot[kSHd];
    const int n = blockIdx.x, slot = blockIdx.y + first, d = threadIdx.x;
    const int p = (pos.dev ? *pos.dev : pos.host) + n;
    int col;
    if(slot < kHeads) col = slot * kSHd + d;
    else if(slot < kHeads + kSKv) col = kSQ + (slot - kHeads) * kSHd + d;
    else col = kSQ + kSKvRows + (slot - kHeads - kSKv) * kSHd + d;
    float x = qkv.at(n, col);
    const float ss = block_sum(x * x, red);
    const float rs = rsqrtf(ss / kSHd + kEps);
    if(slot < kHeads + kSKv) {
        x = x * rs * (slot < kHeads ? qn : kn)[d];
        rot[d] = x;
        __syncthreads();
        const float ang = static_cast<float>(p) * inv[d & (kSHd / 2 - 1)];
        float sn, cs;
        sincosf(ang, &sn, &cs);
        x = d < kSHd / 2 ? x * cs - rot[d + kSHd / 2] * sn : x * cs + rot[d - kSHd / 2] * sn;
    } else {
        x = x * rs;
    }
    if(slot < kHeads) {
        q_out[static_cast<std::size_t>(n) * ldq + slot * kSHd + d] = __float2bfloat16_rn(x);
        return;
    }
    const bool is_k = slot < kHeads + kSKv;
    const int g = is_k ? slot - kHeads : slot - kHeads - kSKv;
    const float amax = block_max(fabsf(x), red);
    const float sc = amax > 0.f ? amax / 448.f : 1.f;
    const std::size_t sl = static_cast<std::size_t>(p % c.ring);
    (is_k ? c.k : c.v)[(sl * kSKv + g) * kSHd + d] = to_e4m3(x / sc);
    if(d == 0) (is_k ? c.ks : c.vs)[sl * kSKv + g] = sc;
}

// Block (row n, slot); one thread per dimension. Slots: queries 0..31, key/value heads 32..35.
__global__ void __launch_bounds__(kGHd) global_prep_kernel(Proj qkv, Pos pos, const float* __restrict__ qn,
                                                          const float* __restrict__ inv, __nv_bfloat16* __restrict__ q_out,
                                                          int ldq, GlobalCache c, int first) {
    const float* kfold = c.kw;
    pdl_wait();
    pdl_trigger();
    __shared__ float red[16];
    __shared__ float rot[kGHd];
    const int n = blockIdx.x, slot = blockIdx.y + first, d = threadIdx.x;
    const int p = (pos.dev ? *pos.dev : pos.host) + n;
    const int col = slot < kHeads ? slot * kGHd + d : kGQ + (slot - kHeads) * kGHd + d;
    float x = qkv.at(n, col);
    const float ss = block_sum(x * x, red);
    const float rs = rsqrtf(ss / kGHd + kEps);
    if(slot < kHeads) {
        x = x * rs * qn[d];
        rot[d] = x;
        __syncthreads();
        const int i = d & (kGHd / 2 - 1);
        if(i < kGRopePairs) {
            float sn, cs;
            sincosf(static_cast<float>(p) * inv[i], &sn, &cs);
            x = d < kGHd / 2 ? x * cs - rot[d + kGHd / 2] * sn : x * cs + rot[d - kGHd / 2] * sn;
        } else if(kfold) {
            x *= kfold[d];  // keys there are u * w: w moves to the query (attn.cuh)
        }
        q_out[static_cast<std::size_t>(n) * ldq + slot * kGHd + d] = __float2bfloat16_rn(x);
        return;
    }
    const int g = slot - kHeads;
    x = x * rs;
    if(c.kr) {
        // Rotated key dims RoPE(u * w) for d < 64 and 256 <= d < 320, quantized with one scale per (position, head).
        rot[d] = x * c.kw[d];
        __syncthreads();
        const int i = d & (kGHd / 2 - 1);
        float k = 0.f;
        if(i < kGRopePairs) {
            float sn, cs;
            sincosf(static_cast<float>(p) * inv[i], &sn, &cs);
            k = d < kGHd / 2 ? rot[d] * cs - rot[d + kGHd / 2] * sn : rot[d] * cs + rot[d - kGHd / 2] * sn;
        }
        const float km = block_max(fabsf(k), red);
        const float ks = km > 0.f ? km / 448.f : 1.f;
        if(i < kGRopePairs)
            c.kr[(static_cast<std::size_t>(p) * kGKv + g) * (2 * kGRopePairs) + (d < kGHd / 2 ? i : kGRopePairs + i)] = to_e4m3(k / ks);
        if(d == 0) c.krs[static_cast<std::size_t>(p) * kGKv + g] = ks;
    }
    const float amax = block_max(fabsf(x), red);
    const float sc = amax > 0.f ? amax / 448.f : 1.f;
    c.u[(static_cast<std::size_t>(p) * kGKv + g) * kGHd + d] = to_e4m3(x / sc);
    if(d == 0) c.us[static_cast<std::size_t>(p) * kGKv + g] = sc;
}

// ------------------------------------------------------------------------------------------------ head
__global__ void argmax_kernel(const float* __restrict__ logits, int ld, int vocab, int* __restrict__ out) {
    pdl_wait();
    pdl_trigger();
    __shared__ float sv[32];
    __shared__ int si[32];
    const int n = blockIdx.x;
    const float* row = logits + static_cast<std::size_t>(n) * ld;
    float best = -INFINITY;
    int bi = 0x7fffffff;
    for(int i = threadIdx.x; i < vocab; i += blockDim.x) {
        const float v = row[i];
        if(v > best) { best = v; bi = i; }
    }
#pragma unroll
    for(int o = 16; o; o >>= 1) {
        const float ov = __shfl_xor_sync(0xffffffffu, best, o);
        const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
        if(ov > best || (ov == best && oi < bi)) { best = ov; bi = oi; }
    }
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    if(lane == 0) { sv[warp] = best; si[warp] = bi; }
    __syncthreads();
    if(threadIdx.x == 0) {
        for(int w = 1; w < static_cast<int>(blockDim.x >> 5); ++w)
            if(sv[w] > best || (sv[w] == best && si[w] < bi)) { best = sv[w]; bi = si[w]; }
        out[n] = bi;
    }
}

// Stage 1: block (slice, column) -> best (value, index) of its slice; stage 2: one warp per column merges the slices.
__global__ void __launch_bounds__(256) argmax_part_kernel(const float* __restrict__ logits, int ld, int vocab,
                                                         float2* __restrict__ scratch) {
    pdl_wait();
    pdl_trigger();
    __shared__ float sv[8];
    __shared__ int si[8];
    const int part = blockIdx.x, n = blockIdx.y;
    const int per = (vocab + kArgParts - 1) / kArgParts, lo = part * per, hi = min(vocab, lo + per);
    const float* row = logits + static_cast<std::size_t>(n) * ld;
    float best = -INFINITY;
    int bi = 0x7fffffff;
    for(int i = lo + threadIdx.x; i < hi; i += blockDim.x) {
        const float v = row[i];
        if(v > best) { best = v; bi = i; }
    }
#pragma unroll
    for(int o = 16; o; o >>= 1) {
        const float ov = __shfl_xor_sync(0xffffffffu, best, o);
        const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
        if(ov > best || (ov == best && oi < bi)) { best = ov; bi = oi; }
    }
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    if(lane == 0) { sv[warp] = best; si[warp] = bi; }
    __syncthreads();
    if(threadIdx.x == 0) {
        for(int w = 1; w < 8; ++w)
            if(sv[w] > best || (sv[w] == best && si[w] < bi)) { best = sv[w]; bi = si[w]; }
        scratch[n * kArgParts + part] = make_float2(best, __int_as_float(bi));
    }
}
__global__ void argmax_merge_kernel(const float2* __restrict__ scratch, int* __restrict__ out) {
    pdl_wait();
    pdl_trigger();
    const int n = blockIdx.x, lane = threadIdx.x;
    float best = -INFINITY;
    int bi = 0x7fffffff;
    for(int p = lane; p < kArgParts; p += 32) {
        const float2 e = scratch[n * kArgParts + p];
        const int id = __float_as_int(e.y);
        if(e.x > best || (e.x == best && id < bi)) { best = e.x; bi = id; }
    }
#pragma unroll
    for(int o = 16; o; o >>= 1) {
        const float ov = __shfl_xor_sync(0xffffffffu, best, o);
        const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
        if(ov > best || (ov == best && oi < bi)) { best = ov; bi = oi; }
    }
    if(lane == 0) out[n] = bi;
}

__global__ void logprob_kernel(const float* __restrict__ logits, int ld, const int* __restrict__ targets,
                               const int* __restrict__ queries, int nq, float* __restrict__ target_lp,
                               float* __restrict__ query_lp, int* __restrict__ argmax_out) {
    pdl_wait();
    pdl_trigger();
    __shared__ float red[32];
    __shared__ float sv[32];
    __shared__ int si[32];
    const int r = blockIdx.x;
    const float* row = logits + static_cast<std::size_t>(r) * ld;
    float mx = -INFINITY;
    int bi = 0x7fffffff;
    for(int i = threadIdx.x; i < kVocabValid; i += blockDim.x) {
        const float v = row[i];
        if(v > mx) { mx = v; bi = i; }
    }
#pragma unroll
    for(int o = 16; o; o >>= 1) {
        const float ov = __shfl_xor_sync(0xffffffffu, mx, o);
        const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
        if(ov > mx || (ov == mx && oi < bi)) { mx = ov; bi = oi; }
    }
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    if(lane == 0) { sv[warp] = mx; si[warp] = bi; }
    __syncthreads();
    float best = sv[0];
    int besti = si[0];
    for(int w = 1; w < static_cast<int>(blockDim.x >> 5); ++w)
        if(sv[w] > best || (sv[w] == best && si[w] < besti)) { best = sv[w]; besti = si[w]; }
    float s = 0.f;
    for(int i = threadIdx.x; i < kVocabValid; i += blockDim.x) s += expf(row[i] - best);
    s = block_sum(s, red);
    const float lse = best + logf(s);
    if(threadIdx.x == 0) {
        target_lp[r] = row[targets[r]] - lse;
        if(argmax_out) argmax_out[r] = besti;
    }
    for(int j = threadIdx.x; j < nq; j += blockDim.x) {
        const int id = queries[static_cast<std::size_t>(r) * nq + j];
        query_lp[static_cast<std::size_t>(r) * nq + j] = (id >= 0 && id < kVocabValid) ? row[id] - lse : -INFINITY;
    }
}

// ------------------------------------------------------------------------------------------------ prompt path
constexpr int kRowThreads = kHidden / 8;  // 672

__global__ void __launch_bounds__(kRowThreads) rmsnorm_quant_fp8_kernel(const float* __restrict__ x, const float* __restrict__ w,
                                                                        std::uint8_t* __restrict__ codes, float* __restrict__ scale) {
    pdl_wait();
    pdl_trigger();
    __shared__ float red[32];
    const int r = blockIdx.x, c0 = threadIdx.x * 8;
    const float* xr = x + static_cast<std::size_t>(r) * kHidden;
    float v[8], ss = 0.f;
#pragma unroll
    for(int i = 0; i < 8; ++i) { v[i] = xr[c0 + i]; ss += v[i] * v[i]; }
    ss = block_sum(ss, red);
    const float rs = rsqrtf(ss / kHidden + kEps);
    float am = 0.f;
#pragma unroll
    for(int i = 0; i < 8; ++i) { v[i] = v[i] * rs * w[c0 + i]; am = fmaxf(am, fabsf(v[i])); }
    am = block_max(am, red);
    const float sc = am > 0.f ? am / 448.f : 1.f;
#pragma unroll
    for(int i = 0; i < 8; ++i) codes[static_cast<std::size_t>(r) * kHidden + c0 + i] = to_e4m3(v[i] / sc);
    if(threadIdx.x == 0) scale[r] = sc;
}

__global__ void __launch_bounds__(kRowThreads) sandwich_rows_kernel(Proj p, int T, float* __restrict__ hidden,
                                                                    const float* __restrict__ w_post, float scalar,
                                                                    const float* __restrict__ w_next, RowOut out) {
    pdl_wait();
    pdl_trigger();
    __shared__ float red[32];
    const int r = blockIdx.x, c0 = threadIdx.x * 8;
    const bool live = r < T;
    float v[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    if(live) {
        float pv[8];
        p.at8(r, c0, pv);
        float ss = 0.f;
#pragma unroll
        for(int i = 0; i < 8; ++i) ss += pv[i] * pv[i];
        ss = block_sum(ss, red);
        const float r1 = rsqrtf(ss / kHidden + kEps);
        float* hr = hidden + static_cast<std::size_t>(r) * kHidden + c0;
        float4* h4 = reinterpret_cast<float4*>(hr);
        const float4 a = h4[0], b = h4[1];
        const float hv[8] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
        float s2 = 0.f;
#pragma unroll
        for(int i = 0; i < 8; ++i) {
            v[i] = (hv[i] + pv[i] * r1 * w_post[c0 + i]) * scalar;
            s2 += v[i] * v[i];
        }
        h4[0] = make_float4(v[0], v[1], v[2], v[3]);
        h4[1] = make_float4(v[4], v[5], v[6], v[7]);
        s2 = block_sum(s2, red);
        const float r2 = rsqrtf(s2 / kHidden + kEps);
#pragma unroll
        for(int i = 0; i < 8; ++i) v[i] = v[i] * r2 * w_next[c0 + i];
    }
    if(out.bf) {
        if(live)
#pragma unroll
            for(int i = 0; i < 8; ++i) out.bf[static_cast<std::size_t>(r) * kHidden + c0 + i] = __float2bfloat16_rn(v[i]);
        return;
    }
    if(out.codes) {
        float am = 0.f;
#pragma unroll
        for(int i = 0; i < 8; ++i) am = fmaxf(am, fabsf(v[i]));
        if(live) am = block_max(am, red);
        if(!live) return;
        const float sc = am > 0.f ? am / 448.f : 1.f;
#pragma unroll
        for(int i = 0; i < 8; ++i) out.codes[static_cast<std::size_t>(r) * kHidden + c0 + i] = to_e4m3(v[i] / sc);
        if(threadIdx.x == 0) out.scale[r] = sc;
        return;
    }
    // NVFP4: thread pairs form a 16-element block; the global scale is static or the row's own.
    float bm = 0.f;
#pragma unroll
    for(int i = 0; i < 8; ++i) bm = fmaxf(bm, fabsf(v[i]));
    float gs = out.gs;
    if(out.rscale) {
        const float am = live ? block_max(bm, red) : 0.f;
        gs = am > 0.f ? 448.f * 6.f / am : 1.f;
        if(threadIdx.x == 0) out.rscale[r] = 1.f / gs;
    }
    bm = fmaxf(bm, __shfl_xor_sync(0xffffffffu, bm, 1));
    float blk[16];
    const bool even = (threadIdx.x & 1) == 0;
#pragma unroll
    for(int i = 0; i < 8; ++i) {
        const float other = __shfl_xor_sync(0xffffffffu, v[i], 1);
        blk[i] = even ? v[i] : other;
        blk[8 + i] = even ? other : v[i];
    }
    if(even) {
        uint2 codes;
        std::uint8_t scode;
        fp4_block(blk, bm, gs, codes, scode);
        const int b = c0 / 16;
        *reinterpret_cast<uint2*>(out.packed + static_cast<std::size_t>(r) * (kHidden / 2) + b * 8) = live ? codes : make_uint2(0, 0);
        out.sf[sf_offset(r, b, sf_blocks_padded(kHidden))] = live ? scode : 0;
    }
}

// Static scale: one thread per 16-element block of the GeGLU output.
__global__ void geglu_fp4_kernel(Proj p, int T, float gs, std::uint8_t* __restrict__ packed, std::uint8_t* __restrict__ sf) {
    pdl_wait();
    pdl_trigger();
    const int r = blockIdx.y;
    const int b = blockIdx.x * blockDim.x + threadIdx.x;
    if(b >= kInter / 16) return;
    float v[16], bm = 0.f;
    if(r < T) {
#pragma unroll
        for(int half = 0; half < 2; ++half) {
            const int blk8 = 2 * b + half;  // output elements blk8 * 8 .. + 7: gate rows blk8 * 16 .., up rows + 8
            float gt[8], up[8];
            p.at8(r, blk8 * 16, gt);
            p.at8(r, blk8 * 16 + 8, up);
#pragma unroll
            for(int j = 0; j < 8; ++j) v[half * 8 + j] = gelu_tanh(gt[j]) * up[j];
        }
    } else {
#pragma unroll
        for(int i = 0; i < 16; ++i) v[i] = 0.f;
    }
#pragma unroll
    for(int i = 0; i < 16; ++i) bm = fmaxf(bm, fabsf(v[i]));
    uint2 codes;
    std::uint8_t scode;
    fp4_block(v, bm, gs, codes, scode);
    *reinterpret_cast<uint2*>(packed + static_cast<std::size_t>(r) * (kInter / 2) + b * 8) = r < T ? codes : make_uint2(0, 0);
    sf[sf_offset(r, b, sf_blocks_padded(kInter))] = r < T ? scode : 0;
}

// Per-row scale: one block per row, thread t owns the 16-element blocks t and t + 672.
constexpr int kGegluThreads = kInter / 32;  // 672
__global__ void __launch_bounds__(kGegluThreads) geglu_fp4_row_kernel(Proj p, int T, float* __restrict__ rscale,
                                                                     std::uint8_t* __restrict__ packed,
                                                                     std::uint8_t* __restrict__ sf) {
    pdl_wait();
    pdl_trigger();
    __shared__ float red[32];
    const int r = blockIdx.x;
    const bool live = r < T;
    float v[2][16], bm[2] = {0.f, 0.f};
#pragma unroll
    for(int k = 0; k < 2; ++k) {
        const int b = threadIdx.x + k * kGegluThreads;
        if(live) {
#pragma unroll
            for(int half = 0; half < 2; ++half) {
                const int blk8 = 2 * b + half;
                float gt[8], up[8];
                p.at8(r, blk8 * 16, gt);
                p.at8(r, blk8 * 16 + 8, up);
#pragma unroll
                for(int j = 0; j < 8; ++j) v[k][half * 8 + j] = gelu_tanh(gt[j]) * up[j];
            }
        } else {
#pragma unroll
            for(int i = 0; i < 16; ++i) v[k][i] = 0.f;
        }
#pragma unroll
        for(int i = 0; i < 16; ++i) bm[k] = fmaxf(bm[k], fabsf(v[k][i]));
    }
    const float am = block_max(fmaxf(bm[0], bm[1]), red);
    const float gs = am > 0.f ? 448.f * 6.f / am : 1.f;
    if(threadIdx.x == 0) rscale[r] = 1.f / gs;
#pragma unroll
    for(int k = 0; k < 2; ++k) {
        const int b = threadIdx.x + k * kGegluThreads;
        uint2 codes;
        std::uint8_t scode;
        fp4_block(v[k], bm[k], gs, codes, scode);
        *reinterpret_cast<uint2*>(packed + static_cast<std::size_t>(r) * (kInter / 2) + b * 8) = live ? codes : make_uint2(0, 0);
        sf[sf_offset(r, b, sf_blocks_padded(kInter))] = live ? scode : 0;
    }
}

__global__ void quant_fp8_rows_kernel(const float* __restrict__ in, int K, std::uint8_t* __restrict__ codes,
                                      float* __restrict__ scale) {
    pdl_wait();
    pdl_trigger();
    __shared__ float red[32];
    const int r = blockIdx.x;
    const float* row = in + static_cast<std::size_t>(r) * K;
    float am = 0.f;
    for(int i = threadIdx.x; i < K; i += blockDim.x) am = fmaxf(am, fabsf(row[i]));
    am = block_max(am, red);
    const float sc = am > 0.f ? am / 448.f : 1.f;
    for(int i = threadIdx.x; i < K; i += blockDim.x) codes[static_cast<std::size_t>(r) * K + i] = to_e4m3(row[i] / sc);
    if(threadIdx.x == 0) scale[r] = sc;
}

__global__ void to_bf16_kernel(const float4* __restrict__ in, int n4, uint2* __restrict__ out) {
    pdl_wait();
    pdl_trigger();
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= n4) return;
    const float4 v = in[i];
    const __nv_bfloat162 a = __floats2bfloat162_rn(v.x, v.y), b = __floats2bfloat162_rn(v.z, v.w);
    out[i] = make_uint2(*reinterpret_cast<const std::uint32_t*>(&a), *reinterpret_cast<const std::uint32_t*>(&b));
}

}  // namespace kops
using namespace kops;

void to_bf16(const float* in, int n, __nv_bfloat16* out, cudaStream_t s) {
    const int n4 = n / 4;
    launch_k(to_bf16_kernel, (n4 + 255) / 256, 256, 0, s, reinterpret_cast<const float4*>(in), n4, reinterpret_cast<uint2*>(out));
    KLAUNCH();
}

void embed(const int* tok, int N, const __nv_bfloat16* table, int dim, float scale, float* out, int ldo, cudaStream_t s) {
    launch_k(embed_kernel, N, 256, 0, s, tok, table, dim, scale, out, ldo);
    KLAUNCH();
}
void rmsnorm_bf16(const float* x, int ldx, const float* w, int N, int dim, __nv_bfloat16* out, int ldo, cudaStream_t s) {
    launch_k(rmsnorm_kernel, N, 1024, 0, s, x, ldx, w, dim, out, ldo);
    KLAUNCH();
}
void sandwich(const float* v, int ldv, float* y, int ldy, const float* w_post, const float* scalar, const float* w_next,
              __nv_bfloat16* h, int ldh, int N, int dim, cudaStream_t s) {
    const int threads = dim >= 4096 ? 768 : 256;
    if(dim > threads * kSandPer) throw std::invalid_argument("sandwich: dim too large");
    launch_k(sandwich_kernel, N, threads, 0, s, v, ldv, y, ldy, w_post, scalar, w_next, h, ldh, dim);
    KLAUNCH();
}
void slide_prep(Proj qkv, int N, Pos pos, const float* q_norm, const float* k_norm, const float* inv, __nv_bfloat16* q_out,
                int ldq, const SlideCache& c, int which, cudaStream_t s) {
    const int first = which == 2 ? kHeads : 0;
    const int slots = which == 0 ? kHeads + 2 * kSKv : which == 1 ? kHeads : 2 * kSKv;
    launch_k(slide_prep_kernel, dim3(N, slots), kSHd, 0, s, qkv, pos, q_norm, k_norm, inv, q_out, ldq, c, first);
    KLAUNCH();
}
void global_prep(Proj qkv, int N, Pos pos, const float* q_norm, const float* inv, __nv_bfloat16* q_out, int ldq,
                 const GlobalCache& c, int which, cudaStream_t s) {
    const int first = which == 2 ? kHeads : 0;
    const int slots = which == 0 ? kHeads + kGKv : which == 1 ? kHeads : kGKv;
    launch_k(global_prep_kernel, dim3(N, slots), kGHd, 0, s, qkv, pos, q_norm, inv, q_out, ldq, c, first);
    KLAUNCH();
}
void argmax(const float* logits, int ld, int vocab, int N, int* out, cudaStream_t s) {
    launch_k(argmax_kernel, N, 1024, 0, s, logits, ld, vocab, out);
    KLAUNCH();
}
void argmax_wide(const float* logits, int ld, int vocab, int N, int* out, float2* scratch, cudaStream_t s) {
    launch_k(argmax_part_kernel, dim3(kArgParts, N), 256, 0, s, logits, ld, vocab, scratch);
    launch_k(argmax_merge_kernel, N, 32, 0, s, static_cast<const float2*>(scratch), out);
    KLAUNCH();
}
void logprob_rows(const float* logits, int ld, int rows, const int* targets, const int* queries, int nq,
                  float* target_lp, float* query_lp, int* argmax_out, cudaStream_t s) {
    launch_k(logprob_kernel, rows, 1024, 0, s, logits, ld, targets, queries, nq, target_lp, query_lp, argmax_out);
    KLAUNCH();
}
void rmsnorm_quant_fp8(const float* x, const float* w, int T, std::uint8_t* codes, float* scale, cudaStream_t s) {
    launch_k(rmsnorm_quant_fp8_kernel, T, kRowThreads, 0, s, x, w, codes, scale);
    KLAUNCH();
}
void sandwich_rows(Proj v, int T, float* hidden, const float* w_post, float scalar, const float* w_next, const RowOut& out,
                   cudaStream_t s) {
    const int rows = out.packed ? out.T_pad : T;
    launch_k(sandwich_rows_kernel, rows, kRowThreads, 0, s, v, T, hidden, w_post, scalar, w_next, out);
    KLAUNCH();
}
void geglu_fp4(Proj gu, int T, int T_pad, float gs, float* rscale, std::uint8_t* packed, std::uint8_t* sf, cudaStream_t s) {
    if(rscale) launch_k(geglu_fp4_row_kernel, T_pad, kGegluThreads, 0, s, gu, T, rscale, packed, sf);
    else launch_k(geglu_fp4_kernel, dim3((kInter / 16 + 127) / 128, T_pad), 128, 0, s, gu, T, gs, packed, sf);
    KLAUNCH();
}
void quant_fp8_rows(const float* in, int T, int K, std::uint8_t* codes, float* scale, cudaStream_t s) {
    launch_k(quant_fp8_rows_kernel, T, 1024, 0, s, in, K, codes, scale);
    KLAUNCH();
}

}  // namespace g4
