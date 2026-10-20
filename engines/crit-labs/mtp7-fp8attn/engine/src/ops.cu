#include "ops.cuh"

#include <cuda_fp8.h>

namespace kestrel {
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
    float t = -INFINITY;
    for(int i = 0; i < nw; ++i) t = fmaxf(t, red[i]);
    return t;
}
__device__ __forceinline__ float silu(float v) { return v / (1.f + expf(-v)); }
__device__ __forceinline__ float sigmoidf_(float v) { return 1.f / (1.f + expf(-v)); }

// ------------------------------------------------------------------------------------------------ embed / norms
__global__ void embed_kernel(const int* __restrict__ tok, const __nv_bfloat16* __restrict__ table, float* __restrict__ out) {
    pdl_wait();
    pdl_trigger();
    const int n = blockIdx.x;
    const std::size_t row = static_cast<std::size_t>(tok[n]) * kHidden;
    for(int i = threadIdx.x; i < kHidden / 8; i += blockDim.x) {
        const uint4 u = *reinterpret_cast<const uint4*>(table + row + i * 8);
        const __nv_bfloat162* b = reinterpret_cast<const __nv_bfloat162*>(&u);
        float4 lo, hi;
        float2 f0 = __bfloat1622float2(b[0]), f1 = __bfloat1622float2(b[1]), f2 = __bfloat1622float2(b[2]),
               f3 = __bfloat1622float2(b[3]);
        lo = make_float4(f0.x, f0.y, f1.x, f1.y);
        hi = make_float4(f2.x, f2.y, f3.x, f3.y);
        float4* o = reinterpret_cast<float4*>(out + static_cast<std::size_t>(n) * kHidden + i * 8);
        o[0] = lo;
        o[1] = hi;
    }
}

template <typename In>
__global__ void rmsnorm_kernel(const In* __restrict__ x, int ldx, const float* __restrict__ w, int dim, __half* __restrict__ out,
                               int ldo) {
    pdl_wait();
    pdl_trigger();
    __shared__ float red[32];
    const int n = blockIdx.x;
    const In* xr = x + static_cast<std::size_t>(n) * ldx;
    float ss = 0.f;
    for(int i = threadIdx.x; i < dim; i += blockDim.x) {
        const float v = static_cast<float>(xr[i]);
        ss += v * v;
    }
    ss = block_sum(ss, red);
    const float r = rsqrtf(ss / dim + kEps);
    for(int i = threadIdx.x; i < dim; i += blockDim.x)
        out[static_cast<std::size_t>(n) * ldo + i] = __float2half_rn(static_cast<float>(xr[i]) * r * (1.f + w[i]));
}

// ------------------------------------------------------------------------------------------------ attention prep
// Block (row n, head slot): slots 0..23 queries, 24..27 keys, 28..31 values; one thread per dimension.
__global__ void __launch_bounds__(kHeadDim) attn_prep_kernel(Proj qkv, Pos pos,
                                                             const float* __restrict__ qn, const float* __restrict__ kn,
                                                             const float* __restrict__ rope_inv, __nv_bfloat16* __restrict__ q_out,
                                                             crit::fp8_kv::Cache cache, std::uint8_t* __restrict__ q8,
                                                             float* __restrict__ q8s) {
    pdl_wait();
    pdl_trigger();
    __shared__ float red[8];
    __shared__ float rot[kRotary];
    const int n = blockIdx.x, slot = blockIdx.y + (gridDim.y == 2 * kKvHeads ? kHeads : 0), d = threadIdx.x;
    const int p = (pos.dev ? *pos.dev : pos.host) + n;
    float x;
    if(slot < kHeads) x = qkv.at(n, slot * 2 * kHeadDim + d);
    else if(slot < kHeads + kKvHeads) x = qkv.at(n, kQRows + (slot - kHeads) * kHeadDim + d);
    else x = qkv.at(n, kQRows + kKvRows + (slot - kHeads - kKvHeads) * kHeadDim + d);
    if(slot < kHeads + kKvHeads) {
        const float ss = block_sum(x * x, red);
        const float* w = slot < kHeads ? qn : kn;
        x = x * rsqrtf(ss / kHeadDim + kEps) * (1.f + w[d]);
        // Partial RoPE on the first 64 dims: rotate_half within them.
        if(d < kRotary) rot[d] = x;
        __syncthreads();
        if(d < kRotary) {
            const int i = d & 31;
            const float ang = static_cast<float>(p) * rope_inv[i];
            const float c = cosf(ang), s = sinf(ang);
            x = d < 32 ? x * c - rot[d + 32] * s : x * c + rot[d - 32] * s;
        }
    }
    if(slot < kHeads) {
        const __nv_bfloat16 xb = __float2bfloat16_rn(x);
        q_out[static_cast<std::size_t>(n) * kAttnOut + slot * kHeadDim + d] = xb;
        if(q8) {
            // FP8 codes of the BF16 query (what the attention kernels would quantize), one scale per row and head.
            const float xv = __bfloat162float(xb);
            const float am = block_max(fabsf(xv), red);
            const float sc = am > 0.f ? am / 448.f : 1.f;
            q8[(static_cast<std::size_t>(n) * kHeads + slot) * kHeadDim + d] =
                static_cast<std::uint8_t>(__nv_cvt_float_to_fp8(xv / sc, __NV_SATFINITE, __NV_E4M3));
            if(d == 0) q8s[static_cast<std::size_t>(n) * kHeads + slot] = sc;
        }
        return;
    }
    const bool is_k = slot < kHeads + kKvHeads;
    const int g = is_k ? slot - kHeads : slot - kHeads - kKvHeads;
    const float amax = block_max(fabsf(x), red);
    const float sc = amax > 0.f ? amax / 448.f : 1.f;
    const std::size_t pp = static_cast<std::size_t>(p);
    std::uint8_t* codes = is_k ? cache.k : cache.v;
    float* scales = is_k ? cache.k_scale : cache.v_scale;
    codes[(pp * kKvHeads + g) * kHeadDim + d] = static_cast<std::uint8_t>(__nv_cvt_float_to_fp8(x / sc, __NV_SATFINITE, __NV_E4M3));
    if(d == 0) scales[pp * kKvHeads + g] = sc;
}

__global__ void attn_gate_kernel(const float* __restrict__ attn, Proj qkv, int N,
                                 __half* __restrict__ out) {
    pdl_wait();
    pdl_trigger();
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= N * kAttnOut) return;
    const int n = idx / kAttnOut, c = idx % kAttnOut, h = c / kHeadDim, d = c % kHeadDim;
    const float gate = qkv.at(n, h * 2 * kHeadDim + kHeadDim + d);
    out[idx] = __float2half_rn(attn[idx] * sigmoidf_(gate));
}

// ------------------------------------------------------------------------------------------------ gated DeltaNet
constexpr int kQkvz = kLinQkv + kLinZ;  // 16384
constexpr int kConvThreads = 640;       // q (128) | k (128) | v of three value heads (384)

// Block per key head: causal conv + SiLU with deferred window commit, q/k L2 norm, and a/b gates of its value heads.
__global__ void __launch_bounds__(kConvThreads) gdn_conv_kernel(GdnLayer L, GdnState st, GdnScratch sc,
                                                                const float* __restrict__ qkvz, const __half* __restrict__ xn,
                                                                int N, const int* __restrict__ pending) {
    pdl_wait();
    pdl_trigger();
    __shared__ float red[32];
    __shared__ float ab[kMaxCols][6];
    const int kh = blockIdx.x, tid = threadIdx.x;
    int ch;
    if(tid < 128) ch = kh * 128 + tid;
    else if(tid < 256) ch = 2048 + kh * 128 + (tid - 128);
    else ch = 4096 + kh * 384 + (tid - 256);
    const int p = pending ? min(*pending, kMaxCols) : 0;
    float w0 = st.conv[ch], w1 = st.conv[kLinQkv + ch], w2 = st.conv[2 * kLinQkv + ch];
    for(int j = 0; j < p; ++j) {
        const float x = st.conv_stash[static_cast<std::size_t>(j) * kLinQkv + ch];
        w0 = w1; w1 = w2; w2 = x;
    }
    st.conv[ch] = w0; st.conv[kLinQkv + ch] = w1; st.conv[2 * kLinQkv + ch] = w2;
    const float4 cw = *reinterpret_cast<const float4*>(L.conv_w + ch * 4);
    // a / b gates: 6 dot products (3 value heads x {a, b}) per column, one warp each per (gate, column).
    const int lane = tid & 31, warp = tid >> 5;
    for(int job = warp; job < 6 * N; job += kConvThreads / 32) {
        const int col = job / 6, which = job % 6, h = kh * 3 + which / 2;
        const __nv_bfloat16* wr = ((which & 1) ? L.in_b : L.in_a) + static_cast<std::size_t>(h) * kHidden;
        const __half* xr = xn + static_cast<std::size_t>(col) * kHidden;
        float acc = 0.f;
        for(int i = lane; i < kHidden; i += 32) acc += __bfloat162float(wr[i]) * __half2float(xr[i]);
        acc = warp_sum(acc);
        if(lane == 0) ab[col][which] = acc;
    }
    for(int n = 0; n < N; ++n) {
        const float x = qkvz[static_cast<std::size_t>(n) * kQkvz + ch];
        st.conv_stash[static_cast<std::size_t>(n) * kLinQkv + ch] = x;
        float y = cw.x * w0;
        y = fmaf(cw.y, w1, y);
        y = fmaf(cw.z, w2, y);
        y = fmaf(cw.w, x, y);
        w0 = w1; w1 = w2; w2 = x;
        y = silu(y);
        // L2 norms of q (threads 0..127) and k (128..255): two block-wide partial sums.
        const float sq = tid < 128 ? y * y : 0.f, sk = (tid >= 128 && tid < 256) ? y * y : 0.f;
        const float tq = block_sum(sq, red);
        const float tk = block_sum(sk, red);
        if(tid < 128) {
            sc.qk[((static_cast<std::size_t>(n) * kLinKHeads + kh) * 2 + 0) * kLinDim + tid] =
                y * rsqrtf(tq + kEps) * rsqrtf(static_cast<float>(kLinDim));
        } else if(tid < 256) {
            sc.qk[((static_cast<std::size_t>(n) * kLinKHeads + kh) * 2 + 1) * kLinDim + (tid - 128)] = y * rsqrtf(tk + kEps);
        } else {
            const int j = (tid - 256) / 128, i = (tid - 256) % 128;
            sc.v[(static_cast<std::size_t>(n) * kLinVHeads + kh * 3 + j) * kLinDim + i] = y;
        }
    }
    __syncthreads();
    if(tid < 3 * N) {
        const int col = tid / 3, j = tid % 3, h = kh * 3 + j;
        const float a = ab[col][2 * j], b = ab[col][2 * j + 1];
        const float t = a + L.dt_bias[h];
        const float sp = t > 20.f ? t : log1pf(expf(t));
        const float g = L.neg_a[h] * sp;
        sc.gb[(static_cast<std::size_t>(col) * kLinVHeads + h) * 2 + 0] = expf(g);
        sc.gb[(static_cast<std::size_t>(col) * kLinVHeads + h) * 2 + 1] = sigmoidf_(b);
    }
}

constexpr int kRecThreads = 512;  // value column v = tid / 4, key quarter tid % 4 (32 keys each)

__global__ void __launch_bounds__(kRecThreads) gdn_recur_kernel(GdnLayer L, GdnState st, GdnScratch sc,
                                                                const float* __restrict__ qkvz, int N,
                                                                const int* __restrict__ pending, __half* __restrict__ out) {
    pdl_wait();
    pdl_trigger();
    __shared__ float o_s[kLinDim];
    __shared__ float red[32];
    const int h = blockIdx.x, kh = h / 3, tid = threadIdx.x, v = tid >> 2, kg = tid & 3;
    float S[32];
    float4* sp = reinterpret_cast<float4*>(st.S + (static_cast<std::size_t>(h) * kRecThreads + tid) * 32);
#pragma unroll
    for(int i = 0; i < 8; ++i) {
        const float4 t = sp[i];
        S[4 * i] = t.x; S[4 * i + 1] = t.y; S[4 * i + 2] = t.z; S[4 * i + 3] = t.w;
    }
    const int p = pending ? min(*pending, kMaxCols) : 0;
    for(int j = 0; j < p; ++j) {
        const float decay = st.stash_decay[j * kLinVHeads + h];
        const float delta = st.stash_delta[(static_cast<std::size_t>(j) * kLinVHeads + h) * kLinDim + v];
        const float4* kp = reinterpret_cast<const float4*>(st.stash_k + (static_cast<std::size_t>(j) * kLinVHeads + h) * kLinDim + kg * 32);
#pragma unroll
        for(int i = 0; i < 8; ++i) {
            const float4 k4 = kp[i];
            S[4 * i] = fmaf(k4.x, delta, S[4 * i] * decay);
            S[4 * i + 1] = fmaf(k4.y, delta, S[4 * i + 1] * decay);
            S[4 * i + 2] = fmaf(k4.z, delta, S[4 * i + 2] * decay);
            S[4 * i + 3] = fmaf(k4.w, delta, S[4 * i + 3] * decay);
        }
    }
    if(p > 0) {
#pragma unroll
        for(int i = 0; i < 8; ++i) sp[i] = make_float4(S[4 * i], S[4 * i + 1], S[4 * i + 2], S[4 * i + 3]);
    }
    __syncthreads();  // every pending stash entry is read before this round overwrites the stash
    for(int n = 0; n < N; ++n) {
        const float decay = sc.gb[(static_cast<std::size_t>(n) * kLinVHeads + h) * 2 + 0];
        const float beta = sc.gb[(static_cast<std::size_t>(n) * kLinVHeads + h) * 2 + 1];
        const float* qp = sc.qk + ((static_cast<std::size_t>(n) * kLinKHeads + kh) * 2 + 0) * kLinDim + kg * 32;
        const float* kp = sc.qk + ((static_cast<std::size_t>(n) * kLinKHeads + kh) * 2 + 1) * kLinDim + kg * 32;
        const float vv = sc.v[(static_cast<std::size_t>(n) * kLinVHeads + h) * kLinDim + v];
        float kv = 0.f;
        float kk[32];
#pragma unroll
        for(int i = 0; i < 8; ++i) {
            const float4 k4 = reinterpret_cast<const float4*>(kp)[i];
            kk[4 * i] = k4.x; kk[4 * i + 1] = k4.y; kk[4 * i + 2] = k4.z; kk[4 * i + 3] = k4.w;
        }
#pragma unroll
        for(int i = 0; i < 32; ++i) {
            S[i] *= decay;
            kv = fmaf(S[i], kk[i], kv);
        }
        kv += __shfl_xor_sync(0xffffffffu, kv, 1);
        kv += __shfl_xor_sync(0xffffffffu, kv, 2);
        const float delta = (vv - kv) * beta;
        float o = 0.f;
#pragma unroll
        for(int i = 0; i < 8; ++i) {
            const float4 q4 = reinterpret_cast<const float4*>(qp)[i];
            S[4 * i] = fmaf(kk[4 * i], delta, S[4 * i]);
            S[4 * i + 1] = fmaf(kk[4 * i + 1], delta, S[4 * i + 1]);
            S[4 * i + 2] = fmaf(kk[4 * i + 2], delta, S[4 * i + 2]);
            S[4 * i + 3] = fmaf(kk[4 * i + 3], delta, S[4 * i + 3]);
            o = fmaf(S[4 * i], q4.x, o);
            o = fmaf(S[4 * i + 1], q4.y, o);
            o = fmaf(S[4 * i + 2], q4.z, o);
            o = fmaf(S[4 * i + 3], q4.w, o);
        }
        o += __shfl_xor_sync(0xffffffffu, o, 1);
        o += __shfl_xor_sync(0xffffffffu, o, 2);
        if(kg == 0) {
            st.stash_delta[(static_cast<std::size_t>(n) * kLinVHeads + h) * kLinDim + v] = delta;
            o_s[v] = o;
        }
        if(tid < kLinDim) {
            st.stash_k[(static_cast<std::size_t>(n) * kLinVHeads + h) * kLinDim + tid] =
                sc.qk[((static_cast<std::size_t>(n) * kLinKHeads + kh) * 2 + 1) * kLinDim + tid];
        }
        if(tid == 0) st.stash_decay[n * kLinVHeads + h] = decay;
        __syncthreads();
        // Gated RMSNorm over the head's 128 outputs, times SiLU(z).
        const float ov = tid < kLinDim ? o_s[tid] : 0.f;
        const float ss = block_sum(ov * ov, red);
        if(tid < kLinDim) {
            const float z = qkvz[static_cast<std::size_t>(n) * kQkvz + kLinQkv + h * kLinDim + tid];
            const float y = L.norm_w[tid] * (ov * rsqrtf(ss / kLinDim + kEps)) * silu(z);
            out[static_cast<std::size_t>(n) * kLinZ + h * kLinDim + tid] = __float2half_rn(y);
        }
        __syncthreads();
    }
}

// One block per value head h (512 threads): the conv of its q/k (key head h / 3) and v channels with the window of
// the previous pass and the pending columns folded in, q/k L2 norms, a/b gates, the recurrence with the deferred
// commit and the gated RMSNorm. Window and conv stash are double-buffered by pass parity: blocks read buffer
// par ^ 1 and the owner of a channel (q/k: the key head's first value head) writes buffer par.
__global__ void __launch_bounds__(kRecThreads) gdn_fused_kernel(GdnLayer L, GdnState st, const float* __restrict__ qkvz,
                                                                const __half* __restrict__ xn, int N,
                                                                const int* __restrict__ pending, const int* __restrict__ pass,
                                                                __half* __restrict__ out) {
    pdl_wait();
    pdl_trigger();
    __shared__ float xq[kMaxCols][kLinDim], xk[kMaxCols][kLinDim], xv[kMaxCols][kLinDim];
    __shared__ float red[kMaxCols][8];
    __shared__ float ab[kMaxCols][2];
    __shared__ float gb[kMaxCols][2];
    __shared__ float o_s[kLinDim];
    __shared__ float red2[32];
    const int h = blockIdx.x, kh = h / 3, tid = threadIdx.x, v = tid >> 2, kg = tid & 3;
    const int lane = tid & 31, warp = tid >> 5;
    const int par = (*pass) & 1;
    const int p = pending ? min(*pending, kMaxCols) : 0;
    float S[32];
    float4* sp = reinterpret_cast<float4*>(st.S + (static_cast<std::size_t>(h) * kRecThreads + tid) * 32);
#pragma unroll
    for(int i = 0; i < 8; ++i) {
        const float4 t = sp[i];
        S[4 * i] = t.x; S[4 * i + 1] = t.y; S[4 * i + 2] = t.z; S[4 * i + 3] = t.w;
    }
    // Convolution: threads 0..127 q, 128..255 k, 256..383 v.
    if(tid < 3 * kLinDim) {
        const int role = tid / kLinDim, i = tid % kLinDim;
        const int ch = role == 0 ? kh * kLinDim + i : role == 1 ? 2048 + kh * kLinDim + i : 4096 + h * kLinDim + i;
        const bool owner = role == 2 || h % 3 == 0;
        const float* win = st.conv + static_cast<std::size_t>(par ^ 1) * 3 * kLinQkv;
        const float* old = st.conv_stash + static_cast<std::size_t>(par ^ 1) * kMaxCols * kLinQkv;
        float w0 = win[ch], w1 = win[kLinQkv + ch], w2 = win[2 * kLinQkv + ch];
        for(int j = 0; j < p; ++j) {
            const float x = old[static_cast<std::size_t>(j) * kLinQkv + ch];
            w0 = w1; w1 = w2; w2 = x;
        }
        float* nwin = st.conv + static_cast<std::size_t>(par) * 3 * kLinQkv;
        float* nst = st.conv_stash + static_cast<std::size_t>(par) * kMaxCols * kLinQkv;
        if(owner) { nwin[ch] = w0; nwin[kLinQkv + ch] = w1; nwin[2 * kLinQkv + ch] = w2; }
        const float4 cw = *reinterpret_cast<const float4*>(L.conv_w + ch * 4);
        for(int n = 0; n < N; ++n) {
            const float x = qkvz[static_cast<std::size_t>(n) * kQkvz + ch];
            if(owner) nst[static_cast<std::size_t>(n) * kLinQkv + ch] = x;
            float y = cw.x * w0;
            y = fmaf(cw.y, w1, y);
            y = fmaf(cw.z, w2, y);
            y = fmaf(cw.w, x, y);
            w0 = w1; w1 = w2; w2 = x;
            y = silu(y);
            (role == 0 ? xq : role == 1 ? xk : xv)[n][i] = y;
            if(role < 2) {
                const float sq = warp_sum(y * y);
                if(lane == 0) red[n][warp] = sq;  // warps 0..3: q, 4..7: k
            }
        }
    } else {
        // a / b gates of this value head: warps 12..15, one (gate, column) job each.
        for(int job = warp - 12; job < 2 * N; job += 4) {
            const int col = job >> 1;
            const __nv_bfloat16* wr = ((job & 1) ? L.in_b : L.in_a) + static_cast<std::size_t>(h) * kHidden;
            const __half* xr = xn + static_cast<std::size_t>(col) * kHidden;
            float acc = 0.f;
            for(int i = lane * 8; i < kHidden; i += 256) {
                const uint4 wu = *reinterpret_cast<const uint4*>(wr + i), xu = *reinterpret_cast<const uint4*>(xr + i);
                const __nv_bfloat162* wp = reinterpret_cast<const __nv_bfloat162*>(&wu);
                const __half2* xp = reinterpret_cast<const __half2*>(&xu);
#pragma unroll
                for(int k = 0; k < 4; ++k) {
                    const float2 a = __bfloat1622float2(wp[k]), b = __half22float2(xp[k]);
                    acc = fmaf(a.x, b.x, acc);
                    acc = fmaf(a.y, b.y, acc);
                }
            }
            acc = warp_sum(acc);
            if(lane == 0) ab[col][job & 1] = acc;
        }
    }
    __syncthreads();
    if(tid < 2 * kLinDim) {
        const int role = tid / kLinDim, i = tid % kLinDim;
        for(int n = 0; n < N; ++n) {
            const float t = red[n][role * 4] + red[n][role * 4 + 1] + red[n][role * 4 + 2] + red[n][role * 4 + 3];
            if(role == 0) xq[n][i] = xq[n][i] * rsqrtf(t + kEps) * rsqrtf(static_cast<float>(kLinDim));
            else xk[n][i] = xk[n][i] * rsqrtf(t + kEps);
        }
    } else if(tid >= 256 && tid < 256 + N) {
        const int col = tid - 256;
        const float x = ab[col][0] + L.dt_bias[h];
        const float spl = x > 20.f ? x : log1pf(expf(x));
        gb[col][0] = expf(L.neg_a[h] * spl);
        gb[col][1] = sigmoidf_(ab[col][1]);
    }
    // Fold the previous pass's committed columns into the state (stash written by this block).
    for(int j = 0; j < p; ++j) {
        const float decay = st.stash_decay[j * kLinVHeads + h];
        const float delta = st.stash_delta[(static_cast<std::size_t>(j) * kLinVHeads + h) * kLinDim + v];
        const float4* kp = reinterpret_cast<const float4*>(st.stash_k + (static_cast<std::size_t>(j) * kLinVHeads + h) * kLinDim + kg * 32);
#pragma unroll
        for(int i = 0; i < 8; ++i) {
            const float4 k4 = kp[i];
            S[4 * i] = fmaf(k4.x, delta, S[4 * i] * decay);
            S[4 * i + 1] = fmaf(k4.y, delta, S[4 * i + 1] * decay);
            S[4 * i + 2] = fmaf(k4.z, delta, S[4 * i + 2] * decay);
            S[4 * i + 3] = fmaf(k4.w, delta, S[4 * i + 3] * decay);
        }
    }
    if(p > 0) {
#pragma unroll
        for(int i = 0; i < 8; ++i) sp[i] = make_float4(S[4 * i], S[4 * i + 1], S[4 * i + 2], S[4 * i + 3]);
    }
    __syncthreads();
    for(int n = 0; n < N; ++n) {
        const float decay = gb[n][0], beta = gb[n][1];
        const float vv = xv[n][v];
        float kk[32];
        float kv = 0.f;
#pragma unroll
        for(int i = 0; i < 32; ++i) {
            kk[i] = xk[n][kg * 32 + i];
            S[i] *= decay;
            kv = fmaf(S[i], kk[i], kv);
        }
        kv += __shfl_xor_sync(0xffffffffu, kv, 1);
        kv += __shfl_xor_sync(0xffffffffu, kv, 2);
        const float delta = (vv - kv) * beta;
        float o = 0.f;
#pragma unroll
        for(int i = 0; i < 32; ++i) {
            S[i] = fmaf(kk[i], delta, S[i]);
            o = fmaf(S[i], xq[n][kg * 32 + i], o);
        }
        o += __shfl_xor_sync(0xffffffffu, o, 1);
        o += __shfl_xor_sync(0xffffffffu, o, 2);
        if(kg == 0) {
            st.stash_delta[(static_cast<std::size_t>(n) * kLinVHeads + h) * kLinDim + v] = delta;
            o_s[v] = o;
        }
        if(tid < kLinDim) st.stash_k[(static_cast<std::size_t>(n) * kLinVHeads + h) * kLinDim + tid] = xk[n][tid];
        if(tid == 0) st.stash_decay[n * kLinVHeads + h] = decay;
        __syncthreads();
        const float ov = tid < kLinDim ? o_s[tid] : 0.f;
        const float ss = block_sum(ov * ov, red2);
        if(tid < kLinDim) {
            const float z = qkvz[static_cast<std::size_t>(n) * kQkvz + kLinQkv + h * kLinDim + tid];
            const float y = L.norm_w[tid] * (ov * rsqrtf(ss / kLinDim + kEps)) * silu(z);
            out[static_cast<std::size_t>(n) * kLinZ + h * kLinDim + tid] = __float2half_rn(y);
        }
        __syncthreads();
    }
}

// ------------------------------------------------------------------------------------------------ head
__global__ void argmax_kernel(const float* __restrict__ logits, int ld, int* __restrict__ out) {
    pdl_wait();
    pdl_trigger();
    __shared__ float sv[32];
    __shared__ int si[32];
    const int n = blockIdx.x;
    const float* row = logits + static_cast<std::size_t>(n) * ld;
    float best = -INFINITY;
    int bi = 0x7fffffff;
    for(int i = threadIdx.x; i < kVocabValid; i += blockDim.x) {
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

}  // namespace kops
using namespace kops;

void embed(const int* tok, int N, const __nv_bfloat16* table, float* out, cudaStream_t s) {
    launch_k(embed_kernel, N, 320, 0, s, tok, table, out);
    KLAUNCH();
}
void rmsnorm_f16(const float* x, int ldx, const float* w, int N, int dim, __half* out, int ldo, cudaStream_t s) {
    launch_k(rmsnorm_kernel<float>, N, 1024, 0, s, x, ldx, w, dim, out, ldo);
    KLAUNCH();
}
void rmsnorm_f16_from_f16(const __half* x, int ldx, const float* w, int N, int dim, __half* out, int ldo, cudaStream_t s) {
    launch_k(rmsnorm_kernel<__half>, N, 1024, 0, s, x, ldx, w, dim, out, ldo);
    KLAUNCH();
}
void attn_prep(Proj qkv, int N, Pos pos, const float* q_norm, const float* k_norm, const float* rope_inv,
               __nv_bfloat16* q_out, const crit::fp8_kv::Cache& cache, cudaStream_t s, bool kv_only, std::uint8_t* q8,
               float* q8_scale) {
    launch_k(attn_prep_kernel, dim3(N, kv_only ? 2 * kKvHeads : kHeads + 2 * kKvHeads), kHeadDim, 0, s, qkv, pos, q_norm, k_norm, rope_inv, q_out, cache, q8, q8_scale);
    KLAUNCH();
}
void attn_gate(const float* attn, Proj qkv, int N, __half* out, cudaStream_t s) {
    const int total = N * kAttnOut;
    launch_k(attn_gate_kernel, (total + 255) / 256, 256, 0, s, attn, qkv, N, out);
    KLAUNCH();
}
void gdn_step(const GdnLayer& L, GdnState st, GdnScratch sc, const float* qkvz, const __half* xn, int N,
              const int* pending, __half* out, cudaStream_t s) {
    launch_k(gdn_conv_kernel, kLinKHeads, kConvThreads, 0, s, L, st, sc, qkvz, xn, N, pending);
    KLAUNCH();
    launch_k(gdn_recur_kernel, kLinVHeads, kRecThreads, 0, s, L, st, sc, qkvz, N, pending, out);
    KLAUNCH();
}
void gdn_fused(const GdnLayer& L, GdnState st, const float* qkvz, const __half* xn, int N, const int* pending,
               const int* pass, __half* out, cudaStream_t s) {
    launch_k(gdn_fused_kernel, kLinVHeads, kRecThreads, 0, s, L, st, qkvz, xn, N, pending, pass, out);
}
void argmax(const float* logits, int ld, int N, int* out, cudaStream_t s) {
    launch_k(argmax_kernel, N, 1024, 0, s, logits, ld, out);
    KLAUNCH();
}
void logprob_rows(const float* logits, int ld, int rows, const int* targets, const int* queries, int nq,
                  float* target_lp, float* query_lp, int* argmax_out, cudaStream_t s) {
    launch_k(logprob_kernel, rows, 1024, 0, s, logits, ld, targets, queries, nq, target_lp, query_lp, argmax_out);
    KLAUNCH();
}

}  // namespace kestrel
