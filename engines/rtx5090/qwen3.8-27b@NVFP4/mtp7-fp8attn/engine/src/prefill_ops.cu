#include "prefill_ops.cuh"

#include "gdn_wy.cuh"

#include <cuda_fp8.h>

namespace kestrel {
namespace kpre {

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
__device__ __forceinline__ float silu(float v) { return v / (1.f + expf(-v)); }
__device__ __forceinline__ float sigmoidf_(float v) { return 1.f / (1.f + expf(-v)); }
__device__ __forceinline__ std::uint8_t to_e4m3(float v) {
    return static_cast<std::uint8_t>(__nv_cvt_float_to_fp8(v, __NV_SATFINITE, __NV_E4M3));
}
__device__ __forceinline__ float from_e4m3(std::uint8_t c) {
    const __half_raw h = __nv_cvt_fp8_to_halfraw(static_cast<__nv_fp8_storage_t>(c), __NV_E4M3);
    return __half2float(*reinterpret_cast<const __half*>(&h));
}
// Two values -> one byte of E2M1 codes, `lo` in the low nibble.
__device__ __forceinline__ std::uint32_t e2m1_pair(float lo, float hi) {
    std::uint32_t r;
    asm("{ .reg .b8 t; cvt.rn.satfinite.e2m1x2.f32 t, %1, %2; cvt.u32.u8 %0, t; }" : "=r"(r) : "f"(hi), "f"(lo));
    return r;
}

// ------------------------------------------------------------------------------------------------ norms / quant
constexpr int kRowThreads = 640;  // 5120 / 8

__global__ void __launch_bounds__(kRowThreads) rmsnorm_quant_fp8_kernel(const float* __restrict__ x, const float* __restrict__ w,
                                                                        int K, std::uint8_t* __restrict__ codes,
                                                                        float* __restrict__ scale, __nv_bfloat16* __restrict__ bf) {
    pdl_wait();
    pdl_trigger();
    __shared__ float red[32];
    const int r = blockIdx.x;
    const float* xr = x + static_cast<std::size_t>(r) * K;
    const int per = K / blockDim.x;  // 8
    float v[8];
    float ss = 0.f;
    for(int i = 0; i < per; ++i) {
        v[i] = xr[threadIdx.x * per + i];
        ss += v[i] * v[i];
    }
    ss = block_sum(ss, red);
    const float rs = rsqrtf(ss / K + kEps);
    float am = 0.f;
    for(int i = 0; i < per; ++i) {
        v[i] = v[i] * rs * (1.f + w[threadIdx.x * per + i]);
        am = fmaxf(am, fabsf(v[i]));
    }
    am = block_max(am, red);
    const float sc = am > 0.f ? am / 448.f : 1.f;
    for(int i = 0; i < per; ++i) {
        const std::size_t o = static_cast<std::size_t>(r) * K + threadIdx.x * per + i;
        codes[o] = to_e4m3(v[i] / sc);
        if(bf) bf[o] = __float2bfloat16_rn(v[i]);
    }
    if(threadIdx.x == 0) scale[r] = sc;
}

__global__ void __launch_bounds__(kRowThreads) rmsnorm_bf16_amax_kernel(const float* __restrict__ x, const float* __restrict__ w,
                                                                        int K, __nv_bfloat16* __restrict__ out, unsigned* amax) {
    pdl_wait();
    pdl_trigger();
    __shared__ float red[32];
    const int r = blockIdx.x;
    const float* xr = x + static_cast<std::size_t>(r) * K;
    const int per = K / blockDim.x;
    float v[8];
    float ss = 0.f;
    for(int i = 0; i < per; ++i) {
        v[i] = xr[threadIdx.x * per + i];
        ss += v[i] * v[i];
    }
    ss = block_sum(ss, red);
    const float rs = rsqrtf(ss / K + kEps);
    float am = 0.f;
    for(int i = 0; i < per; ++i) {
        const __nv_bfloat16 b = __float2bfloat16_rn(v[i] * rs * (1.f + w[threadIdx.x * per + i]));
        out[static_cast<std::size_t>(r) * K + threadIdx.x * per + i] = b;
        am = fmaxf(am, fabsf(__bfloat162float(b)));
    }
    am = block_max(am, red);
    if(threadIdx.x == 0) atomicMax(amax, __float_as_uint(am));
}

__device__ __forceinline__ float tof(__nv_bfloat16 v) { return __bfloat162float(v); }
__device__ __forceinline__ float tof(__half v) { return __half2float(v); }
template <typename In>
__global__ void quant_fp8_rows_kernel(const In* __restrict__ in, int K, std::uint8_t* __restrict__ codes,
                                      float* __restrict__ scale) {
    pdl_wait();
    pdl_trigger();
    __shared__ float red[32];
    const int r = blockIdx.x;
    const In* row = in + static_cast<std::size_t>(r) * K;
    float am = 0.f;
    for(int i = threadIdx.x; i < K; i += blockDim.x) am = fmaxf(am, fabsf(tof(row[i])));
    am = block_max(am, red);
    const float sc = am > 0.f ? am / 448.f : 1.f;
    for(int i = threadIdx.x; i < K; i += blockDim.x)
        codes[static_cast<std::size_t>(r) * K + i] = to_e4m3(tof(row[i]) / sc);
    if(threadIdx.x == 0) scale[r] = sc;
}

// One thread per 16-element block.
__global__ void quant_fp4_kernel(const __nv_bfloat16* __restrict__ in, int T, int K, const unsigned* __restrict__ amax,
                                 float w_alpha, std::uint8_t* __restrict__ packed, std::uint8_t* __restrict__ sf,
                                 float* __restrict__ alpha_out) {
    pdl_wait();
    pdl_trigger();
    const int blocks = K / 16;
    const int r = blockIdx.y;
    const int b = blockIdx.x * blockDim.x + threadIdx.x;
    if(b >= blocks) return;
    const float am = __uint_as_float(*amax);
    const float gs = am > 0.f ? 448.f * 6.f / am : 1.f;
    if(r == 0 && b == 0) *alpha_out = w_alpha / gs;
    float v[16];
    float bm = 0.f;
    if(r < T) {
        const uint4* src = reinterpret_cast<const uint4*>(in + static_cast<std::size_t>(r) * K + b * 16);
        const uint4 u0 = src[0], u1 = src[1];
        const __nv_bfloat162* p0 = reinterpret_cast<const __nv_bfloat162*>(&u0);
        const __nv_bfloat162* p1 = reinterpret_cast<const __nv_bfloat162*>(&u1);
#pragma unroll
        for(int i = 0; i < 4; ++i) {
            const float2 a = __bfloat1622float2(p0[i]), c = __bfloat1622float2(p1[i]);
            v[2 * i] = a.x; v[2 * i + 1] = a.y; v[8 + 2 * i] = c.x; v[8 + 2 * i + 1] = c.y;
        }
#pragma unroll
        for(int i = 0; i < 16; ++i) bm = fmaxf(bm, fabsf(v[i]));
    } else {
#pragma unroll
        for(int i = 0; i < 16; ++i) v[i] = 0.f;
    }
    const std::uint8_t code = to_e4m3(bm / 6.f * gs);
    const float sdec = from_e4m3(code);
    const float inv = sdec > 0.f ? gs / sdec : 0.f;
    std::uint32_t lo = 0, hi = 0;
#pragma unroll
    for(int i = 0; i < 4; ++i) lo |= e2m1_pair(v[2 * i] * inv, v[2 * i + 1] * inv) << (8 * i);
#pragma unroll
    for(int i = 0; i < 4; ++i) hi |= e2m1_pair(v[8 + 2 * i] * inv, v[8 + 2 * i + 1] * inv) << (8 * i);
    *reinterpret_cast<uint2*>(packed + static_cast<std::size_t>(r) * (K / 2) + b * 8) = make_uint2(lo, hi);
    sf[sf_offset(r, b, sf_blocks_padded(K))] = code;
}

__global__ void __launch_bounds__(kRowThreads) residual_norm_kernel(Proj p, float* __restrict__ hidden, const float* __restrict__ w,
                                                                    std::uint8_t* __restrict__ codes, float* __restrict__ scale,
                                                                    __nv_bfloat16* __restrict__ bf, unsigned* amax) {
    pdl_wait();
    pdl_trigger();
    __shared__ float red[32];
    const int r = blockIdx.x;
    float* hr = hidden + static_cast<std::size_t>(r) * kHidden;
    const int c0 = threadIdx.x * 8;
    float v[8];
    float ss = 0.f;
    {
        float pv[8];
        p.at8(r, c0, pv);
        float4* h4 = reinterpret_cast<float4*>(hr + c0);
        const float4 a = h4[0], b = h4[1];
        v[0] = a.x + pv[0]; v[1] = a.y + pv[1]; v[2] = a.z + pv[2]; v[3] = a.w + pv[3];
        v[4] = b.x + pv[4]; v[5] = b.y + pv[5]; v[6] = b.z + pv[6]; v[7] = b.w + pv[7];
        h4[0] = make_float4(v[0], v[1], v[2], v[3]);
        h4[1] = make_float4(v[4], v[5], v[6], v[7]);
#pragma unroll
        for(int i = 0; i < 8; ++i) ss += v[i] * v[i];
    }
    ss = block_sum(ss, red);
    const float rs = rsqrtf(ss / kHidden + kEps);
    float am = 0.f;
#pragma unroll
    for(int i = 0; i < 8; ++i) {
        v[i] = v[i] * rs * (1.f + w[c0 + i]);
        if(!codes) v[i] = __bfloat162float(__float2bfloat16_rn(v[i]));
        am = fmaxf(am, fabsf(v[i]));
    }
    am = block_max(am, red);
    if(codes) {
        const float sc = am > 0.f ? am / 448.f : 1.f;
#pragma unroll
        for(int i = 0; i < 8; ++i) {
            const std::size_t o = static_cast<std::size_t>(r) * kHidden + c0 + i;
            codes[o] = to_e4m3(v[i] / sc);
            if(bf) bf[o] = __float2bfloat16_rn(v[i]);
        }
        if(threadIdx.x == 0) scale[r] = sc;
    } else {
#pragma unroll
        for(int i = 0; i < 8; ++i) bf[static_cast<std::size_t>(r) * kHidden + c0 + i] = __float2bfloat16_rn(v[i]);
        if(threadIdx.x == 0) atomicMax(amax, __float_as_uint(am));
    }
}

// 16 values -> 8 bytes of E2M1 codes with block scale code (static global scale gs).
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

// Row kernel, 640 threads x 8 values; thread pairs form a 16-element block.
__global__ void __launch_bounds__(kRowThreads) residual_norm_fp4_kernel(Proj p, int T, float* __restrict__ hidden,
                                                                        const float* __restrict__ w, float gs,
                                                                        std::uint8_t* __restrict__ packed, std::uint8_t* __restrict__ sf) {
    pdl_wait();
    pdl_trigger();
    __shared__ float red[32];
    const int r = blockIdx.x, c0 = threadIdx.x * 8;
    const bool live = r < T;
    float v[8];
    float ss = 0.f;
    if(live) {
        float* hr = hidden + static_cast<std::size_t>(r) * kHidden;
        const bool add = p.f32 || p.bf;
        float pv[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
        if(add) p.at8(r, c0, pv);
        float4* h4 = reinterpret_cast<float4*>(hr + c0);
        const float4 a = h4[0], b = h4[1];
        v[0] = a.x + pv[0]; v[1] = a.y + pv[1]; v[2] = a.z + pv[2]; v[3] = a.w + pv[3];
        v[4] = b.x + pv[4]; v[5] = b.y + pv[5]; v[6] = b.z + pv[6]; v[7] = b.w + pv[7];
        if(add) {
            h4[0] = make_float4(v[0], v[1], v[2], v[3]);
            h4[1] = make_float4(v[4], v[5], v[6], v[7]);
        }
#pragma unroll
        for(int i = 0; i < 8; ++i) ss += v[i] * v[i];
    } else {
#pragma unroll
        for(int i = 0; i < 8; ++i) v[i] = 0.f;
    }
    ss = block_sum(ss, red);
    const float rs = rsqrtf(ss / kHidden + kEps);
    float bm = 0.f;
#pragma unroll
    for(int i = 0; i < 8; ++i) {
        v[i] = live ? v[i] * rs * (1.f + w[c0 + i]) : 0.f;
        bm = fmaxf(bm, fabsf(v[i]));
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
        *reinterpret_cast<uint2*>(packed + static_cast<std::size_t>(r) * (kHidden / 2) + b * 8) = codes;
        sf[sf_offset(r, b, sf_blocks_padded(kHidden))] = live ? scode : 0;
    }
}

// One thread per 16-element block of the SwiGLU output.
__global__ void swiglu_fp4_kernel(Proj p, int T, float gs, std::uint8_t* __restrict__ packed, std::uint8_t* __restrict__ sf) {
    pdl_wait();
    pdl_trigger();
    const int r = blockIdx.y;
    const int b = blockIdx.x * blockDim.x + threadIdx.x;
    if(b >= kInter / 16) return;
    float v[16], bm = 0.f;
    if(r < T) {
#pragma unroll
        for(int half = 0; half < 2; ++half) {
            const int blk8 = 2 * b + half;
            float gt[8], up[8];
            p.at8(r, blk8 * 16, gt);
            p.at8(r, blk8 * 16 + 8, up);
#pragma unroll
            for(int j = 0; j < 8; ++j) v[half * 8 + j] = silu(gt[j]) * up[j];
        }
    } else {
#pragma unroll
        for(int i = 0; i < 16; ++i) v[i] = 0.f;
    }
#pragma unroll
    for(int i = 0; i < 16; ++i) {
        // Round through BF16 like the dynamic path's materialized intermediate.
        v[i] = __bfloat162float(__float2bfloat16_rn(v[i]));
        bm = fmaxf(bm, fabsf(v[i]));
    }
    uint2 codes;
    std::uint8_t scode;
    fp4_block(v, bm, gs, codes, scode);
    *reinterpret_cast<uint2*>(packed + static_cast<std::size_t>(r) * (kInter / 2) + b * 8) = codes;
    sf[sf_offset(r, b, sf_blocks_padded(kInter))] = r < T ? scode : 0;
}

__global__ void residual_add_kernel(Proj p, int T, float* __restrict__ hidden) {
    pdl_wait();
    pdl_trigger();
    const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if(idx >= static_cast<std::size_t>(T) * kHidden) return;
    const int r = static_cast<int>(idx / kHidden), c = static_cast<int>(idx % kHidden);
    hidden[idx] += p.at(r, c);
}

__global__ void swiglu_kernel(Proj p, int T, __nv_bfloat16* __restrict__ out, unsigned* amax) {
    pdl_wait();
    pdl_trigger();
    __shared__ float red[32];
    const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    float am = 0.f;
    if(idx < static_cast<std::size_t>(T) * kInter) {
        const int r = static_cast<int>(idx / kInter), i = static_cast<int>(idx % kInter);
        const int b = i >> 3, j = i & 7;
        const float g = p.at(r, b * 16 + j), u = p.at(r, b * 16 + 8 + j);
        const __nv_bfloat16 y = __float2bfloat16_rn(silu(g) * u);
        out[idx] = y;
        am = fabsf(__bfloat162float(y));
    }
    am = block_max(am, red);
    if(threadIdx.x == 0) atomicMax(amax, __float_as_uint(am));
}

constexpr int kWide = 768;  // 6144 / 8

__global__ void __launch_bounds__(kWide) attn_gate_quant_kernel(const float* __restrict__ attn, Proj qkv,
                                                                std::uint8_t* __restrict__ codes, float* __restrict__ scale) {
    pdl_wait();
    pdl_trigger();
    __shared__ float red[32];
    const int r = blockIdx.x;
    float v[8];
    float am = 0.f;
    {
        const int c = threadIdx.x * 8, h = c / kHeadDim, d = c % kHeadDim;
        float gate[8];
        qkv.at8(r, h * 2 * kHeadDim + kHeadDim + d, gate);
        const float4* ap = reinterpret_cast<const float4*>(attn + static_cast<std::size_t>(r) * kAttnOut + c);
        const float4 a0 = ap[0], a1 = ap[1];
        const float av[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
#pragma unroll
        for(int i = 0; i < 8; ++i) {
            v[i] = av[i] * sigmoidf_(gate[i]);
            am = fmaxf(am, fabsf(v[i]));
        }
    }
    am = block_max(am, red);
    const float sc = am > 0.f ? am / 448.f : 1.f;
#pragma unroll
    for(int i = 0; i < 8; ++i) codes[static_cast<std::size_t>(r) * kAttnOut + threadIdx.x * 8 + i] = to_e4m3(v[i] / sc);
    if(threadIdx.x == 0) scale[r] = sc;
}

// ------------------------------------------------------------------------------------------------ gated DeltaNet
constexpr int kConvThreads = 640;

// Block (token t, key head kh): conv over the window from the committed state, SiLU, q/k L2 norm, a/b gates.
__global__ void __launch_bounds__(kConvThreads) gdn_prefill_conv_kernel(GdnLayer L, GdnState st, GdnPrefillScratch sc,
                                                                        Proj qkvz, const float* __restrict__ abp) {
    pdl_wait();
    pdl_trigger();
    __shared__ float red[32];
    const int t = blockIdx.x, kh = blockIdx.y, tid = threadIdx.x;
    int ch;
    if(tid < 128) ch = kh * 128 + tid;
    else if(tid < 256) ch = 2048 + kh * 128 + (tid - 128);
    else ch = 4096 + kh * 384 + (tid - 256);
    float win[4];
#pragma unroll
    for(int j = 0; j < 4; ++j) {
        const int tt = t - 3 + j;
        win[j] = tt >= 0 ? qkvz.at(tt, ch) : st.conv[(3 + tt) * kLinQkv + ch];
    }
    const float4 cw = *reinterpret_cast<const float4*>(L.conv_w + ch * 4);
    float y = cw.x * win[0];
    y = fmaf(cw.y, win[1], y);
    y = fmaf(cw.z, win[2], y);
    y = fmaf(cw.w, win[3], y);
    y = silu(y);
    const float tq = block_sum(tid < 128 ? y * y : 0.f, red);
    const float tk = block_sum((tid >= 128 && tid < 256) ? y * y : 0.f, red);
    if(tid < 128) {
        sc.qk[((static_cast<std::size_t>(t) * kLinKHeads + kh) * 2 + 0) * kLinDim + tid] =
            y * rsqrtf(tq + kEps) * rsqrtf(static_cast<float>(kLinDim));
    } else if(tid < 256) {
        sc.qk[((static_cast<std::size_t>(t) * kLinKHeads + kh) * 2 + 1) * kLinDim + (tid - 128)] = y * rsqrtf(tk + kEps);
    } else {
        const int j = (tid - 256) / 128, i = (tid - 256) % 128;
        sc.v[(static_cast<std::size_t>(t) * kLinVHeads + kh * 3 + j) * kLinDim + i] = y;
    }
    if(tid < 3) {
        const int h = kh * 3 + tid;
        const float a = abp[static_cast<std::size_t>(t) * 2 * kLinVHeads + h];
        const float b = abp[static_cast<std::size_t>(t) * 2 * kLinVHeads + kLinVHeads + h];
        const float x = a + L.dt_bias[h];
        const float sp = x > 20.f ? x : log1pf(expf(x));
        sc.gb[(static_cast<std::size_t>(t) * kLinVHeads + h) * 2 + 0] = L.neg_a[h] * sp;  // log decay
        sc.gb[(static_cast<std::size_t>(t) * kLinVHeads + h) * 2 + 1] = sigmoidf_(b);
    }
}


constexpr int kConvTile = 32;
// Block (key head kh, 32-token tile): thread = channel (q 128 | k 128 | v of the 3 value heads 384).
__global__ void __launch_bounds__(kConvThreads) gdn_conv_tile_kernel(GdnLayer L, GdnState st, GdnPrefillScratch sc, Proj qkvz,
                                                                     const float* __restrict__ abp, int T) {
    pdl_wait();
    pdl_trigger();
    __shared__ float yq[kConvTile][kLinDim + 1], yk[kConvTile][kLinDim + 1];
    __shared__ float inv[kConvTile][2];
    const int kh = blockIdx.x, t0 = blockIdx.y * kConvTile, tid = threadIdx.x;
    const int n = min(kConvTile, T - t0);
    int ch;
    if(tid < 128) ch = kh * 128 + tid;
    else if(tid < 256) ch = 2048 + kh * 128 + (tid - 128);
    else ch = 4096 + kh * 384 + (tid - 256);
    float w0, w1, w2;
    {
        float w[3];
#pragma unroll
        for(int j = 0; j < 3; ++j) {
            const int tt = t0 - 3 + j;
            w[j] = tt >= 0 ? qkvz.at(tt, ch) : st.conv[(3 + tt) * kLinQkv + ch];
        }
        w0 = w[0]; w1 = w[1]; w2 = w[2];
    }
    const float4 cw = *reinterpret_cast<const float4*>(L.conv_w + ch * 4);
    for(int i = 0; i < n; ++i) {
        const float x = qkvz.at(t0 + i, ch);
        float y = cw.x * w0;
        y = fmaf(cw.y, w1, y);
        y = fmaf(cw.z, w2, y);
        y = fmaf(cw.w, x, y);
        w0 = w1; w1 = w2; w2 = x;
        y = silu(y);
        if(tid < 128) yq[i][tid] = y;
        else if(tid < 256) yk[i][tid - 128] = y;
        else {
            const int j = (tid - 256) / 128, c = (tid - 256) % 128;
            sc.v[(static_cast<std::size_t>(t0 + i) * kLinVHeads + kh * 3 + j) * kLinDim + c] = y;
        }
    }
    __syncthreads();
    // Inverse L2 norms: one warp per (token, q|k).
    const int lane = tid & 31, warp = tid >> 5;
    for(int job = warp; job < 2 * n; job += kConvThreads / 32) {
        const int i = job >> 1;
        const float* row = (job & 1) ? yk[i] : yq[i];
        float ss = 0.f;
#pragma unroll
        for(int c = lane; c < kLinDim; c += 32) ss += row[c] * row[c];
        ss = warp_sum(ss);
        if(lane == 0) inv[i][job & 1] = rsqrtf(ss + kEps);
    }
    __syncthreads();
    if(tid < 256) {
        const int c = tid & 127, which = tid >> 7;
        const float extra = which ? 1.f : rsqrtf(static_cast<float>(kLinDim));
        for(int i = 0; i < n; ++i) {
            const float y = which ? yk[i][c] : yq[i][c];
            sc.qk[((static_cast<std::size_t>(t0 + i) * kLinKHeads + kh) * 2 + which) * kLinDim + c] = y * inv[i][which] * extra;
        }
    } else if(tid < 256 + 3 * kConvTile) {
        const int e = tid - 256, i = e / 3, h = kh * 3 + e % 3, t = t0 + i;
        if(i < n) {
            const float a = abp[static_cast<std::size_t>(t) * 2 * kLinVHeads + h];
            const float b = abp[static_cast<std::size_t>(t) * 2 * kLinVHeads + kLinVHeads + h];
            const float x = a + L.dt_bias[h];
            const float sp = x > 20.f ? x : log1pf(expf(x));
            sc.gb[(static_cast<std::size_t>(t) * kLinVHeads + h) * 2 + 0] = L.neg_a[h] * sp;  // log decay
            sc.gb[(static_cast<std::size_t>(t) * kLinVHeads + h) * 2 + 1] = sigmoidf_(b);
        }
    }
}

// New committed window: the last three inputs of [old window, x_0 .. x_{T-1}].
__global__ void gdn_window_kernel(GdnState st, Proj qkvz, int T) {
    pdl_wait();
    pdl_trigger();
    const int ch = blockIdx.x * blockDim.x + threadIdx.x;
    if(ch >= kLinQkv) return;
    float w[3];
#pragma unroll
    for(int j = 0; j < 3; ++j) {
        const int tt = T - 3 + j;
        w[j] = tt >= 0 ? qkvz.at(tt, ch) : st.conv[(3 + tt) * kLinQkv + ch];
    }
#pragma unroll
    for(int j = 0; j < 3; ++j) st.conv[j * kLinQkv + ch] = w[j];
}

// Block (value head, 64-column slice); thread (v, key eighth) holds 16 state entries. Sequential over tokens, with
// 16-token tiles of normalized q / k (rows padded to 36 floats per key quarter), the slice's v and the gates staged in
// shared memory by cp.async, double buffered.
constexpr int kScanTile = 16;
constexpr int kScanRow = 20;  // floats per key eighth row (16 + 4 padding: conflict-free float4 reads)
struct ScanStage {
    float k[kScanTile][8][kScanRow];
    float q[kScanTile][8][kScanRow];
    float v[kScanTile][64];
    float gb[kScanTile][2];
};
__device__ __forceinline__ void cp16(void* dst, const void* src) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(static_cast<unsigned>(__cvta_generic_to_shared(dst))), "l"(src));
}
__device__ __forceinline__ void cp8(void* dst, const void* src) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 8;" ::"r"(static_cast<unsigned>(__cvta_generic_to_shared(dst))), "l"(src));
}

__global__ void __launch_bounds__(512) gdn_scan_kernel(GdnState st, GdnPrefillScratch sc, int T) {
    pdl_wait();
    pdl_trigger();
    extern __shared__ __align__(16) unsigned char scan_smem[];
    ScanStage* stage = reinterpret_cast<ScanStage*>(scan_smem);
    const int h = blockIdx.x, slice = blockIdx.y, kh = h / 3, tid = threadIdx.x, nt = blockDim.x;
    // Slices: 2 x 64 columns, or 48 + 40 + 40 (gridDim.y == 3).
    const int cs = gridDim.y == 3 ? (slice == 0 ? 0 : 8 + 40 * slice) : slice * 64;
    const int cw = gridDim.y == 3 ? (slice == 0 ? 48 : 40) : 64;
    const int vl = tid >> 3, kg = tid & 7;
    const bool active = vl < cw;
    const int v = cs + min(vl, cw - 1);
    auto load_tile = [&](int tile, ScanStage& S) {
        const int t0 = tile * kScanTile;
        // q and k: per token 2 x 32 chunks of 16 bytes; v: 8 chunks; gates: one 8-byte copy.
        const int per = 64 + (cw + 3) / 4;
        for(int e = tid; e < kScanTile * per; e += nt) {
            const int tt = e / per, r = e % per, t = min(t0 + tt, T - 1);
            if(r < 64) {
                const int which = r / 32, c = r % 32, quarter = c / 4, part = c % 4;
                const float* src = sc.qk + ((static_cast<std::size_t>(t) * kLinKHeads + kh) * 2 + (which ? 0 : 1)) * kLinDim + quarter * 16 + part * 4;
                cp16(&(which ? S.q : S.k)[tt][quarter][part * 4], src);
            } else {
                const int c = r - 64;
                cp16(&S.v[tt][c * 4], sc.v + (static_cast<std::size_t>(t) * kLinVHeads + h) * kLinDim + cs + c * 4);
            }
        }
        for(int tt = tid; tt < kScanTile; tt += nt) {
            const int t = min(t0 + tt, T - 1);
            cp8(&S.gb[tt][0], sc.gb + (static_cast<std::size_t>(t) * kLinVHeads + h) * 2);
        }
        asm volatile("cp.async.commit_group;");
    };
    // State layout: [head][v * 4 + key quarter][32]; this thread holds half of a quarter.
    float S[16];
    float4* sp = reinterpret_cast<float4*>(st.S + (static_cast<std::size_t>(h) * 512 + v * 4 + (kg >> 1)) * 32 + (kg & 1) * 16);
#pragma unroll
    for(int i = 0; i < 4; ++i) {
        const float4 t4 = sp[i];
        S[4 * i] = t4.x; S[4 * i + 1] = t4.y; S[4 * i + 2] = t4.z; S[4 * i + 3] = t4.w;
    }
    const int tiles = (T + kScanTile - 1) / kScanTile;
    load_tile(0, stage[0]);
    for(int j = 0; j < tiles; ++j) {
        if(j + 1 < tiles) {
            load_tile(j + 1, stage[(j + 1) & 1]);
            asm volatile("cp.async.wait_group 1;" ::: "memory");
        } else {
            asm volatile("cp.async.wait_group 0;" ::: "memory");
        }
        __syncthreads();
        const ScanStage& G = stage[j & 1];
        const int n = min(kScanTile, T - j * kScanTile);
#pragma unroll 2
        for(int tt = 0; tt < n; ++tt) {
            const float decay = __expf(G.gb[tt][0]), beta = G.gb[tt][1];
            const float vv = G.v[tt][vl];
            const float4* kp = reinterpret_cast<const float4*>(&G.k[tt][kg][0]);
            const float4* qp = reinterpret_cast<const float4*>(&G.q[tt][kg][0]);
            float kk[16];
#pragma unroll
            for(int i = 0; i < 4; ++i) {
                const float4 k4 = kp[i];
                kk[4 * i] = k4.x; kk[4 * i + 1] = k4.y; kk[4 * i + 2] = k4.z; kk[4 * i + 3] = k4.w;
            }
            float kp4[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
            for(int i = 0; i < 16; ++i) {
                S[i] *= decay;
                kp4[i & 3] = fmaf(S[i], kk[i], kp4[i & 3]);
            }
            float kv = (kp4[0] + kp4[1]) + (kp4[2] + kp4[3]);
            kv += __shfl_xor_sync(0xffffffffu, kv, 1);
            kv += __shfl_xor_sync(0xffffffffu, kv, 2);
            kv += __shfl_xor_sync(0xffffffffu, kv, 4);
            const float delta = (vv - kv) * beta;
            float op4[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
            for(int i = 0; i < 4; ++i) {
                const float4 q4 = qp[i];
                S[4 * i] = fmaf(kk[4 * i], delta, S[4 * i]);
                S[4 * i + 1] = fmaf(kk[4 * i + 1], delta, S[4 * i + 1]);
                S[4 * i + 2] = fmaf(kk[4 * i + 2], delta, S[4 * i + 2]);
                S[4 * i + 3] = fmaf(kk[4 * i + 3], delta, S[4 * i + 3]);
                op4[0] = fmaf(S[4 * i], q4.x, op4[0]);
                op4[1] = fmaf(S[4 * i + 1], q4.y, op4[1]);
                op4[2] = fmaf(S[4 * i + 2], q4.z, op4[2]);
                op4[3] = fmaf(S[4 * i + 3], q4.w, op4[3]);
            }
            float o = (op4[0] + op4[1]) + (op4[2] + op4[3]);
            o += __shfl_xor_sync(0xffffffffu, o, 1);
            o += __shfl_xor_sync(0xffffffffu, o, 2);
            o += __shfl_xor_sync(0xffffffffu, o, 4);
            if(kg == 0 && active) sc.o[(static_cast<std::size_t>(j * kScanTile + tt) * kLinVHeads + h) * kLinDim + v] = o;
        }
        __syncthreads();
    }
    if(active)
#pragma unroll
        for(int i = 0; i < 4; ++i) sp[i] = make_float4(S[4 * i], S[4 * i + 1], S[4 * i + 2], S[4 * i + 3]);
}

// Per token: gated RMSNorm per head (16 threads x 8 values per head) times SiLU(z), then FP8 per-row quantization.
__global__ void __launch_bounds__(kWide) gdn_out_kernel(GdnLayer L, GdnPrefillScratch sc, Proj qkvz,
                                                        std::uint8_t* __restrict__ codes, float* __restrict__ scale) {
    pdl_wait();
    pdl_trigger();
    __shared__ float red[32];
    const int r = blockIdx.x, c0 = threadIdx.x * 8;
    float o[8];
    float ss = 0.f;
#pragma unroll
    for(int i = 0; i < 8; ++i) {
        o[i] = sc.o[static_cast<std::size_t>(r) * kLinZ + c0 + i];
        ss += o[i] * o[i];
    }
#pragma unroll
    for(int m = 8; m; m >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, m);
    const float rs = rsqrtf(ss / kLinDim + kEps);
    float am = 0.f;
    float zv[8];
    qkvz.at8(r, kLinQkv + c0, zv);
#pragma unroll
    for(int i = 0; i < 8; ++i) {
        const int c = c0 + i;
        o[i] = L.norm_w[c % kLinDim] * (o[i] * rs) * silu(zv[i]);
        am = fmaxf(am, fabsf(o[i]));
    }
    am = block_max(am, red);
    const float sc_ = am > 0.f ? am / 448.f : 1.f;
#pragma unroll
    for(int i = 0; i < 8; ++i) codes[static_cast<std::size_t>(r) * kLinZ + c0 + i] = to_e4m3(o[i] / sc_);
    if(threadIdx.x == 0) scale[r] = sc_;
}

}  // namespace kpre
using namespace kpre;

void rmsnorm_quant_fp8(const float* x, const float* w, int T, int K, std::uint8_t* codes, float* scale,
                       __nv_bfloat16* bf_out, cudaStream_t s) {
    if(K != kHidden) throw std::invalid_argument("rmsnorm_quant_fp8: K must be 5120");
    launch_k(rmsnorm_quant_fp8_kernel, T, kRowThreads, 0, s, x, w, K, codes, scale, bf_out);
    KLAUNCH();
}
void rmsnorm_bf16_amax(const float* x, const float* w, int T, int K, __nv_bfloat16* out, unsigned* amax, cudaStream_t s) {
    if(K != kHidden) throw std::invalid_argument("rmsnorm_bf16_amax: K must be 5120");
    launch_k(rmsnorm_bf16_amax_kernel, T, kRowThreads, 0, s, x, w, K, out, amax);
    KLAUNCH();
}
void quant_fp8_rows(const __nv_bfloat16* in, int T, int K, std::uint8_t* codes, float* scale, cudaStream_t s) {
    launch_k(quant_fp8_rows_kernel<__nv_bfloat16>, T, 1024, 0, s, in, K, codes, scale);
    KLAUNCH();
}
void quant_fp8_rows(const __half* in, int T, int K, std::uint8_t* codes, float* scale, cudaStream_t s) {
    launch_k(quant_fp8_rows_kernel<__half>, T, 1024, 0, s, in, K, codes, scale);
    KLAUNCH();
}
void quant_fp4(const __nv_bfloat16* in, int T, int T_pad, int K, const unsigned* amax, float w_alpha, std::uint8_t* packed,
               std::uint8_t* sf, float* alpha_out, cudaStream_t s) {
    const int blocks = K / 16;
    launch_k(quant_fp4_kernel, dim3((blocks + 127) / 128, T_pad), 128, 0, s, in, T, K, amax, w_alpha, packed, sf, alpha_out);
    KLAUNCH();
}
void residual_norm(Proj p, int T, float* hidden, const float* w, std::uint8_t* codes, float* scale, __nv_bfloat16* bf_out,
                   unsigned* amax, cudaStream_t s) {
    launch_k(residual_norm_kernel, T, kRowThreads, 0, s, p, hidden, w, codes, scale, bf_out, amax);
}
void residual_norm_fp4(Proj p, int T, int T_pad, float* hidden, const float* w, float gs, std::uint8_t* packed,
                       std::uint8_t* sf, cudaStream_t s) {
    launch_k(residual_norm_fp4_kernel, T_pad, kRowThreads, 0, s, p, T, hidden, w, gs, packed, sf);
}
void swiglu_fp4(Proj p, int T, int T_pad, float gs, std::uint8_t* packed, std::uint8_t* sf, cudaStream_t s) {
    launch_k(swiglu_fp4_kernel, dim3((kInter / 16 + 127) / 128, T_pad), 128, 0, s, p, T, gs, packed, sf);
}
void residual_add(Proj p, int T, float* hidden, cudaStream_t s) {
    const std::size_t total = static_cast<std::size_t>(T) * kHidden;
    launch_k(residual_add_kernel, static_cast<unsigned>((total + 255) / 256), 256, 0, s, p, T, hidden);
    KLAUNCH();
}
void swiglu_bf16(Proj p, int T, __nv_bfloat16* out, unsigned* amax, cudaStream_t s) {
    const std::size_t total = static_cast<std::size_t>(T) * kInter;
    launch_k(swiglu_kernel, static_cast<unsigned>((total + 255) / 256), 256, 0, s, p, T, out, amax);
    KLAUNCH();
}
void attn_gate_quant(const float* attn, Proj qkv, int T, std::uint8_t* codes, float* scale, cudaStream_t s) {
    launch_k(attn_gate_quant_kernel, T, kWide, 0, s, attn, qkv, codes, scale);
    KLAUNCH();
}

// Scan slicing (set once at engine creation; same for every engine in the process).
static bool scan3_ = true;
void set_scan3(bool on) { scan3_ = on; }

void gdn_prefill(const GdnLayer& L, GdnState st, GdnPrefillScratch sc, Proj qkvz, const float* ab, int T,
                 std::uint8_t* codes, float* scale, const WyScratch* wy, cudaStream_t s) {
    launch_k(gdn_conv_tile_kernel, dim3(kLinKHeads, (T + kConvTile - 1) / kConvTile), kConvThreads, 0, s, L, st, sc, qkvz, ab, T);
    KLAUNCH();
    launch_k(gdn_window_kernel, (kLinQkv + 255) / 256, 256, 0, s, st, qkvz, T);
    KLAUNCH();
    if(wy) {
        gdn_scan_wy_tc(st, sc, T, *wy, s);
    } else {
        KCHECK(cudaFuncSetAttribute(gdn_scan_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 2 * sizeof(ScanStage)));
        if(scan3_) launch_k(gdn_scan_kernel, dim3(kLinVHeads, 3), 384, 2 * sizeof(ScanStage), s, st, sc, T);
        else launch_k(gdn_scan_kernel, dim3(kLinVHeads, 2), 512, 2 * sizeof(ScanStage), s, st, sc, T);
    }
    KLAUNCH();
    launch_k(gdn_out_kernel, T, kWide, 0, s, L, sc, qkvz, codes, scale);
    KLAUNCH();
}

}  // namespace kestrel
